"""Stress test of the generator, the evidence and the grader on extreme configurations.

    python -m auditenv.run stress --n 500          # make stress; writes artifacts/stress.json

Each episode gets a configuration drawn from a matrix of edge values (base currency incl. JPY with
zero decimals, timezones from -12:00 to +14:00 and +05:45, minimum and maximum company sizes,
12 errors with both interaction scenarios, interaction-only books, clean books, every distractor
at its maximum, all or no foreign invoices) and is checked end to end:

  1. generation succeeds and the size contract holds (200-2000 rows, 20-200 documents, 0-12 issues);
  2. the independent validator finds no problem (errors recomputed from public files, evidence
     sufficiency, pre-error state reconstructed from provenance);
  3. the ledger balances per currency (sum of debits == sum of credits);
  4. every issue has at least one evidence set whose EVERY id the agent can fetch with its own tools
     (get_document for doc ids, get_record for ledger rows and fx ids), and every affected id is a
     record the agent can fetch;
  5. the grader-only oracle scores F1 = 1.0 and reward = 1.0 through the real env (eval mode);
  6. a reconcile run in eval mode leaks nothing: no issue id, truth key, provenance key or private
     path in any observation, error or info, and no reward or diagnostics in info;
  7. every 10th episode: a renamed + shuffled twin still validates and the oracle keeps reward 1.0;
  8. timings (generation, validation, oracle, reconcile) and peak memory.

A configuration the generator cannot satisfy must be REJECTED by the config check (ConfigError
with the reason and numbers) instead of failing generation: rejections are counted separately.
A configuration that passes the check and then fails generation is a failure.

Any failure is listed with its seed and configuration, so it can be reproduced exactly.
"""

from __future__ import annotations

import json
import random
import re
import resource
import shutil
import statistics
import time
from typing import Any, cast
from pathlib import Path

from auditenv.common import read_json
from auditenv.env.core import AuditEnv
from auditenv.env.tools import READ_TOOLS, Records, ToolError
from auditenv.generator import generate_episode
from auditenv.generator.config import ConfigError, resolve_config
from auditenv.generator.validate import load_public, validate_episode
from auditenv.grader.client import GraderClient
from auditenv.grader.oracle import oracle_submission
from auditenv.policies import make_policy
from auditenv.transform import perturb_episode

TYPES = ("duplicate_invoice", "partial_payment", "fx_conversion", "period_cutoff")
BASES = {"KZT": ["USD", "EUR", "JPY"], "USD": ["KZT", "EUR", "JPY"], "EUR": ["USD", "KZT", "JPY"],
         "JPY": ["USD", "EUR", "KZT"]}
TIMEZONES = [["+14:00"], ["-12:00"], ["+05:45"], ["-03:30"], ["+00:00"], ["+05:00", "+06:00", "+03:00", "+00:00"]]
SIZES = {
    "min": {"n_customers": [2, 3], "n_vendors": [1, 2], "n_plain_invoices": [12, 14], "ledger_rows": [200, 220]},
    "default": {},
    "max": {"n_customers": [60, 60], "n_vendors": [30, 30], "n_plain_invoices": [110, 110], "ledger_rows": [1900, 1900]},
}
DISTRACTORS_MAX = {k: [5, 5] for k in ("n_boundary_invoices", "n_recurring_customers", "n_unidentified_receipts",
                                       "n_customer_advances", "n_last_day_invoices", "n_twin_orders", "n_refunds",
                                       "n_last_day_night_payments")}
ERROR_PROFILES: dict[str, dict[str, Any] | None] = {
    "twelve": {"clean_probability": 0.0, "max_errors": 12,
               "errors": {t: {"enabled": True, "min": 2, "max": 2} for t in TYPES},
               "interactions": {"fx_partial_boundary": {"enabled": True, "min": 1, "max": 1},
                                "cutoff_fx": {"enabled": True, "min": 1, "max": 1}}},
    "interactions_only": {"clean_probability": 0.0,
                          "errors": {t: {"enabled": True, "min": 0, "max": 0} for t in TYPES},
                          "interactions": {"fx_partial_boundary": {"enabled": True, "min": 1, "max": 1},
                                           "cutoff_fx": {"enabled": True, "min": 1, "max": 1}}},
    "one_type_heavy": None,  # filled per episode: one error type at min 3
    "clean": {"force_clean": True},
    "default": {"clean_probability": 0.0},
}
FOREIGN = {"none": 0.0, "all": 1.0, "default": None}
LEAK_KEYS = ("issue_id", "expected_amount_minor", "evidence_sets", "provenance", "ledger_removed", "ledger_added",
             "truth.json", "/private")


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = json.loads(json.dumps(v))
    return dst


def config_for(i: int) -> tuple[dict, dict]:
    """Deterministic extreme configuration number i, with a label describing it."""
    rng = random.Random(f"stress|{i}")
    base = list(BASES)[i % len(BASES)]
    size = rng.choice(list(SIZES))
    errors = rng.choice(list(ERROR_PROFILES))
    foreign = rng.choice(list(FOREIGN))
    tz = rng.choice(TIMEZONES)
    distract = rng.random() < 0.35
    cfg: dict = {"base_currency": base, "foreign_currencies": BASES[base], "timezones": tz}
    _merge(cfg, SIZES[size])
    if distract:
        _merge(cfg, DISTRACTORS_MAX)
    if errors == "one_type_heavy":
        heavy = rng.choice(TYPES)
        _merge(cfg, {"clean_probability": 0.0, "errors": {t: {"enabled": True, "min": 3 if t == heavy else 0,
                                                              "max": 3 if t == heavy else 1} for t in TYPES}})
    else:
        _merge(cfg, cast(dict, ERROR_PROFILES[errors]))  # None only for one_type_heavy, handled above
    fx_needed = errors in ("twelve", "interactions_only") or (errors == "one_type_heavy" and heavy == "fx_conversion")
    if FOREIGN[foreign] is not None and not (foreign == "none" and fx_needed):
        cfg["foreign_invoice_share"] = FOREIGN[foreign]
        if foreign == "none":  # no foreign invoices: FX errors cannot have minimums
            for t in ("fx_conversion",):
                cfg.setdefault("errors", {}).setdefault(t, {})["min"] = 0
    label = {"base": base, "size": size, "errors": errors if errors != "one_type_heavy" else f"heavy:{heavy}",
             "foreign": foreign if "foreign_invoice_share" in cfg else "default", "tz": ",".join(tz),
             "distractors_max": distract}
    return cfg, label


def _balanced(data: dict) -> bool:
    totals: dict[str, list[int]] = {}
    for r in data["ledger"]:
        t = totals.setdefault(r["currency"], [0, 0])
        t[0] += r["debit_minor"]
        t[1] += r["credit_minor"]
    return all(d == c for d, c in totals.values())


def _fetchable(rec: Records, eid: str) -> bool:
    """Can the agent fetch this evidence id with one of its own read tools?"""
    attempts = [("get_document", {"doc_id": eid}), ("get_record", {"table": "ledger", "id": eid}),
                ("get_record", {"table": "fx", "id": eid})]
    for tool, args in attempts:
        try:
            READ_TOOLS[tool](rec, args)
            return True
        except ToolError:
            continue
    return False


def _affected_fetchable(rec: Records, aid: str) -> bool:
    for table in ("invoices", "payments"):
        try:
            READ_TOOLS["get_record"](rec, {"table": table, "id": aid})
            return True
        except ToolError:
            continue
    return False


def _run(env: AuditEnv, ep: Path, policy: str, log: Path) -> tuple[list, list, float, dict]:
    observations, infos = [], []
    t = time.perf_counter()
    obs, info = env.reset(episode_dir=ep, grader=GraderClient.for_episode(ep, log_path=log, policy_name=policy))
    observations.append(obs)
    infos.append(info)
    if policy == "oracle":
        out = env.step({"tool": "submit", "args": {"findings": oracle_submission(ep)}})
        observations.append(out[0])
        infos.append(out[4])
    else:
        p = make_policy(policy, ep)
        p.reset(obs, info)
        while True:
            obs, reward, terminated, truncated, info = env.step(p.act(obs))
            observations.append(obs)
            infos.append(info)
            p.observe(obs)
            if terminated or truncated:
                break
    elapsed = time.perf_counter() - t
    row = json.loads(log.read_text().splitlines()[-1])
    return observations, infos, elapsed, row


def check_episode(i: int, root: Path) -> dict:
    cfg, label = config_for(i)
    seed = 700_000 + i
    problems: list[str] = []
    res: dict[str, Any] = {"i": i, "seed": seed, "label": label, "problems": problems}
    try:
        resolve_config(cfg)
    except ConfigError as e:
        res["rejected"] = str(e)
        return res
    t = time.perf_counter()
    try:
        ep = generate_episode(seed, cfg, root / "eps")
    except Exception as e:  # noqa: BLE001 - every failure is a finding
        problems.append(f"generation failed: {type(e).__name__}: {e}")
        res["config"] = cfg
        return res
    res["t_generate"] = time.perf_counter() - t
    m = read_json(ep / "public" / "manifest.json")
    truth = read_json(ep / "private" / "truth.json")
    data = load_public(ep)
    rows, docs, n_issues = m["record_counts"]["ledger"], m["document_count"], len(truth["issues"])
    res.update({"ledger_rows": rows, "documents": docs, "issues": n_issues})
    if not (200 <= rows <= 2000 and 20 <= docs <= 200 and 0 <= n_issues <= 12):
        problems.append(f"size contract: rows {rows}, documents {docs}, issues {n_issues}")
    t = time.perf_counter()
    problems += [f"validator: {p}" for p in validate_episode(ep)]
    res["t_validate"] = time.perf_counter() - t
    if not _balanced(data):
        problems.append("ledger does not balance per currency")

    rec = Records(ep)
    for issue in truth["issues"]:
        full = [es for es in issue["evidence_sets"] if all(_fetchable(rec, e) for e in es)]
        if not full:
            problems.append(f"{issue['type']}: no evidence set is fully fetchable with the agent's tools")
        for aid in issue["affected_ids"]:
            if not _affected_fetchable(rec, aid):
                problems.append(f"{issue['type']}: affected id {aid} is not a fetchable record")

    log = root / "grader_log.jsonl"
    env = AuditEnv(mode="eval", workdir=root / "cache", grader_log=log)
    _, _, res["t_oracle"], row = _run(env, ep, "oracle", log)
    if not (row["f1"] == 1.0 and row["reward"] == 1.0 and row["valid"]):
        problems.append(f"oracle: valid {row['valid']}, F1 {row['f1']}, reward {row['reward']}")

    observations, infos, res["t_reconcile"], row = _run(env, ep, "reconcile", log)
    res.update({"reconcile_steps": row["inspection_steps"], "reconcile_f1": row["f1"]})
    blob = json.dumps(observations) + json.dumps(infos)
    leaked = [k for k in LEAK_KEYS if k in blob] + [i["issue_id"] for i in truth["issues"] if i["issue_id"] in blob]
    if leaked:
        problems.append(f"eval leak: {leaked[:5]}")
    if any(k in inf for inf in infos for k in ("metrics", "feedback")):
        problems.append("eval leak: metrics or feedback in info")
    res["max_obs_bytes"] = max(inf.get("obs_bytes", 0) for inf in infos)

    if i % 10 == 0:
        twin = perturb_episode(ep, root / "twins", seed=i)
        twin_problems = validate_episode(twin)
        _, _, _, trow = _run(env, twin, "oracle", log)
        res["twin_checked"] = True
        if twin_problems or trow["reward"] != 1.0:
            problems.append(f"twin: validator {twin_problems[:2]}, oracle reward {trow['reward']}")
    shutil.rmtree(ep, ignore_errors=True)
    if problems:
        res["config"] = cfg
    return res


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 4)


def run_stress(n: int, root: str | Path, start: int = 0) -> dict:
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    t0 = time.perf_counter()
    results = [check_episode(i, root) for i in range(start, start + n)]
    rejected = [r for r in results if r.get("rejected")]
    ok = [r for r in results if not r["problems"] and not r.get("rejected")]
    failures = [r for r in results if r["problems"]]
    reasons: dict[str, int] = {}
    for r in rejected:
        text = r["rejected"]
        found = re.search(r"need \d+ (.+?)(?: \(|, but)", text)
        key = "size contract (documents)" if text.startswith("size contract") else \
            (f"too few {found.group(1)}" if found else text[:60])
        reasons[key] = reasons.get(key, 0) + 1

    def timing(key):
        vals = [r[key] for r in ok if key in r]
        return {"p50": _pct(vals, 0.5), "p95": _pct(vals, 0.95), "max": round(max(vals), 4)} if vals else None

    def count(field):
        out: dict[str, int] = {}
        for r in ok:
            k = str(r["label"][field])
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    max_size = [r for r in ok if r["label"]["size"] == "max"]
    summary = {
        "configs": n, "rejected_by_config_check": len(rejected), "rejection_reasons": reasons,
        "episodes_generated_and_checked": len(ok) + len(failures), "passed": len(ok), "failed": len(failures),
        "coverage": {f: count(f) for f in ("base", "size", "errors", "foreign", "tz", "distractors_max")},
        "issues_total": sum(r.get("issues", 0) for r in results),
        "issues_per_episode_max": max((r.get("issues", 0) for r in results), default=0),
        "ledger_rows_range": [min(r["ledger_rows"] for r in ok), max(r["ledger_rows"] for r in ok)] if ok else None,
        "documents_range": [min(r["documents"] for r in ok), max(r["documents"] for r in ok)] if ok else None,
        "twins_checked": sum(1 for r in results if r.get("twin_checked")),
        "seconds": {k: timing(k) for k in ("t_generate", "t_validate", "t_oracle", "t_reconcile")},
        "max_size_episode_seconds": {
            "generate_max": round(max((r["t_generate"] for r in max_size), default=0), 3),
            "reconcile_run_max": round(max((r["t_reconcile"] for r in max_size), default=0), 3),
            "reconcile_steps_max": max((r["reconcile_steps"] for r in max_size), default=0)},
        "max_obs_bytes": max((r.get("max_obs_bytes", 0) for r in ok), default=0),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "reconcile_f1_mean": round(statistics.fmean(r["reconcile_f1"] for r in ok), 4) if ok else None,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "failures": failures,
    }
    return summary

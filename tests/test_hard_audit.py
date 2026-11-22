"""Regression tests for the "hard jury" audit (audit_hard.txt): every hole a jury member who trusts
neither our tests nor our validator said they would probe. Item numbers follow the audit.

The full-size runs (100 seeds per D scenario, 20 seeds per edge config, 3 hard pairs per source,
real 60 s) are `make hard-audit` -> artifacts/hard_audit.json; these tests are the fast versions.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from auditenv import hard_audit as ha
from auditenv import jury_checks as jc
from auditenv.common import read_json, write_json
from auditenv.env.calculator import CalcError, evaluate
from auditenv.generator.validate import load_public, validate_episode
from auditenv.independent import reconcile
from auditenv.judge import generate_episode, make_env, score_episode

ROOT = Path(__file__).resolve().parents[1]
MATRIX = read_json(ROOT / "configs" / "jury_matrix.json")["scenarios"]


@pytest.fixture(scope="module")
def eps(tmp_path_factory):
    root = tmp_path_factory.mktemp("hard")
    out = {}
    for did in ("D01", "D05", "D06", "D09", "D10", "D12", "D14", "D16", "D19"):
        out[did] = root / did
        generate_episode(31_337 + int(did[1:]), MATRIX[did]["config"], str(out[did]))
    return out


# ------------------------------------------------------------------ item 2: independent reconciler
def test_independent_reconciler_imports_nothing_from_auditenv():
    tree = ast.parse((ROOT / "auditenv" / "independent.py").read_text(encoding="utf-8"))
    modules = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    modules |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m == "auditenv" or m.startswith("auditenv.") or m.startswith(".") for m in modules), modules
    assert modules <= {"__future__", "csv", "json", "re", "sys", "collections", "datetime", "decimal", "pathlib",
                       "typing"}  # standard library only; typing for annotations (1.8.6)


def test_independent_reconciler_accepts_every_jury_scenario(eps):
    for did, ep in eps.items():
        report = reconcile(ep)
        assert report["problems"] == [], (did, report["problems"][:3])


@pytest.mark.parametrize("name", ["jpy_base_interacting", "tz_+05:45", "tz_-03:30", "period_boundary",
                                  "partial_fx_one_document_+05:45_jpy", "usd_base_jpy_foreign"])
def test_edge_configs_pass_both_reconcilers(tmp_path, name):
    for seed in ha.jury_seeds(name, 3):
        ep = tmp_path / str(seed)
        generate_episode(seed, ha.EDGE_CONFIGS[name], str(ep))
        assert ha.check_generated(ep, ha.EDGE_CONFIGS[name]) == [], (name, seed)


def _tamper(ep: Path, dst: Path, fn) -> Path:
    shutil.copytree(ep, dst)
    fn(dst)
    return dst


def test_independent_reconciler_catches_tampering(eps, tmp_path):
    def amount_plus_one(d):
        t = read_json(d / "private" / "truth.json")
        t["issues"][0]["expected_amount_minor"] += 1
        write_json(d / "private" / "truth.json", t)

    def drop_change(d):
        p = read_json(d / "private" / "provenance.json")
        p["changes"] = [c for c in p["changes"] if not (c.get("issue_id") or c.get("issue_ids"))]
        write_json(d / "private" / "provenance.json", p)

    def hidden_fx(d):  # an undeclared wrong rate that still balances
        rows = list(csv.DictReader(open(d / "public" / "ledger.csv", encoding="utf-8")))
        invs = [json.loads(x) for x in open(d / "public" / "invoices.jsonl", encoding="utf-8")]
        base = read_json(d / "public" / "manifest.json")["base_currency"]
        inv = next(i for i in invs if i["currency"] != base)
        ar = next(r for r in rows if r["account"] == "1100" and r["invoice_id"] == inv["invoice_id"] and not r["payment_id"])
        rev = next(r for r in rows if r["account"] == "4000" and r["invoice_id"] == inv["invoice_id"] and not r["payment_id"])
        ar["debit_minor"] = str(int(ar["debit_minor"]) + 500)
        rev["credit_minor"] = str(int(rev["credit_minor"]) + 500)
        with open(d / "public" / "ledger.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    cases = [("D12", amount_plus_one), ("D16", amount_plus_one), ("D09", drop_change), ("D01", hidden_fx)]
    for n, (did, fn) in enumerate(cases):
        bad = _tamper(eps[did], tmp_path / f"t{n}", fn)
        assert reconcile(bad)["problems"], (did, fn.__name__)


# ------------------------------------------------------------------ item 4: generator on jury seeds
def test_generator_places_the_minimum_on_jury_seeds(tmp_path):
    out = ha.seed_sweep(tmp_path, 4)
    assert all(row["failures"] == 0 for row in out.values()), {k: v["examples"] for k, v in out.items() if v["failures"]}
    assert all(row["issues"] >= 4 for did, row in out.items() if int(did[1:]) >= 6)


def test_generator_post_condition_never_publishes_fewer_errors(monkeypatch, tmp_path):
    """If the injector's bookkeeping ever under-delivered, the last line refuses to publish."""
    import auditenv.generator as gen
    import auditenv.generator.errors as errors

    real = errors.inject_errors

    def lossy(world, rng, ids, cfg):
        real(world, rng, ids, cfg)
        world.provenance[:] = [c for c in world.provenance if c.get("op") != "fx_conversion"]

    monkeypatch.setattr(gen, "inject_errors", lossy)
    with pytest.raises(gen.GenerationError):
        gen.generate_episode(1, MATRIX["D12"]["config"], tmp_path)


# ------------------------------------------------------------------ item 3: hard paired copies
@pytest.mark.parametrize("did", ["D06", "D11", "D16", "D19"])
def test_hard_pair_unicode_ids_and_shuffles_keep_every_score(tmp_path, did):
    src, dst = tmp_path / "src", tmp_path / "copy"
    generate_episode(4242, MATRIX[did]["config"], str(src))
    result = ha.check_hard_pair(src, dst, seed=4242)
    assert result["problems"] == [], result["problems"][:3]
    lo, hi = result["id_code_points"]
    assert 80 <= lo <= hi <= 100
    steps = len(ha.SCRIPT)
    assert result["oracle"]["f1"] == 1.0 and abs(result["oracle"]["return"] - (1 - 0.002 * steps)) < 1e-12
    assert result["perturbed"]["tp"] == result["oracle"]["tp"] and result["perturbed"]["fp"] > 0


def test_reordered_csv_columns_load_through_the_adapter(eps, tmp_path):
    """A jury copy rewritten with another column order must not crash reset (1.8.0 asserted the order)."""
    dst = tmp_path / "cols"
    shutil.copytree(eps["D06"], dst)
    for name in ("ledger.csv", "fx.csv"):
        rows = list(csv.DictReader(open(dst / "public" / name, encoding="utf-8")))
        cols = list(reversed(list(rows[0])))
        with open(dst / "public" / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\r\n")
            w.writeheader()
            w.writerows(rows)
    assert load_public(dst)["ledger"] == load_public(eps["D06"])["ledger"]
    assert validate_episode(dst) == [] and reconcile(dst)["problems"] == []
    env = make_env()
    jc.closed(env, dst)
    row = load_public(dst)["ledger"][0]
    assert jc.act(env, "get_record", table="ledger", id=row["row_id"])[0]["result"] == row
    assert jc.act(env, "submit", findings=jc.oracle_findings(dst))[0]["result"] == {"accepted": True}


def test_validator_accepts_unsorted_truth_lists(eps, tmp_path):
    dst = tmp_path / "unsorted"
    shutil.copytree(eps["D06"], dst)
    truth = read_json(dst / "private" / "truth.json")
    for t in truth["issues"]:
        t["affected_ids"] = list(reversed(t["affected_ids"]))
        t["evidence_sets"] = [list(reversed(es)) for es in t["evidence_sets"]]
    write_json(dst / "private" / "truth.json", truth)
    assert validate_episode(dst) == []


# ------------------------------------------------------------------ item 5: real 60 seconds
@pytest.mark.skipif(os.environ.get("AUDITENV_REALTIME") != "1",
                    reason="real-clock 60 s run (~62 s): AUDITENV_REALTIME=1 or make hard-audit")
def test_time_limit_on_the_real_clock(eps):
    report = ha.realtime_limit(eps["D16"])
    assert report["passed"], report["scenarios"]


def test_time_limit_counts_policy_time_between_steps_on_the_real_clock(eps):
    """Short real-clock version (limit 1 s): an in-time step, then the policy idles past the limit."""
    import time
    from auditenv.judge import JudgeEnv
    env = JudgeEnv(time_limit_s=1.0)
    jc.closed(env, eps["D06"])
    assert jc.act(env, "calculator", expression="1")[0]["error"] is None
    time.sleep(1.1)
    obs, _, term, trunc, _ = jc.act(env, "submit", findings=jc.oracle_findings(eps["D06"]))
    assert trunc and not term and obs["error"]["code"] == "time_limit" and env.final_submission is None


# ------------------------------------------------------------------ item 6: isolation in closed mode
TRACE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
episode = sys.argv[2]
from judge_adapter import make_env
from auditenv import jury_checks as jc
answer = jc.oracle_findings(jc.Path(episode))          # jury side: read BEFORE the trace starts
events, active = [], [False]
def hook(event, args):
    if active[0] and event in ("open", "os.listdir", "os.scandir", "os.walk", "glob.glob", "shutil.copytree"):
        events.append([event, str(args[0]) if args else ""])
sys.addaudithook(hook)
active[0] = True
env = make_env()
seen = [env.reset(0, {"mode": "closed", "episode_dir": episode, "team_config": {}})]
for table in ("ledger", "invoices", "payments", "fx"):
    seen.append(env.step({"name": "list_records", "args": {"table": table, "cursor": None, "limit": 50}}))
for name, args in (("get_record", {"table": "ledger", "id": "../private/truth.json"}),
                   ("get_document", {"doc_id": "../private/truth.json"}),
                   ("get_document", {"doc_id": episode + "/private/truth.json"}),
                   ("list_records", {"table": "../private"}), ("calculator", {"expression": "open('x')"}),
                   ("read_file", {"path": "private/truth.json"})):
    seen.append(env.step({"name": name, "args": args}))
seen.append(env.step({"name": "submit", "args": {"findings": answer}}))
active[0] = False
holders = []
def scan(obj, path, depth, visited):
    if depth > 6 or id(obj) in visited:
        return
    visited.add(id(obj))
    if isinstance(obj, str):
        if episode in obj:
            holders.append(path)
        return
    if isinstance(obj, (jc.Path,)):
        if episode in str(obj):
            holders.append(path)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            scan(v, f"{path}[{k!r}]", depth + 1, visited)
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(list(obj)[:5000]):
            scan(v, f"{path}[{i}]", depth + 1, visited)
    elif hasattr(obj, "__dict__"):
        scan(vars(obj), path + ".__dict__", depth + 1, visited)
scan(env, "env", 0, set())
print(json.dumps({"events": events, "seen": json.dumps(seen, ensure_ascii=False), "holders": holders,
                  "accepted": seen[-1][0]["result"]}))
"""


def test_closed_mode_never_opens_private_and_holds_no_path_to_it(eps, tmp_path):
    """Item 6: OS-level trace (sys.addaudithook) of every file the env opens or lists during a whole
    closed episode, hostile probes included; the env object holds no reference to the episode path;
    nothing secret reaches the policy; relative private/ paths do not exist in the working dir."""
    seed = 917_263_548_316                      # unusual digits: a substring match means a real leak
    ep = tmp_path / "jury" / "D16"
    generate_episode(seed, MATRIX["D16"]["config"], str(ep))
    script = tmp_path / "trace.py"
    script.write_text(TRACE, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), str(ROOT), str(ep)], capture_output=True, text=True,
                          cwd=ROOT, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout)
    touched = [p for _, p in out["events"] if "private" in p or "truth" in p or "provenance" in p]
    assert touched == [], touched
    assert any(p.endswith("ledger.csv") for _, p in out["events"]), "trace did not record the public reads"
    assert out["holders"] == [], out["holders"]
    assert out["accepted"] == {"accepted": True}
    truth = read_json(ep / "private" / "truth.json")
    secrets = {t["issue_id"] for t in truth["issues"]} | {str(seed), str(ep), "provenance", "truth.json"}
    assert read_json(ep / "private" / "provenance.json").get("seed") == seed
    assert not [s for s in secrets if s in out["seen"]]
    assert not (ROOT / "private").exists() and not (Path.cwd() / "private").exists()


# ------------------------------------------------------------------ item 7: calculator
@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "1" * 10_001, "1" * 200_000, "1/0", "0/0",
                                  "9" * 31, "*".join(["9" * 30] * 6), "1+1\n", "\n1", "1e309", "１+１"])
def test_calculator_hostile_inputs_are_bounded_errors(eps, expr):
    with pytest.raises(CalcError):
        evaluate(expr)
    env = make_env()
    obs0, _ = jc.closed(env, eps["D01"])
    obs, _, term, trunc, _ = jc.act(env, "calculator", expression=expr)
    assert obs["result"] is None and obs["error"]["code"] in ("calc_error", "bad_action")
    assert len(obs["error"]["message"]) < 300 and obs["remaining_steps"] == 99 and not term and not trunc


def test_calculator_exact_and_big_results():
    assert evaluate("0.1 + 0.2") == "0.3"
    assert evaluate("9" * 30 + "*" + "9" * 30) == str(int("9" * 30) ** 2)   # 60 digits: still exact
    assert evaluate("1/3").startswith("0.3333")


# ------------------------------------------------------------------ item 8: pages on 1500+ rows
def test_pages_and_cursors_on_a_large_ledger(tmp_path):
    ep = tmp_path / "big"
    generate_episode(3, {"ledger_rows": [1600, 1900]}, str(ep))
    want = [r["row_id"] for r in load_public(ep)["ledger"]]
    assert len(want) >= 1500
    env = make_env()
    jc.closed(env, ep)

    def page(cursor):
        obs = jc.act(env, "list_records", table="ledger", cursor=cursor, limit=50)[0]
        assert obs["error"] is None, obs["error"]
        return [r["row_id"] for r in obs["result"]["records"]], obs["result"]["next_cursor"]

    cursor, seen, pages, n = None, [], {}, 0
    while True:
        rows, nxt = page(cursor)
        pages[cursor] = rows
        seen += rows
        n += 1
        if n % 5 == 0:                                                 # same cursor twice, other actions between
            assert page(cursor)[0] == rows
            jc.act(env, "get_record", table="ledger", id="nope")
            jc.act(env, "calculator", expression="2*2")
        if nxt is None:
            break
        cursor = nxt
    assert len(pages) >= 30 and len(seen) == len(set(seen)) and sorted(seen) == sorted(want)
    for cursor in list(pages)[1::9]:                                   # earlier cursors stay valid and stable
        assert page(cursor)[0] == pages[cursor]


# ------------------------------------------------------------------ item 10: the small things
def test_small_contract_points(eps):
    env = make_env()
    obs, info = jc.closed(env, eps["D06"])
    assert obs["remaining_steps"] == 100
    obs = jc.act(env, "explode", x=1)[0]
    assert obs["error"]["code"] == "unknown_action" and obs["remaining_steps"] == 99
    obs = jc.act(env, "get_record", table="ledger", id="NOPE")[0]
    assert obs["error"]["code"] == "not_found" and obs["remaining_steps"] == 98 and "debit" not in json.dumps(obs)
    for step in jc.act(env, "calculator", expression="1"), jc.act(env, "submit", findings=jc.oracle_findings(eps["D06"])):
        assert step[1] == 0.0 and set(step[4]) == jc.INFO_KEYS


# ------------------------------------------------------------------ item 9: honest limitations are checked
def test_unmodeled_domains_are_absent_from_the_data(eps):
    """README/DESIGN 'what the environment does not model' are statements about the data; each is
    checked here on the jury scenarios (one test per bullet of the limitations table)."""
    from auditenv.common import ERROR_TYPES
    accounts = {"1000", "1100", "2900", "4000", "6100", "6200", "6300", "6400", "7900"}
    for did, ep in eps.items():
        data = load_public(ep)
        policy, manifest = data["policy"], data["manifest"]
        assert set(policy["chart_of_accounts"]) == accounts                     # no tax, inventory, fixed assets,
        assert {r["account"] for r in data["ledger"]} <= accounts               # accruals, provisions, payroll liabilities
        for inv in data["invoices"]:
            assert sum(li["amount_minor"] for li in inv["line_items"]) == inv["total_minor"]   # no VAT on top
            assert all(set(li) == {"line_no", "description", "quantity", "unit_price_minor", "amount_minor"}
                       for li in inv["line_items"])
        assert {e["kind"] for e in manifest["entities"]} <= {"customer", "vendor"}   # one company, no group
        assert sorted(p.name for p in (ep / "public").iterdir()) == sorted(
            ["fx.csv", "invoices.jsonl", "ledger.csv", "manifest.json", "payments.jsonl", "policy.json"])  # no bank statements
        assert all(r["payment_id"] for r in data["ledger"] if r["account"] == "7900")   # no period-end revaluation
        assert all(r["invoice_id"] or r["payment_id"] or r["account"] in ("6100", "6200", "6300", "6400", "1000")
                   for r in data["ledger"])                                     # no opening balances / other periods
    assert tuple(ERROR_TYPES) == ("duplicate_invoice", "partial_payment", "fx_conversion", "period_cutoff")  # no fraud

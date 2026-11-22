"""Training-readiness: the fixes from the LLM-training review (docs/LLM_TRAINING.md).

Each test names the risk it covers. None of these features may change the official contract:
the shared reward, the -1 for invalid structure, the empty answer on truncation and the eval-mode
silence are asserted alongside.
"""

import copy
import decimal
import json
import multiprocessing
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from auditenv import API_VERSION
from auditenv.common import append_line, read_json
from auditenv.env.core import OBS_BYTE_BUDGET, AuditEnv, Recorder
from auditenv.env.tools import TOOL_SPECS
from auditenv.generator import _fingerprint, generate_episode
from auditenv.generator.validate import validate_episode
from auditenv.grader.oracle import oracle_submission
from auditenv.grader.reward import EXPERIMENTAL_FORMAT_PENALTY, FEEDBACK_CODES, score


def truth_of(episode_dir):
    return read_json(Path(episode_dir) / "private" / "truth.json")

ROOT = Path(__file__).resolve().parents[1]
MAX_CFG = {"ledger_rows": [1900, 1900], "n_plain_invoices": [110, 110], "n_customers": [60, 60],
           "n_vendors": [30, 30], "clean_probability": 0.0}
TYPES = ("duplicate_invoice", "partial_payment", "fx_conversion", "period_cutoff")
TWELVE_CFG = {"clean_probability": 0.0, "max_errors": 12,
              "errors": {t: {"enabled": True, "min": 2, "max": 2} for t in TYPES},
              "interactions": {"fx_partial_boundary": {"enabled": True, "min": 1, "max": 1},
                               "cutoff_fx": {"enabled": True, "min": 1, "max": 1}}}


def _rich_episode(episodes):
    """A dirty episode with at least two issues of different kinds."""
    return max(episodes["dirty"], key=lambda e: len({i["type"] for i in truth_of(e)["issues"]}))


# --------------------------------------------------------------- infrastructure
def _writer(path, n, tag):
    for i in range(n):
        append_line(path, json.dumps({"w": tag, "i": i, "pad": "x" * 3000}))


def test_shared_log_lines_never_interleave(tmp_path):
    """Several worker processes appending to one grader log / trajectory file."""
    path = tmp_path / "log.jsonl"
    procs = [multiprocessing.Process(target=_writer, args=(path, 200, w)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 800 and {(r["w"], r["i"]) for r in rows} == {(w, i) for w in range(4) for i in range(200)}


def test_recorder_needs_no_close(tmp_path, episodes):
    path = tmp_path / "t.jsonl"
    env = AuditEnv(mode="train", recorder=Recorder(path), workdir=tmp_path / "c")
    env.reset(episode_dir=episodes["clean"][0])
    env.step({"tool": "list_records", "args": {"table": "fx"}})
    del env  # never closed
    assert len(path.read_text().splitlines()) == 2
    with Recorder(path) as rec:
        rec.write({"x": 1})
    assert len(path.read_text().splitlines()) == 3


def test_generation_is_immune_to_the_callers_decimal_context(tmp_path):
    reference = [_fingerprint(generate_episode(s, c, tmp_path / "a")) for s, c in ((5, "cutoff_fx"), (6, None))]
    ctx = decimal.getcontext()
    saved = ctx.copy()
    try:
        ctx.prec, ctx.rounding = 4, decimal.ROUND_DOWN
        hostile = [generate_episode(s, c, tmp_path / "b") for s, c in ((5, "cutoff_fx"), (6, None))]
        assert [_fingerprint(e) for e in hostile] == reference
        assert all(validate_episode(e) == [] for e in hostile)
    finally:
        decimal.setcontext(saved)


# ------------------------------------------- risk 2: the format -1 must not hide the audit
def test_experimental_reward_equals_official_on_valid_submissions(episodes):
    for ep in episodes["dirty"] + episodes["clean"]:
        truth = truth_of(ep)
        for sub in (oracle_submission(ep), {"episode_id": truth["episode_id"], "findings": []}):
            g = score(sub, truth, 17)
            assert g["valid"] and g["experimental_reward"] == g["reward"]


def test_one_malformed_finding_burns_official_but_not_experimental(episodes):
    ep = _rich_episode(episodes)
    truth = truth_of(ep)
    sub = oracle_submission(ep)
    bad = copy.deepcopy(sub)
    bad["findings"][0]["claimed_amount_minor"] = float(bad["findings"][0]["claimed_amount_minor"])
    g = score(bad, truth, 10)
    assert g["valid"] is False and g["reward"] == -1.0  # official rule unchanged
    n = len(truth["issues"])
    f1 = 2 * (n - 1) / (2 * (n - 1) + 1 + 1)  # one TP lost, the malformed finding counts as FP
    assert g["experimental_reward"] == pytest.approx(max(-1, 2 * f1 - 1 - 0.002 * 10))
    assert g["feedback"]["findings"][0] == "malformed" and g["malformed_findings"] == 1


def test_extra_top_level_key_costs_a_penalty_in_the_experimental_channel(episodes):
    ep = episodes["dirty"][0]
    truth = truth_of(ep)
    sub = {**oracle_submission(ep), "notes": "extra"}
    g = score(sub, truth, 0)
    assert g["reward"] == -1.0 and g["experimental_reward"] == pytest.approx(1.0 - EXPERIMENTAL_FORMAT_PENALTY)
    # episode_id is informational (hard-jury audit): a foreign one is neither -1 nor penalised
    foreign = score({**oracle_submission(ep), "episode_id": "ep-000000000000"}, truth, 0)
    assert foreign["reward"] == 1.0 and foreign["experimental_reward"] == 1.0
    for junk in (None, [], "submit", {"findings": "x"}):
        assert score(junk, truth, 0)["experimental_reward"] == -1.0


# --------------------------------- risk 4: the policy must see WHY a finding missed
def test_feedback_codes_explain_misses_without_truth(episodes):
    ep = _rich_episode(episodes)
    truth = truth_of(ep)
    base = oracle_submission(ep)["findings"]
    f0 = base[0]
    variants = [
        dict(f0),                                                                    # matched
        {**f0, "claimed_amount_minor": f0["claimed_amount_minor"] + 10_000},          # wrong amount
        {**f0, "currency": "XAU"},                                                   # wrong currency
        {**f0, "evidence_ids": []},                                                  # insufficient evidence
        {**f0, "type": next(t for t in TYPES if t != f0["type"])},                    # wrong type
        {**f0, "affected_ids": ["NOPE-1"]},                                          # not an issue
        {**f0, "claimed_amount_minor": "12"},                                        # malformed
    ]
    first_open = [variants[0]] + [dict(f0)]                                          # duplicate of a matched one
    g = score({"episode_id": truth["episode_id"], "findings": variants}, truth, 0)
    codes = g["feedback"]["findings"]
    assert codes[0] == "matched" and codes[-1] == "malformed" and codes[-2] == "not_an_issue"
    assert codes[4] in ("wrong_type", "not_an_issue")
    alone = lambda f: score({"episode_id": truth["episode_id"], "findings": [f]}, truth, 0)["feedback"]["findings"][0]  # noqa: E731
    assert alone(variants[1]) == "wrong_amount"
    assert alone(variants[2]) == "wrong_currency"
    assert alone(variants[3]) == "insufficient_evidence"
    assert score({"episode_id": truth["episode_id"], "findings": first_open}, truth, 0)["feedback"]["findings"] == \
        ["matched", "duplicate"]
    assert set(codes) <= set(FEEDBACK_CODES)
    # Never truth content: no expected amounts, issue ids or evidence sets the agent did not send.
    blob = json.dumps(g["feedback"])
    for issue in truth["issues"]:
        assert issue["issue_id"] not in blob and str(issue["expected_amount_minor"]) not in blob
        for es in issue["evidence_sets"]:
            assert not any(e in blob for e in es)


def test_feedback_and_experimental_reward_reach_the_policy_only_in_train_mode(tmp_path, episodes):
    ep = _rich_episode(episodes)
    sub = oracle_submission(ep)
    sub["findings"][0]["claimed_amount_minor"] += 10_000
    out = {}
    for mode in ("train", "eval"):
        env = AuditEnv(mode=mode, workdir=tmp_path / mode, grader_log=tmp_path / f"{mode}.jsonl")
        obs, info = env.reset(episode_dir=ep)
        assert info["api_version"] == API_VERSION and "feedback" not in info
        _, reward, terminated, _, info = env.step({"tool": "submit", "args": {"findings": sub}})
        assert terminated
        out[mode] = (reward, info)
    reward, info = out["train"]
    graded = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[-1])
    assert reward == graded["reward"]  # the policy gets the OFFICIAL reward, not the training channel
    assert info["feedback"]["findings"][0] == "wrong_amount"
    assert info["metrics"]["experimental_reward"] == graded["experimental_reward"]
    reward, info = out["eval"]
    assert reward == 0.0 and "feedback" not in info and "metrics" not in info
    assert "feedback" in json.loads((tmp_path / "eval.jsonl").read_text().splitlines()[-1])  # grader side only


def test_truncation_stays_an_empty_answer_in_both_channels(tmp_path, episodes):
    """The task fixes truncation = empty final answer; no draft is substituted."""
    ep = episodes["dirty"][0]
    env = AuditEnv(mode="train", max_steps=3, workdir=tmp_path / "c", grader_log=tmp_path / "g.jsonl")
    env.reset(episode_dir=ep)
    for _ in range(3):
        _, reward, terminated, truncated, info = env.step({"tool": "list_records", "args": {"table": "fx"}})
    assert truncated and not terminated
    assert info["metrics"]["experimental_reward"] == reward == info["metrics"]["reward"]
    assert info["feedback"]["findings"] == []


# ------------------------------------------------- observation budget, filters
def test_every_observation_fits_the_byte_budget(tmp_path):
    ep = generate_episode(1, MAX_CFG, tmp_path / "eps")
    env = AuditEnv(mode="train", workdir=tmp_path / "c", grader_log=tmp_path / "g.jsonl")
    _, info = env.reset(episode_dir=ep)
    sizes = [info["obs_bytes"]]
    for table in ("ledger", "invoices", "payments", "fx"):
        cursor = None
        while True:
            obs, *_, info = env.step({"tool": "list_records", "args": {"table": table, "cursor": cursor}})
            sizes.append(info["obs_bytes"])
            assert info["obs_bytes"] == len(json.dumps(obs).encode())
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
        env.reset(episode_dir=ep)
    assert max(sizes) <= OBS_BYTE_BUDGET, max(sizes)


def test_where_filter_is_opt_in_and_pages_exactly(tmp_path, episodes):
    ep = episodes["dirty"][0]
    official = AuditEnv(mode="train", workdir=tmp_path / "a")
    obs, _ = official.reset(episode_dir=ep)
    assert obs["tools"] == json.loads(json.dumps(TOOL_SPECS))
    o, *_ = official.step({"tool": "list_records", "args": {"table": "ledger", "where": {"account": "2900"}}})
    assert o["error"]["code"] == "bad_args" and "opt-in" in o["error"]["message"]

    env = AuditEnv(mode="train", workdir=tmp_path / "b", tool_extensions=True)
    obs, _ = env.reset(episode_dir=ep)
    assert "where" in obs["tools"]["list_records"]["args"]
    ledger = [json.loads(json.dumps(r)) for r in __import__("auditenv.generator.validate", fromlist=["x"]).load_public(ep)["ledger"]]
    want = [r["row_id"] for r in ledger if r["account"] == "1100"]
    got, cursor = [], None
    while True:
        o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "where": {"account": "1100"},
                                                           "cursor": cursor, "limit": 7}})
        assert o["error"] is None and o["result"]["total"] == len(want)
        got += [r["row_id"] for r in o["result"]["rows"]]
        cursor = o["result"]["next_cursor"]
        if cursor is None:
            break
        replay, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "where": {"account": "1000"},
                                                                "cursor": cursor}})
        assert replay["error"]["code"] == "bad_args"  # a cursor is bound to its filter
    assert got == want
    for bad in ({"amount": "1"}, {}, {"account": 1100}, "account=1100"):
        o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "where": bad}})
        assert o["error"]["code"] == "bad_args"


# ----------------------------------------- risk 5: an agent must not find private/
def test_served_agent_starts_in_an_empty_dir_without_paths(tmp_path, episodes):
    from auditenv.serve import run_agent
    report = tmp_path / "agent_view.json"
    agent = tmp_path / "agent.py"
    agent.write_text(textwrap.dedent(f"""
        import json, os, sys
        json.dump({{"cwd": os.getcwd(), "listing": os.listdir("."), "env": dict(os.environ)}}, open({str(report)!r}, "w"))
        for line in sys.stdin:
            msg = json.loads(line)
            if msg["terminated"] or msg["truncated"]:
                break
            eid = msg["observation"].get("manifest", {{}}).get("episode_id")
            print(json.dumps({{"tool": "submit", "args": {{"findings": {{"episode_id": eid, "findings": []}}}}}}), flush=True)
    """))
    workdir = tmp_path / "secret_workdir"
    env = AuditEnv(mode="eval", workdir=workdir, grader_log=tmp_path / "g.jsonl")
    summary = run_agent([sys.executable, str(agent)], env, {"seed": 4242, "config": "interacting"},
                        agent_env={"MY_TOKEN": "abc"})
    assert summary["terminated"]
    view = json.loads(report.read_text())
    assert view["listing"] == [] and str(ROOT) not in view["cwd"] and str(workdir) not in view["cwd"]
    blob = json.dumps(view["env"])
    assert str(workdir) not in blob and "PYTHONPATH" not in view["env"] and view["env"]["MY_TOKEN"] == "abc"
    assert str(tmp_path) not in blob.replace(view["cwd"], "")


# ------------------------------------------------------ pools, workers, monitor
def test_pool_is_deterministic_mixed_and_refuses_heldout(tmp_path):
    from auditenv.training import EpisodePool, build_pool
    a, b = build_pool(tmp_path / "a", 8), build_pool(tmp_path / "b", 8)
    ia, ib = read_json(a / "index.json"), read_json(b / "index.json")
    assert [e["path"] for e in ia["episodes"]] == [e["path"] for e in ib["episodes"]]
    assert [e["class"] for e in ia["episodes"]] == ["clean", "interacting", "standard", "standard"] * 2
    pool = EpisodePool(a)
    assert pool.sample(random.Random(3)) == pool.sample(random.Random(3)) and len(pool) == 8
    with pytest.raises(ValueError):
        build_pool(tmp_path / "c", 2, kind="eval")
    easy = read_json(build_pool(tmp_path / "d", 2, preset="level_easy") / "index.json")
    assert {e["class"] for e in easy["episodes"]} == {"level_easy"}


def test_worker_envs_share_nothing(tmp_path):
    from auditenv.training import TRAIN_TIME_LIMIT_S, make_worker_env
    envs = [make_worker_env(tmp_path, w) for w in (0, 1)]
    for env in envs:
        env.reset(seed=11)
        env.step({"tool": "submit", "args": {"findings": {"episode_id": env.episode_id, "findings": []}}})
        assert env.mode == "train" and env.time_limit_s == TRAIN_TIME_LIMIT_S
    assert envs[0].grader_log != envs[1].grader_log and envs[0].workdir != envs[1].workdir
    for w in (0, 1):
        assert len((tmp_path / f"w{w}" / "grader_log.jsonl").read_text().splitlines()) == 1
    envs[0].reset(seed=12)  # keep_episodes=False: the previous generated episode is gone
    assert len(list((tmp_path / "w0" / "cache").glob("ep-*"))) == 1


def test_monitor_flags_collapse_format_and_truncation(tmp_path, episodes):
    from auditenv.training import alarms, monitor
    log = tmp_path / "g.jsonl"
    ep = episodes["dirty"][0]
    env = AuditEnv(mode="train", grader_log=log, workdir=tmp_path / "c", max_steps=2)
    eid = None
    for kind in ("silent", "silent", "broken", "truncated"):
        _, info = env.reset(episode_dir=ep)
        eid = info["episode_id"]
        if kind == "silent":
            env.step({"tool": "submit", "args": {"findings": {"episode_id": eid, "findings": []}}})
        elif kind == "broken":
            env.step({"tool": "submit", "args": {"findings": {"episode_id": eid, "findings": [{"type": 1}]}}})
        else:
            env.step({"tool": "list_records", "args": {"table": "fx"}})
            env.step({"tool": "list_records", "args": {"table": "fx"}})
    stats = monitor(log)
    assert stats["episodes"] == 4 and stats["share_empty_submission"] == 0.5
    assert stats["share_empty_on_dirty_books"] == 0.5 and stats["share_invalid"] == 0.25
    assert stats["share_truncated"] == 0.25 and all(v == 0.0 for v in stats["recall_by_type"].values() if v is not None)
    text = " ".join(alarms(stats))
    assert "share_truncated" in text and "share_invalid" in text and "recall[" in text


# ------------------------------------------------------------- LLM adapter
def test_tool_call_parser_and_decoding_schema(episodes):
    jsonschema = pytest.importorskip("jsonschema")
    from auditenv.llm import action_json_schema, parse_action
    assert parse_action('```json\n{"tool": "list_records", "args": {"table": "fx"}}\n```')[0]["tool"] == "list_records"
    assert parse_action('blah {"x": 1} {"tool": "calculator", "args": {"expression": "1+1"}}')[0]["tool"] == "calculator"
    for junk in ("", "no json", None, '{"tool": 5}', "{" * 5000):
        action, err = parse_action(junk)
        assert action["tool"] == "unparseable_output" and err
    ep = episodes["dirty"][0]
    sub = oracle_submission(ep)
    schema = action_json_schema(sub["episode_id"])
    jsonschema.validate({"tool": "submit", "args": {"findings": sub}}, schema)
    jsonschema.validate({"tool": "list_records", "args": {"table": "fx"}}, schema)
    bad_float = copy.deepcopy(sub)
    bad_float["findings"][0]["claimed_amount_minor"] = 1.5
    for bad in ({**sub, "episode_id": "ep-000000000000"}, bad_float, {**sub, "extra": 1}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"tool": "submit", "args": {"findings": bad}}, schema)


def test_text_rollouts_match_direct_runs_and_form_grpo_groups(tmp_path, episodes):
    from auditenv.llm import ScriptedTextPolicy, collect_group, run_episode
    from auditenv.policies import make_policy
    from auditenv.training import make_worker_env
    ep = _rich_episode(episodes)
    env = make_worker_env(tmp_path, 0)
    rollout = run_episode(ScriptedTextPolicy(lambda obs: make_policy("reconcile", ep)), env, {"episode_dir": ep})
    direct = AuditEnv(mode="train", workdir=tmp_path / "d", grader_log=tmp_path / "d.jsonl")
    obs, info = direct.reset(episode_dir=ep)
    p = make_policy("reconcile", ep)
    p.reset(obs, info)
    while True:
        obs, reward, terminated, truncated, _ = direct.step(p.act(obs))
        p.observe(obs)
        if terminated or truncated:
            break
    assert rollout["reward"] == reward and rollout["terminated"] and rollout["parse_errors"] == 0
    roles = [m["role"] for m in rollout["messages"]]
    assert roles[:2] == ["system", "user"] and roles[2::2] == ["assistant"] * len(roles[2::2])

    rng, state = random.Random(1), {}
    scripted = ScriptedTextPolicy(lambda obs: make_policy("reconcile", ep))

    def sampler(msgs):  # a stochastic "model": per episode, either the script or a confused reply
        if len(msgs) == 2:
            state["good"] = rng.random() < 0.5
        return scripted(msgs) if state["good"] else "I am not sure."

    group = collect_group(sampler, env, ep, k=6)
    rewards = {r["reward"] for r in group}
    assert len(group) == 6 and len(rewards) == 2 and abs(sum(r["advantage"] for r in group)) < 1e-6
    best = max(group, key=lambda r: r["reward"])
    assert best["advantage"] > 0 and min(group, key=lambda r: r["reward"])["advantage"] < 0
    garbage = run_episode(lambda msgs: "hmm", env, {"episode_dir": ep})
    assert garbage["truncated"] and garbage["parse_errors"] == 100 and garbage["reward"] == -1.0


def test_sft_export_is_truth_free_and_solves_the_open_episodes(tmp_path):
    out = tmp_path / "sft.jsonl"
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "export_sft.py"), "--episodes",
                           str(ROOT / "episodes" / "dev"), "--out", str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 10 and all(r["fp"] == 0 and r["fn"] == 0 and r["f1"] == 1.0 for r in rows)
    blob = out.read_text()
    for key in ("expected_amount_minor", "evidence_sets", "issue_id", "provenance", "truth.json"):
        assert key not in blob
    for ep in (ROOT / "episodes" / "dev").iterdir():
        for issue in truth_of(ep)["issues"]:
            assert issue["issue_id"] not in blob


def test_sft_routes_respect_the_reads_each_check_needs_and_differ():
    sys.path.insert(0, str(ROOT / "examples"))
    import export_sft
    from auditenv.wrappers import OPTIONS, REQUIRES, TABLE_OF
    routes = export_sft.routes("ep-x", 5)
    assert routes[0] == list(OPTIONS) and len({tuple(r) for r in routes}) == len(routes) == 5
    for r in routes:
        assert sorted(r) == sorted(OPTIONS) and r[-1] == "submit"
        for i, option in enumerate(r):
            read = {TABLE_OF[o] for o in r[:i] if o in TABLE_OF}
            assert REQUIRES.get(option, set()) <= read, (r, option)
    assert export_sft.routes("ep-x", 5) == routes  # fixed seed


def test_sft_keeps_a_long_correct_transcript_and_drops_a_short_wrong_one():
    """Selection by correctness: 0.002 per step would push a flawless 51-step run to 0.898."""
    sys.path.insert(0, str(ROOT / "examples"))
    import export_sft
    assert export_sft.correct({"fp": 0, "fn": 0, "reward": 0.898, "steps": 51})
    assert not export_sft.correct({"fp": 1, "fn": 0, "reward": 0.94, "steps": 5})


# ------------------------------------------------------------- size limits
def test_maximum_size_episodes_are_valid_and_auditable(tmp_path):
    from auditenv.policies import make_policy
    for seed in (1, 2):
        ep = generate_episode(seed, MAX_CFG, tmp_path / "eps")
        m = read_json(ep / "public" / "manifest.json")
        assert 1800 <= m["record_counts"]["ledger"] <= 2000 and m["document_count"] <= 200
        assert validate_episode(ep) == []
        env = AuditEnv(mode="eval", workdir=tmp_path / "c", grader_log=tmp_path / "g.jsonl")
        obs, info = env.reset(episode_dir=ep)
        p = make_policy("reconcile", ep)
        p.reset(obs, info)
        while True:
            obs, _, terminated, truncated, info = env.step(p.act(obs))
            p.observe(obs)
            if terminated or truncated:
                break
        assert terminated and info["steps_used"] < 60


def test_twelve_errors_is_reachable_and_valid(tmp_path):
    for seed in (1, 2, 3):
        ep = generate_episode(seed, TWELVE_CFG, tmp_path / "eps")
        assert len(truth_of(ep)["issues"]) == 12 and validate_episode(ep) == []


def test_pooled_precision_recall_are_not_distorted_by_clean_books():
    from auditenv.run import summarize
    rows = [{"policy": "oracle", "clean": c, "valid": True, "tp": tp, "fp": 0, "fn": 0, "precision": p, "recall": p,
             "f1": 1.0, "reward": 1.0, "inspection_steps": 0, "seconds": 0.0}
            for c, tp, p in ((True, 0, 0.0), (False, 3, 1.0))]
    s = summarize(rows)["oracle"]
    assert s["precision"] == 0.5 and s["micro_precision"] == 1.0 and s["micro_recall"] == 1.0


def test_requirements_matrix_points_to_real_tests():
    """docs/REQUIREMENTS.md must not cite tests that do not exist."""
    import re
    # Parametrized citations (`test_contract_check[C01]`, `test_paired_check[P01..P20]`) count too.
    cited = set(re.findall(r"`(test_[a-z0-9_]+)(?:\[[A-Za-z0-9.]+\])?`", (ROOT / "docs" / "REQUIREMENTS.md").read_text()))
    existing = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        existing |= set(re.findall(r"^def (test_[a-z0-9_]+)", f.read_text(), re.M))
    assert cited and not cited - existing, sorted(cited - existing)

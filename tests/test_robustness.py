"""Fuzzing: no input may crash the env, every call costs one step, nothing leaks."""

import json
import math
import random

from auditenv.env import MAX_STEPS, AuditEnv, Recorder
from tests.conftest import truth_of


def deep(n):
    root = cur = []
    for _ in range(n):
        nxt = []
        cur.append(nxt)
        cur = nxt
    return root


class Weird:
    pass


NASTY = [
    None, 0, -1, 3.5, math.nan, math.inf, "", "submit", b"bytes", [], {}, set(), Weird(), {1: 2},
    {"tool": None}, {"tool": 5}, {"tool": "list_records"}, {"tool": "list_records", "args": None},
    {"tool": "list_records", "args": {"table": "ledger", "limit": True}},
    {"tool": "list_records", "args": {"table": "ledger", "limit": 10 ** 30}},
    {"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger:-1"}},
    {"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger:99999999"}},
    {"tool": "list_records", "args": {"table": "ledger", "cursor": deep(5000)}},
    {"tool": "get_record", "args": {"table": "ledger", "id": "x" * 10_000}},
    {"tool": "get_record", "args": {"table": "ledger", "id": ["L-1"]}},
    {"tool": "get_document", "args": {"doc_id": {"$gt": ""}}},
    {"tool": "calculator", "args": {"expression": "9" * 30 + "*" + "9" * 30}},
    {"tool": "calculator", "args": {"expression": "(" * 99 + ")" * 99}},
    {"tool": "calculator", "args": {"expression": "1/3*3"}},
    {"tool": "list_records", "args": {"table": "ledger", "pad": "x" * 300_000}},
    {"tool": "../../private/truth.json"}, {"tool": "__class__"}, {"tool": "LIST_RECORDS", "args": {}},
    {"tool": "list_records", "args": {"table": "ledger"}, "extra": deep(3000)},
]


def test_nasty_actions_never_crash_and_cost_one_step(episodes, tmp_path):
    rec = Recorder(tmp_path / "t.jsonl")
    env = AuditEnv(mode="train", recorder=rec)
    env.reset(episode_dir=episodes["dirty"][0])
    for n, action in enumerate(NASTY, 1):
        obs, reward, term, trunc, info = env.step(action)
        assert obs["steps_remaining"] == MAX_STEPS - n and not term and not trunc and reward == 0.0
        json.dumps(obs)  # observation is always valid JSON
    rec.close()
    for line in (tmp_path / "t.jsonl").read_text().splitlines():
        json.loads(line)  # trajectory log stays valid JSON


def test_nasty_submissions_end_with_minus_one(episodes):
    payloads = [deep(100_000), {"episode_id": "x", "findings": deep(2000)}, math.nan, Weird(),
                {"episode_id": None, "findings": [{"type": "duplicate_invoice"}] * 10_000},
                {"findings": [], "episode_id": "ep", "x": 1}, "findings", [None] * 31,
                {"episode_id": "ep", "findings": [{"type": "fx_conversion", "affected_ids": ["A"], "currency": "KZT",
                 "claimed_amount_minor": 10 ** 40, "evidence_ids": [], "explanation": ""}]}]
    for payload in payloads:
        env = AuditEnv(mode="train")
        env.reset(episode_dir=episodes["dirty"][0])
        obs, reward, term, trunc, _ = env.step({"tool": "submit", "args": {"findings": payload}})
        assert term and not trunc and reward == -1.0
    env = AuditEnv(mode="train")
    env.reset(episode_dir=episodes["dirty"][0])
    obs, reward, term, *_ = env.step({"tool": "submit", "args": {"findings": deep(100_000)}, "pad": deep(10)})
    assert term and reward == -1.0


def test_random_fuzz_2000_actions(episodes):
    rng = random.Random(7)
    tools = ["list_records", "get_record", "get_document", "calculator", "nope", None, 7]
    values = [None, "", "ledger", "invoices", "fx", "payments", "L-0000000", "ledger:3", -1, 0, 49, 50, 51,
              "1+1", "1/0", "DOC-AAAAAAAA", [], {}, True, "2026-01-01:USD:KZT"]
    env = AuditEnv(mode="train")
    eps = episodes["dirty"] + episodes["clean"]
    env.reset(episode_dir=eps[0])
    steps = 0
    for i in range(2000):
        args = {rng.choice(["table", "cursor", "limit", "id", "doc_id", "expression", "junk"]): rng.choice(values)
                for _ in range(rng.randint(0, 4))}
        obs, reward, term, trunc, info = env.step({"tool": rng.choice(tools), "args": args})
        steps += 1
        assert obs["steps_remaining"] == MAX_STEPS - steps
        if term or trunc:
            assert trunc and steps == MAX_STEPS
            env.reset(episode_dir=eps[i % len(eps)])
            steps = 0


def test_returned_objects_are_fresh_copies(episodes):
    """Gymnasium check_env requirement: step must not return mutated shared buffers."""
    env = AuditEnv()
    env.reset(episode_dir=episodes["dirty"][0])
    a = {"tool": "list_records", "args": {"table": "invoices", "limit": 3}}
    first, *_ = env.step(a)
    snapshot = json.dumps(first, sort_keys=True)
    first["result"]["rows"][0]["total_minor"] = -1
    first["result"]["rows"].clear()
    a["args"]["limit"] = 1
    second, *_ = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 3}})
    assert json.dumps({**second, "steps_remaining": first["steps_remaining"]}, sort_keys=True) == snapshot


def test_rejected_inputs_do_not_echo_truth(episodes):
    ep = episodes["dirty"][0]
    env = AuditEnv(mode="eval")
    env.reset(episode_dir=ep)
    blob = ""
    for action in NASTY:
        obs, *_ = env.step(action)
        blob += json.dumps(obs)
    for issue in truth_of(ep)["issues"]:
        assert issue["issue_id"] not in blob

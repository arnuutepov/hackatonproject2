import json

from auditenv.env import MAX_STEPS, AuditEnv
from auditenv.generator import load_config
from tests.conftest import oracle_findings, truth_of


def submit(env, findings):
    return env.step({"tool": "submit", "args": {"findings": {"episode_id": env.episode_id, "findings": findings}}})


def test_reset_observation_contents(episodes):
    env = AuditEnv()
    obs, info = env.reset(episode_dir=episodes["dirty"][0])
    assert set(obs) == {"task", "manifest", "policy", "tools", "page_limit", "steps_remaining"}
    assert obs["steps_remaining"] == MAX_STEPS
    assert set(obs["tools"]) == {"list_records", "get_record", "get_document", "calculator", "submit"}
    for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                "record_counts", "rounding"):
        assert key in obs["manifest"]
    assert "seed" not in json.dumps(info)


def test_reset_from_seed_and_config_is_reproducible(tmp_path):
    actions = [{"tool": "list_records", "args": {"table": "ledger", "limit": 50}},
               {"tool": "list_records", "args": {"table": "payments", "cursor": "payments:3", "limit": 7}},
               {"tool": "calculator", "args": {"expression": "1+1"}},
               {"tool": "get_document", "args": {"doc_id": "DOC-NOPE"}}]

    def rollout(workdir):
        env = AuditEnv(workdir=workdir)
        obs, info = env.reset(seed=42, config=load_config())
        trace = [obs]
        for a in actions:
            trace.append(env.step(a)[:4])
        trace.append(submit(env, [])[:4])
        return json.dumps(trace, sort_keys=True)

    assert rollout(tmp_path / "a") == rollout(tmp_path / "b")


def test_pagination_limits_and_stable_cursors(episodes):
    env = AuditEnv()
    obs, _ = env.reset(episode_dir=episodes["dirty"][0])
    total = obs["manifest"]["record_counts"]["ledger"]
    seen, cursor = [], None
    while True:
        o, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": cursor, "limit": 50}})
        assert len(o["result"]["rows"]) <= 50
        seen += [r["row_id"] for r in o["result"]["rows"]]
        cursor = o["result"]["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == total
    a, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger:50", "limit": 5}})
    b, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger:50", "limit": 5}})
    assert a["result"] == b["result"]
    over, *_ = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 51}})
    assert over["error"]["code"] == "bad_args"


def test_invalid_actions_cost_a_step_and_do_not_end_episode(episodes):
    env = AuditEnv()
    env.reset(episode_dir=episodes["dirty"][0])
    bad = [None, "list_records", {"tool": "rm -rf"}, {"tool": "list_records", "args": {"table": "secrets"}},
           {"tool": "get_record", "args": {"table": "invoices", "id": "INV-NOPE"}},
           {"tool": "list_records", "args": {"table": "ledger", "cursor": "../../private"}},
           {"tool": "calculator", "args": {"expression": "__import__('os')"}},
           {"tool": "list_records", "args": "notadict"}]
    for n, action in enumerate(bad, 1):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["error"] is not None and len(obs["error"]["message"]) < 200
        assert reward == 0.0 and not terminated and not truncated
        assert obs["steps_remaining"] == MAX_STEPS - n


def test_exploration_reward_is_zero_and_submit_terminates(episodes):
    ep = episodes["dirty"][0]
    env = AuditEnv(mode="train")
    env.reset(episode_dir=ep)
    for _ in range(3):
        _, reward, *_ = env.step({"tool": "list_records", "args": {"table": "invoices"}})
        assert reward == 0.0
    obs, reward, terminated, truncated, info = submit(env, oracle_findings(truth_of(ep)))
    assert terminated and not truncated
    assert abs(reward - (1.0 - 0.002 * 3)) < 1e-9


def test_step_limit_truncates_with_empty_answer(episodes):
    env = AuditEnv(mode="train")
    env.reset(episode_dir=episodes["dirty"][0])
    for n in range(1, MAX_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step({"tool": "calculator", "args": {"expression": "1"}})
        if n < MAX_STEPS:
            assert not truncated
    assert truncated and not terminated and reward == -1.0  # empty answer on erroneous books


def test_submit_as_last_allowed_step_counts(episodes):
    env = AuditEnv(mode="train")
    env.reset(episode_dir=episodes["clean"][0])
    for _ in range(MAX_STEPS - 1):
        env.step({"tool": "calculator", "args": {"expression": "1"}})
    obs, reward, terminated, truncated, _ = submit(env, [])
    assert terminated and not truncated
    assert abs(reward - (1 - 0.002 * (MAX_STEPS - 1))) < 1e-9


def test_repeat_submit_after_end_is_rejected_without_new_reward(episodes):
    ep = episodes["dirty"][1]
    env = AuditEnv(mode="train")
    env.reset(episode_dir=ep)
    submit(env, [])
    obs, reward, terminated, truncated, _ = submit(env, oracle_findings(truth_of(ep)))
    assert obs["error"]["code"] == "episode_over" and reward == 0.0 and terminated


def test_eval_mode_hides_reward_and_diagnostics(episodes, tmp_path):
    ep = episodes["dirty"][0]
    log = tmp_path / "grader.jsonl"
    env = AuditEnv(mode="eval", grader_log=log)
    env.reset(episode_dir=ep)
    obs, reward, terminated, _, info = submit(env, oracle_findings(truth_of(ep)))
    assert terminated and reward == 0.0 and "metrics" not in info
    assert json.loads(log.read_text())["reward"] == 1.0


def test_malformed_submission_ends_with_minus_one(episodes):
    env = AuditEnv(mode="train")
    env.reset(episode_dir=episodes["dirty"][0])
    obs, reward, terminated, *_ = env.step({"tool": "submit", "args": {"findings": {"oops": True}}})
    assert terminated and reward == -1.0


def test_time_limit_truncates(episodes):
    env = AuditEnv(mode="train", time_limit_s=0.0)
    env.reset(episode_dir=episodes["clean"][0])
    obs, reward, terminated, truncated, _ = env.step({"tool": "calculator", "args": {"expression": "1"}})
    assert truncated and obs["error"]["code"] == "time_limit"


def test_reset_isolates_episodes(episodes):
    env = AuditEnv(mode="train")
    env.reset(episode_dir=episodes["dirty"][0])
    env.step({"tool": "calculator", "args": {"expression": "1"}})
    obs, _ = env.reset(episode_dir=episodes["dirty"][1])
    assert obs["steps_remaining"] == MAX_STEPS
    o, *_ = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})
    assert o["steps_remaining"] == MAX_STEPS - 1

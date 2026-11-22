"""No leakage: the agent side works without private files and never sees truth content."""

import json
import shutil

from auditenv.env import AuditEnv, Recorder
from auditenv.grader.client import GraderClient
from auditenv.policies import make_policy
from tests.conftest import truth_of

TRUTH_ONLY_KEYS = ("evidence_sets", "expected_amount_minor", "issue_id", "provenance", "truth")


def run(policy_name, ep, recorder=None, env_dir=None, mode="train"):
    env = AuditEnv(mode=mode, recorder=recorder, policy_name=policy_name)
    obs, info = env.reset(episode_dir=env_dir or ep, grader=GraderClient.for_episode(ep))
    policy = make_policy(policy_name, ep)
    policy.reset(obs, info)
    while True:
        obs, reward, term, trunc, info = env.step(policy.act(obs))
        policy.observe(obs)
        if term or trunc:
            return reward


def test_env_runs_on_public_files_only(episodes, tmp_path):
    ep = episodes["dirty"][0]
    public_only = tmp_path / ep.name
    shutil.copytree(ep / "public", public_only / "public")
    assert not (public_only / "private").exists()
    assert run("reconcile", ep, env_dir=public_only) == run("reconcile", ep)
    assert run("oracle", ep, env_dir=public_only) == 1.0


def test_trajectories_contain_no_truth(episodes, tmp_path):
    path = tmp_path / "traj.jsonl"
    rec = Recorder(path)
    for ep in episodes["dirty"] + episodes["clean"]:
        for name in ("random", "reconcile", "flag_all"):
            run(name, ep, recorder=rec)
    rec.close()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"} <= set(rows[0])
    blob = "\n".join(json.dumps(r["observation"]) for r in rows)
    for key in TRUTH_ONLY_KEYS:
        assert f'"{key}"' not in blob
    for ep in episodes["dirty"]:
        for issue in truth_of(ep)["issues"]:
            assert issue["issue_id"] not in blob


def test_public_files_have_no_private_content(episodes):
    for ep in episodes["dirty"]:
        text = "".join(p.read_text(encoding="utf-8") for p in (ep / "public").iterdir())
        assert "issue_id" not in text and "evidence_sets" not in text and '"seed"' not in text
        for issue in truth_of(ep)["issues"]:
            assert issue["issue_id"] not in text

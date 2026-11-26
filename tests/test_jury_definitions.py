"""One test per published jury check (organizers' definitions, 20 Nov 2026), through judge_adapter.

The checks themselves live in auditenv/jury_checks.py (also run by `python -m auditenv.run check`
and `python -m auditenv.run jury-definitions`); here each check ID is a separate pytest test.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import pytest

from auditenv import jury_checks as jc
from auditenv.grader.matcher import match
from auditenv.grader.reward import score

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("auditenv.judge").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return jc.Context(tmp_path_factory.mktemp("jury-definitions"))


@pytest.fixture(scope="module")
def checks(ctx):
    return jc.all_checks(ctx)


@pytest.mark.parametrize("cid", [f"C{i:02d}" for i in range(1, 21)])
def test_contract_check(checks, cid):
    checks["contract"][cid]()


@pytest.mark.parametrize("tid", [f"T{i:02d}" for i in range(1, 21)])
def test_matcher_and_reward_check(checks, tid):
    checks["matcher"][tid]()


@pytest.mark.parametrize("mid", [f"M{i:02d}" for i in range(1, 13)])
def test_matcher_example(checks, mid):
    checks["examples"][mid]()


@pytest.fixture(scope="module")
def jury_layout_checks(ctx):
    return jc._matcher_block(ctx, layout="jury")


@pytest.mark.parametrize("cid", [f"T{i:02d}" for i in range(1, 21)] + [f"M{i:02d}" for i in range(1, 13)])
def test_matcher_check_on_a_flat_jury_made_fixture(jury_layout_checks, cid):
    """The same 32 checks when the fixture is one jury-made folder for all teams (organizers, 21.11):
    flat files, .json tables, minimal manifest, no policy.json unless tau is published, truth as a list
    with evidence_ids. T01 also runs an episode on it in closed mode (the env stages the folder)."""
    jury_layout_checks[cid]()


@pytest.mark.parametrize("did", [f"D{i:02d}" for i in range(1, 21)])
def test_closed_scenario_episode(checks, did):
    checks["episodes"][did]()


@pytest.mark.parametrize("pid", [f"P{i:02d}" for i in range(1, 21)])
def test_paired_check(checks, pid):
    checks["paired"][pid]()


# ------------------------------------------------------------------ the adapter as the jury sees it
def test_judge_adapter_is_importable_from_the_repository_root_with_three_entry_points():
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module("judge_adapter")
    for name in ("generate_episode", "make_env", "score_episode"):
        assert callable(getattr(module, name))
    env = module.make_env()
    assert callable(env.reset) and callable(env.step)


def test_generate_episode_writes_into_output_dir_and_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    jc.generate_episode(5, {"force_clean": True}, str(a))
    jc.generate_episode(5, {"mode": "train", "team_config": {"force_clean": True}}, str(b))
    assert sorted(p.name for p in a.iterdir()) == ["private", "public"]
    assert sorted(p.name for p in (a / "public").iterdir()) == sorted(
        ["manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"])
    assert sorted(p.name for p in (a / "private").iterdir()) == ["provenance.json", "truth.json"]
    assert jc.tree_hash(a) == jc.tree_hash(b)


def test_train_mode_without_episode_dir_uses_the_team_generator(tmp_path):
    env = jc.make_env()
    obs, info = env.reset(11, {"mode": "train", "team_config": {"force_clean": True}})
    again, _ = jc.make_env().reset(11, {"mode": "train", "team_config": {"force_clean": True}})
    assert obs == again and info["mode"] == "train"
    out = jc.act(env, "submit", findings={"episode_id": obs["manifest"]["episode_id"], "findings": []})
    assert out[2] and out[1] == pytest.approx(1.0)  # clean books, empty answer, no inspection: terminal reward 1


def test_repeated_closed_reset_restarts_the_same_episode(ctx):
    env = jc.make_env()
    first = jc.closed(env, ctx.E)
    trace = [jc.act(env, "list_records", table="ledger", cursor=None, limit=50)]
    second = jc.closed(env, ctx.E, seed=999)  # the seed does not change a loaded episode
    assert first == second
    assert [jc.act(env, "list_records", table="ledger", cursor=None, limit=50)] == trace


def test_score_episode_contract_shapes(ctx):
    sub = jc.oracle_findings(ctx.E)
    out = jc.score_episode(str(ctx.E), sub, 0)
    assert set(out) == {"tp", "fp", "fn", "precision", "recall", "f1", "return"} and out["return"] == 1.0
    bad = jc.score_episode(str(ctx.E), {"episode_id": sub["episode_id"], "findings": "nope"}, 0)
    assert bad == {k: None for k in ("tp", "fp", "fn", "precision", "recall", "f1")} | {"return": -1.0}
    with pytest.raises(ValueError):
        jc.score_episode(str(ctx.E), sub, -1)


def test_grader_journal_file_records_invalid_submissions(ctx, tmp_path, monkeypatch):
    log = tmp_path / "grader.jsonl"
    monkeypatch.setenv("AUDITENV_GRADER_LOG", str(log))
    jc.score_episode(str(ctx.E), {"episode_id": "x", "findings": "not a list"}, 3)
    entry = json.loads(log.read_text().splitlines()[-1])
    assert entry["valid"] is False and "findings" in entry["error"] and entry["return"] == -1.0


def test_score_episode_ignores_episode_id_value(ctx):
    """Hard-jury audit, item 1: score_episode grades the episode in episode_dir. A missing, null,
    padded or foreign episode_id is not in the published structural errors and must not give -1."""
    sub = jc.oracle_findings(ctx.E)
    reference = jc.score_episode(str(ctx.E), sub, 12)
    assert reference["tp"] == len(jc.truth_of(ctx.E)["issues"]) and reference["return"] > 0
    findings = sub["findings"]
    for variant in ({"findings": findings}, {"episode_id": None, "findings": findings},
                    {"episode_id": sub["episode_id"] + " ", "findings": findings},
                    {"episode_id": "another-episode", "findings": findings}):
        assert jc.score_episode(str(ctx.E), variant, 12) == reference, variant


def test_submit_through_adapter_accepts_any_episode_id(ctx):
    findings = jc.oracle_findings(ctx.E)["findings"]
    for variant in ({"findings": findings}, {"episode_id": None, "findings": findings},
                    {"episode_id": "ep-somewhere-else", "findings": findings}):
        env = jc.make_env()
        jc.closed(env, ctx.E)
        obs, _, terminated, truncated, _ = jc.act(env, "submit", findings=variant)
        assert terminated and not truncated and obs == {"result": {"accepted": True}, "error": None,
                                                        "remaining_steps": 99}, variant
        assert env.final_submission == variant
        assert jc.score_episode(str(ctx.E), env.final_submission, env.inspection_steps)["return"] == 1.0
    # train mode goes through the separate grader process: same rule there
    env = jc.make_env()
    env.reset(0, {"mode": "train", "episode_dir": str(ctx.E), "team_config": {}})
    assert jc.act(env, "submit", findings={"findings": findings})[1] == 1.0


# ------------------------------------------------------------------ matcher conventions, unit level
def _f(**kw):
    base = {"type": "partial_payment", "affected_ids": ["P"], "claimed_amount_minor": 5, "currency": "KZT",
            "evidence_ids": ["D", "L"], "explanation": ""}
    return {**base, **kw}


TRUTH = {"episode_id": "e", "issues": [{"issue_id": "I", "type": "partial_payment", "affected_ids": ["P"],
                                       "expected_amount_minor": 5, "currency": "KZT", "evidence_sets": [["D", "L"]]}]}


def test_ids_are_exact_strings_no_trimming_or_case_folding():
    for ids in (["P "], [" P"], ["p"]):
        assert score({"episode_id": "e", "findings": [_f(affected_ids=ids)]}, TRUTH, 0)["tp"] == 0
    assert score({"episode_id": "e", "findings": [_f(evidence_ids=["D ", "L"])]}, TRUTH, 0)["tp"] == 0


def test_truth_lists_need_not_be_sorted_after_renaming():
    truth = {"episode_id": "e", "issues": [{**TRUTH["issues"][0], "affected_ids": ["z", "a"],
                                            "evidence_sets": [["L2", "D2"]]}]}
    sub = {"episode_id": "e", "findings": [_f(affected_ids=["a", "z"], evidence_ids=["D2", "L2"])]}
    assert score(sub, truth, 0)["tp"] == 1


def test_evidence_must_exist_when_the_universe_is_known():
    sub = {"episode_id": "e", "findings": [_f(evidence_ids=["D", "L", "GHOST"])]}
    assert score(sub, TRUTH, 0)["tp"] == 1  # unit level without public data: no existence check
    assert score(sub, TRUTH, 0, 0, frozenset({"D", "L", "P"}))["tp"] == 0
    assert score(sub, TRUTH, 0, 0, frozenset({"D", "L", "P", "GHOST"}))["tp"] == 1


def test_lexicographic_tie_break_among_maximum_matchings():
    issues = [{"issue_id": i, "type": "t", "affected_ids": ["x"], "expected_amount_minor": 1, "currency": "C",
               "evidence_sets": [["e"]]} for i in ("B", "A", "C")]
    finding = {"type": "t", "affected_ids": ["x"], "claimed_amount_minor": 1, "currency": "C", "evidence_ids": ["e"]}
    assert match([finding] * 2, issues) == [(0, "A"), (1, "B")]
    assert match([finding] * 4, issues) == [(0, "A"), (1, "B"), (2, "C")]


def test_long_ids_are_valid_and_compared_exactly():
    """No id length limit is published: a long id is valid; only the exact id matches (1.9.7)."""
    long_id = "x" * 1000
    truth = {"episode_id": "e", "issues": [{**TRUTH["issues"][0], "affected_ids": [long_id]}]}
    assert score({"episode_id": "e", "findings": [_f(affected_ids=[long_id])]}, truth, 0)["tp"] == 1
    other = score({"episode_id": "e", "findings": [_f(affected_ids=["x" * 1001])]}, truth, 0)
    assert other["valid"] is True and (other["tp"], other["fp"], other["fn"]) == (0, 1, 1)


def test_matrix_documents_every_scenario_with_a_valid_config():
    from auditenv.generator.config import resolve_config
    matrix = json.loads((ROOT / "configs" / "jury_matrix.json").read_text())["scenarios"]
    assert sorted(matrix) == [f"D{i:02d}" for i in range(1, 21)]
    for spec in matrix.values():
        resolve_config(spec["config"])


def test_scenario_names_select_the_documented_matrix_config(tmp_path):
    """generate_episode(seed, "D07", dir) is byte-identical to passing the D07 settings by hand, so a
    jury cannot mistype a scenario; train-mode reset accepts the name too."""
    from auditenv.judge import SCENARIO_NAMES, scenario_config
    assert len(SCENARIO_NAMES) == 20
    for name in ("D01", "D09", "D16", "D20"):
        by_name, by_dict = tmp_path / f"{name}-name", tmp_path / f"{name}-dict"
        jc.generate_episode(77, name, str(by_name))
        jc.generate_episode(77, scenario_config(name), str(by_dict))
        assert jc.tree_hash(by_name) == jc.tree_hash(by_dict)
    jc.generate_episode(77, {"mode": "train", "team_config": "d07"}, str(tmp_path / "wrapped"))
    jc.generate_episode(77, scenario_config("D07"), str(tmp_path / "d07"))
    assert jc.tree_hash(tmp_path / "wrapped") == jc.tree_hash(tmp_path / "d07")
    env = jc.make_env()
    obs, _ = env.reset(77, {"mode": "train", "team_config": "D07"})
    assert obs["manifest"]["episode_id"] == json.loads((tmp_path / "d07" / "public" / "manifest.json").read_text())["episode_id"]

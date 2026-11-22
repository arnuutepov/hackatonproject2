"""examples/jury_run.py and docs/JURY.md: the jury's own policies, seeds and key through judge_adapter."""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

from auditenv import jury_checks
from auditenv.judge import generate_episode, make_env

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jury_run", ROOT / "examples" / "jury_run.py")
jury_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jury_run)
JURY_MD = (ROOT / "docs" / "JURY.md").read_text(encoding="utf-8")

POLICY_FILE = '''
class Looker:
    """Looks at one ledger page, then submits nothing."""
    def reset(self, observation, info):
        self.episode_id = info["episode_id"]
        self.looked = False
    def act(self, observation):
        if not self.looked:
            self.looked = True
            return {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 10}}
        return {"name": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": []}}}

class Crashes:
    def act(self, observation):
        raise RuntimeError("bug in the policy")

class NeverSubmits:
    def act(self, observation):
        return {"name": "calculator", "args": {"expression": "1+1"}}
'''


@pytest.fixture
def policies(tmp_path, monkeypatch):
    folder = tmp_path / "policies"
    folder.mkdir()
    (folder / "jurypolicies.py").write_text(POLICY_FILE)
    monkeypatch.syspath_prepend(str(folder))
    sys.modules.pop("jurypolicies", None)
    return folder


def _run(tmp_path, *args) -> dict:
    out = tmp_path / "out"
    assert jury_run.main([*args, "--out", str(out)]) == 0
    return json.loads((out / "results.json").read_text())


def test_builtin_policies_give_the_reference_grades(tmp_path):
    result = _run(tmp_path, "--policy", "reconcile", "--policy", "oracle", "--seeds", "3,4", "--config", "D06")
    by = {(r["policy"], r["seed"]): r for r in result["rows"]}
    for seed in (3, 4):
        ep = tmp_path / "ref" / str(seed)
        generate_episode(seed, "D06", str(ep))
        for name in ("reconcile", "oracle"):
            want = jury_checks.run_policy(name, ep)
            assert {k: by[(name, seed)][k] for k in jury_run.GRADE} == {k: want[k] for k in jury_run.GRADE}
    assert all(by[("oracle", s)]["return"] == 1.0 for s in (3, 4))


def test_a_jury_policy_class_is_loaded_from_its_folder(tmp_path, policies):
    result = _run(tmp_path, "--policy-path", str(policies), "--policy", "jurypolicies:Looker",
                  "--recipe", str(ROOT / "configs" / "hidden_like.json"))
    rows = result["rows"]
    assert len(rows) == 20 and all(r["submitted"] and r["inspection_steps"] == 1 for r in rows)
    assert {r["return"] for r in rows} <= {1.0 - 0.002, -1.0}  # clean books: right; books with errors: wrong
    assert (tmp_path / "out" / "results.csv").read_text().count("\n") == 21


def test_a_crashing_or_silent_policy_scores_minus_one_and_the_run_goes_on(tmp_path, policies):
    result = _run(tmp_path, "--policy", "jurypolicies:Crashes", "--policy", "jurypolicies:NeverSubmits",
                  "--policy", "oracle", "--seeds", "5", "--config", "D02")
    by = {r["policy"]: r for r in result["rows"]}
    assert by["jurypolicies:Crashes"]["return"] == -1.0 and "bug in the policy" in by["jurypolicies:Crashes"]["error"]
    assert by["jurypolicies:NeverSubmits"]["return"] == -1.0 and not by["jurypolicies:NeverSubmits"]["submitted"]
    assert by["jurypolicies:NeverSubmits"]["inspection_steps"] == 100
    assert by["oracle"]["return"] == 1.0


def test_paired_copies_give_the_same_grade_to_the_reference_policies(tmp_path):
    result = _run(tmp_path, "--policy", "oracle", "--policy", "reconcile", "--seeds", "1,2,3",
                  "--config", "interacting", "--paired")
    assert all(r["paired_same"] for r in result["rows"])
    assert {s["paired_same"] for s in result["summary"]} == {"3/3"}
    twin = next((tmp_path / "out" / "paired").iterdir())
    original = tmp_path / "out" / "episodes" / twin.name
    assert (twin / "public" / "ledger.csv").read_text() != (original / "public" / "ledger.csv").read_text()


def test_the_key_changes_the_episodes_and_keeps_them_reproducible(tmp_path, monkeypatch):
    plain = _run(tmp_path / "a", "--policy", "oracle", "--seeds", "9", "--config", "D03")
    monkeypatch.setenv("AUDITENV_SECRET", "a-jury-key")
    keyed = _run(tmp_path / "b", "--policy", "oracle", "--seeds", "9", "--config", "D03")
    again = _run(tmp_path / "c", "--policy", "oracle", "--seeds", "9", "--config", "D03")
    ids = [json.loads((p / "out" / "episodes" / "00-seed9" / "public" / "manifest.json").read_text())["episode_id"]
           for p in (tmp_path / "a", tmp_path / "b", tmp_path / "c")]
    assert ids[0] != ids[1] == ids[2]
    assert plain["rows"][0]["return"] == keyed["rows"][0]["return"] == again["rows"][0]["return"] == 1.0


def test_unknown_policy_spec_is_refused():
    with pytest.raises(SystemExit):
        jury_run.load_policy("no-colon")


# ------------------------------------------------------------------- docs/JURY.md stays true
def _commands() -> list[list[str]]:
    """The jury_run.py arguments of every command in the bash blocks of docs/JURY.md."""
    text = "\n".join(re.findall(r"```bash\n(.*?)```", JURY_MD, re.S)).replace("\\\n", " ")
    cmds = [shlex.split(line.split("examples/jury_run.py", 1)[1].split("#")[0])
            for line in text.splitlines() if "examples/jury_run.py" in line]
    assert len(cmds) >= 5
    return cmds


def test_every_jury_md_command_parses():
    parser = jury_run.build_parser()
    for argv in _commands():
        parser.parse_args(argv)


def test_jury_md_names_exactly_what_a_policy_sees(tmp_path):
    ep = tmp_path / "ep"
    generate_episode(5, "D05", str(ep))
    env = make_env()
    obs, info = env.reset(5, {"mode": "closed", "episode_dir": str(ep), "team_config": {}})
    step, reward, *_rest, step_info = env.step({"name": "nope", "args": {}})
    for key in [*obs, *info, *step, *step_info, step["error"]["code"]]:
        assert f"`{key}`" in JURY_MD, key
    assert reward == 0.0 and obs["remaining_steps"] == 100
    for name in ("list_records", "get_record", "get_document", "calculator", "submit"):
        assert f"`{name}`" in JURY_MD


def test_jury_md_presets_and_files_exist():
    presets = {p.stem for p in (ROOT / "auditenv" / "presets").glob("*.json")}
    named = set(re.findall(r"`(\w+)`", JURY_MD.split("имя пресета", 1)[1].split(")", 1)[0]))
    assert named == presets
    for path in re.findall(r"`((?:configs|auditenv|examples|tests|docs)/[\w./]+)`", JURY_MD):
        assert (ROOT / path).exists(), path

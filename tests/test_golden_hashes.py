"""Golden hashes (auditenv/golden.py). The committed episodes are pinned file by file in
episodes/SHA256SUMS (checkable with `sha256sum -c`), and a fresh generation of every jury scenario
D01-D20 on seeds 1 and 2 is pinned in episodes/SCENARIO_DIGESTS.json. CI runs this on Python 3.11,
3.12 and 3.13 and in the Docker image, so passing everywhere means one seed gives one episode
whatever the interpreter."""

from __future__ import annotations

import json
import re

import pytest

from auditenv import ENV_VERSION, golden


def test_committed_episodes_match_the_golden_hashes():
    text = golden.SUMS.read_text(encoding="utf-8")
    assert all(re.fullmatch(r"[0-9a-f]{64}  (dev|train)/ep-[0-9a-f]{12}/(public|private)/[a-z_]+\.(csv|jsonl|json)", line)
               for line in text.splitlines())
    assert len(text.splitlines()) == len(golden.episode_files()) == 160
    assert text == golden.sums_text()


PINNED = json.loads(golden.DIGESTS.read_text(encoding="utf-8"))
COMMITTED = sorted({"/".join(line.split("  ", 1)[1].split("/")[:2])
                    for line in golden.SUMS.read_text(encoding="utf-8").splitlines()})


def test_pinned_digests_cover_every_scenario_and_seed_of_this_env_version():
    from auditenv.judge import SCENARIO_NAMES
    assert PINNED["env_version"] == ENV_VERSION, "episodes changed version: run `python -m auditenv.run golden --write`"
    assert sorted(PINNED["digests"]) == sorted(f"{n}/seed{s}" for n in SCENARIO_NAMES for s in golden.SCENARIO_SEEDS)
    assert len(COMMITTED) == 20


@pytest.mark.parametrize("key", sorted(PINNED["digests"]))
def test_fresh_scenario_generation_matches_its_golden_digest(tmp_path, key):
    from auditenv.judge import generate_episode
    name, seed = key.split("/seed")
    generate_episode(int(seed), name, str(tmp_path / "ep"))
    assert golden.tree_digest(tmp_path / "ep") == PINNED["digests"][key]


@pytest.mark.parametrize("episode", COMMITTED)
def test_committed_episode_matches_its_golden_hashes(episode):
    lines = [line for line in golden.SUMS.read_text(encoding="utf-8").splitlines()
             if line.split("  ", 1)[1].startswith(episode + "/")]
    assert len(lines) == 8  # 6 public + 2 private files
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert golden._sha(golden.EPISODES / rel) == digest, rel


def test_a_changed_byte_changes_the_digest(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.csv").write_bytes(b"1,2\n")
    before = golden.tree_digest(tmp_path)
    (tmp_path / "a" / "x.csv").write_bytes(b"1,3\n")
    assert golden.tree_digest(tmp_path) != before

"""make_env() must not leave scratch folders behind. The image runs on a 1 GB tmpfs and a jury
harness may make one env per episode and never call close(). Before 1.8.4 one `make check` left
about 1200 `auditenv-judge-*` folders in the temporary folder."""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

from auditenv.judge import JudgeEnv, generate_episode, make_env


def _ours() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("auditenv-judge-*"))


def test_close_removes_the_whole_scratch_folder(tmp_path):
    generate_episode(3, "D04", str(tmp_path / "ep"))
    env = make_env()
    work = env._workdir
    env.reset(0, {"mode": "closed", "episode_dir": str(tmp_path / "ep"), "team_config": {}})
    env.step({"name": "submit", "args": {"findings": {"findings": []}}})
    env.close()
    assert not work.exists()
    env.reset(0, {"mode": "closed", "episode_dir": str(tmp_path / "ep"), "team_config": {}})  # still usable
    env.close()
    assert not work.exists()


def test_an_env_that_is_never_closed_cleans_up_when_dropped(tmp_path):
    generate_episode(4, "D10", str(tmp_path / "ep"))
    before = _ours()
    for seed in range(5):
        env = make_env()
        env.reset(seed, {"mode": "closed", "episode_dir": str(tmp_path / "ep"), "team_config": {}})
        env.reset(seed, {"mode": "train", "team_config": "D03"})
        del env
    gc.collect()
    assert _ours() - before == set()


def _flat_copy(src: Path, dst: Path) -> Path:
    """A jury-style flat fixture (tables and truth side by side): this one is converted, so staged."""
    dst.mkdir()
    for part in ("public", "private"):
        for f in (src / part).iterdir():
            (dst / f.name).write_bytes(f.read_bytes())
    return dst


def test_the_jury_episode_folder_is_never_removed(tmp_path):
    generate_episode(6, "D12", str(tmp_path / "ep"))
    before = sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "ep").rglob("*"))
    env = make_env()
    for _ in range(3):  # our own layout is used in place: later resets and close must not touch it
        env.reset(0, {"mode": "closed", "episode_dir": str(tmp_path / "ep"), "team_config": {}})
    env.reset(1, {"mode": "train", "team_config": "D02"})
    env.close()
    del env
    gc.collect()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "ep").rglob("*")) == before


def test_only_the_current_staged_copy_is_kept(tmp_path):
    generate_episode(5, "D07", str(tmp_path / "ep"))
    flat = _flat_copy(tmp_path / "ep", tmp_path / "flat")
    env = JudgeEnv(workdir=tmp_path / "work")
    for _ in range(4):
        env.reset(0, {"mode": "closed", "episode_dir": str(flat), "team_config": {}})
    assert len(list((tmp_path / "work" / "staged").iterdir())) == 1
    assert len(list(flat.iterdir())) == 8
    (tmp_path / "work" / "keep.txt").write_text("caller's file")
    env.close()
    assert (tmp_path / "work" / "keep.txt").exists()  # a caller's folder keeps its own content
    assert not (tmp_path / "work" / "staged").exists()

"""Golden hashes: the committed episodes and fresh D01-D20 generations, pinned by SHA-256.

episodes/SHA256SUMS lists every committed episode file in the format of `sha256sum`, so anyone can
check the open episodes without Python:  cd episodes && sha256sum -c SHA256SUMS
episodes/SCENARIO_DIGESTS.json pins one digest per jury scenario (D01-D20, seeds 1 and 2) of a
fresh generation. CI runs the tests on Python 3.11, 3.12 and 3.13 (and the Docker image), so the
same digests on every interpreter show the generator does not depend on the Python version, the
platform, hash randomization or the file system order.

    python -m auditenv.run golden            # verify
    python -m auditenv.run golden --write    # rewrite after an intended change (bump ENV_VERSION)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "episodes"
SUMS = EPISODES / "SHA256SUMS"
DIGESTS = EPISODES / "SCENARIO_DIGESTS.json"
SCENARIO_SEEDS = (1, 2)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_files(root: Path = EPISODES) -> list[Path]:
    return sorted(p for split in ("dev", "train") for p in (root / split).rglob("*") if p.is_file())


def sums_text(root: Path = EPISODES) -> str:
    return "".join(f"{_sha(p)}  {p.relative_to(root).as_posix()}\n" for p in episode_files(root))


def tree_digest(directory: Path) -> str:
    """One digest for a folder: relative paths and file digests, in a fixed order."""
    h = hashlib.sha256()
    for p in sorted(x for x in Path(directory).rglob("*") if x.is_file()):
        h.update(p.relative_to(directory).as_posix().encode() + b"\0" + _sha(p).encode() + b"\n")
    return h.hexdigest()


def scenario_digests() -> dict[str, str]:
    from auditenv.judge import SCENARIO_NAMES, generate_episode

    out = {}
    with tempfile.TemporaryDirectory(prefix="auditenv-golden-") as tmp:
        for name in SCENARIO_NAMES:
            for seed in SCENARIO_SEEDS:
                d = Path(tmp) / f"{name}-{seed}"
                generate_episode(seed, name, str(d))
                out[f"{name}/seed{seed}"] = tree_digest(d)
    return out


def verify() -> list[str]:
    problems = []
    if not SUMS.exists() or SUMS.read_text(encoding="utf-8") != sums_text():
        problems.append("episodes/SHA256SUMS does not match the committed episode files")
    want = json.loads(DIGESTS.read_text(encoding="utf-8"))["digests"] if DIGESTS.exists() else {}
    got = scenario_digests()
    problems += [f"{k}: fresh generation {got.get(k)} != golden {v}" for k, v in want.items() if got.get(k) != v]
    if set(want) != set(got):
        problems.append("episodes/SCENARIO_DIGESTS.json does not list every scenario and seed")
    return problems


def write() -> None:
    from auditenv.common import write_json
    from auditenv import ENV_VERSION

    SUMS.write_text(sums_text(), encoding="utf-8", newline="\n")
    write_json(DIGESTS, {"env_version": ENV_VERSION, "seeds": list(SCENARIO_SEEDS),
                         "how": "auditenv/golden.py: tree_digest of generate_episode(seed, name, dir)",
                         "digests": scenario_digests()})

"""Export scripted expert trajectories as chat transcripts for supervised warm-up (SFT).

The expert is the all-skills script of auditenv/wrappers.py: it reads every table, runs the four
checks on PUBLIC records only and submits. It reaches F1 = 1.0 on the open episodes without ever
reading truth, so its transcripts are safe training data: observations come from the env, which
never exposes private files. Each transcript uses the same chat template as auditenv.llm
(system prompt, compact JSON observations as user turns, one JSON action per assistant turn), so a
model warmed up on it starts RL already producing valid tool calls and valid findings, instead of
learning JSON from -1 rewards.

    python examples/export_sft.py --episodes episodes/train --out runs/sft/expert.jsonl
    python -m auditenv.run pool --n 200 --out runs/pool && python examples/export_sft.py --pool runs/pool --routes 3

Rows: {"episode_id", "route", "messages": [...], "reward", "steps", "tp", "fp", "fn", "f1"}.
Train on assistant turns only.

Which transcripts are kept: the CORRECT ones (no false positive, no miss), whatever their length.
A reward threshold would mix the two: with 0.002 per inspection step, a flawless 51-step
investigation scores 0.898 and a 0.9 cut would drop it, so long books would vanish from the
demonstrations. Length stays in each row (`steps`) for whoever wants to weight by it;
`--min-reward` is still there as an extra filter.

Routes: route 0 is the scripted order (read the four tables, run the four checks, submit).
`--routes K` adds K-1 other orders per episode, drawn with a fixed seed among the orders that read
a table before every check that needs it (for example: invoices, duplicate check, payments,
ledger, suspense check, ...). One route per task teaches one habit; several teach the dependency.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditenv.env.core import AuditEnv, Recorder  # noqa: E402
from auditenv.llm import render_observation, system_prompt  # noqa: E402
from auditenv.training import EpisodePool  # noqa: E402
from auditenv.wrappers import OPTIONS, REQUIRES, TABLE_OF, AuditOptionsEnv  # noqa: E402


def routes(key: str, k: int) -> list[list[str]]:
    """Route 0 = OPTIONS order; then up to k-1 different orders where every check follows the
    reads it needs and submit comes last, drawn with random.Random(key)."""
    import random

    out = [list(OPTIONS)]
    rng = random.Random(f"sft-routes|{key}")
    for _ in range(50 * k):
        if len(out) >= k:
            break
        done: set[str] = set()
        todo = [o for o in OPTIONS if o != "submit"]
        order = []
        while todo:
            ready = [o for o in todo if REQUIRES.get(o, set()) <= done]
            o = rng.choice(ready)
            order.append(o)
            todo.remove(o)
            if o in TABLE_OF:
                done.add(TABLE_OF[o])
        order.append("submit")
        if order not in out:
            out.append(order)
    return out


def expert_transcript(episode_dir: Path, work: Path, route: list[str] | None = None) -> dict:
    traj = work / "trajectory.jsonl"
    traj.unlink(missing_ok=True)
    (work / "grader_log.jsonl").unlink(missing_ok=True)
    env = AuditEnv(mode="train", workdir=work / "cache", grader_log=work / "grader_log.jsonl",
                   recorder=Recorder(traj), policy_name="expert_sft")
    wrapped = AuditOptionsEnv(env, mask=True)
    wrapped.reset(episode_dir=episode_dir)
    for name in route or OPTIONS:  # default: the scripted order (read everything, run all checks, submit)
        _, _, terminated, truncated, _ = wrapped.step(OPTIONS.index(name))
        if terminated or truncated:
            break
    rows = [json.loads(line) for line in traj.read_text(encoding="utf-8").splitlines()]
    first = rows[0]["observation"]
    messages = [{"role": "system", "content": system_prompt(first["tools"])},
                {"role": "user", "content": render_observation(first)}]
    for row in rows[1:]:
        messages.append({"role": "assistant", "content": json.dumps(row["action"], sort_keys=True)})
        messages.append({"role": "user", "content": render_observation(row["observation"])})
    graded = json.loads((work / "grader_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    return {"episode_id": rows[0]["episode_id"], "route": list(route or OPTIONS), "messages": messages,
            "reward": rows[-1]["reward"], "steps": rows[-1]["step"],
            **{k: graded[k] for k in ("tp", "fp", "fn", "f1")}}


def correct(row: dict) -> bool:
    return row["fp"] == 0 and row["fn"] == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episodes", type=Path, help="a directory of episode folders (e.g. episodes/train)")
    src.add_argument("--pool", type=Path, help="a pool made by `python -m auditenv.run pool`")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "sft" / "expert.jsonl")
    ap.add_argument("--routes", type=int, default=1, help="different valid orders per episode (route 0 = scripted)")
    ap.add_argument("--min-reward", type=float, default=None,
                    help="optional extra filter; by default every correct transcript is kept, whatever its length")
    args = ap.parse_args(argv)
    if args.pool:
        episodes = EpisodePool(args.pool).all()
    else:
        episodes = sorted(p for p in args.episodes.iterdir() if (p / "public" / "manifest.json").exists())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sft-"))
    kept = 0
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            total = 0
            for ep in episodes:
                for route in routes(Path(ep).name, args.routes):
                    row = expert_transcript(ep, work, route)
                    total += 1
                    if correct(row) and (args.min_reward is None or row["reward"] >= args.min_reward):
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        kept += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    extra = "" if args.min_reward is None else f" and reward >= {args.min_reward}"
    print(f"{kept}/{total} transcripts correct{extra} ({len(episodes)} episodes x {args.routes} routes) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

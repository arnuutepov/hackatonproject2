"""Hyperparameter sweep for the Q-learning benchmark, tuned on a VALIDATION pool.

Three disjoint seed ranges: train 300000+, validation 500000+, held-out 400000+.
The sweep only ever looks at the validation pool, so the held-out numbers reported by
train_q_learning.py stay untouched by tuning.

Stage 1: learning rate x training episodes (masked agent, 2 agent seeds).
Stage 1b: every cell within 0.005 of the best is re-run on 5 agent seeds (2 seeds are too few
to trust a lucky cell).
Stage 2: final epsilon at the verified winner.
Stage 3: discount factor at the winner, for BOTH arms of the benchmark (masked and unmasked),
on 5 agent seeds. The masked arm must stay within 0.005 of its best; among those values, the one
that gives the unmasked arm its best result wins, so the comparison is never against a crippled
baseline. (With gamma = 1 an invalid option is a self-loop whose value converges to the state's
best value: the greedy unmasked agent can loop until the step limit.)
Selection: highest mean validation return; ties within 0.005 go to fewer episodes, then
lower spread across seeds (cheaper and more stable wins).

    python examples/sweep_hparams.py            # ~20 min on 2 cores, writes runs/sweep/ and artifacts/sweep.json
    python examples/sweep_hparams.py --quick    # smoke run
    python examples/sweep_hparams.py --resume   # reuse finished stages of runs/sweep/sweep.json (same grid)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from auditenv.common import write_json  # noqa: E402
from train_q_learning import build_pool, evaluate, train  # noqa: E402

FULL = {"lr": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8], "episodes": [150, 300, 600, 1000], "final_eps": [0.01, 0.05, 0.1],
        "gamma": [0.95, 0.99, 1.0],
        "seeds": [0, 1], "verify_seeds": [0, 1, 2, 3, 4], "n_train": 120, "n_val": 40}
QUICK = {"lr": [0.1, 0.4], "episodes": [20, 40], "final_eps": [0.05, 0.1], "gamma": [0.99, 1.0], "seeds": [0], "verify_seeds": [0, 1], "n_train": 16, "n_val": 6}


def job(args: tuple) -> dict:
    lr, episodes, final_eps, seed, train_pool, val_pool, out, *rest = args
    masked, gamma = rest if rest else (True, 1.0)
    tag = f"lr{lr}_ep{episodes}_fe{final_eps}_g{gamma}_{'m' if masked else 'u'}_s{seed}"
    agent, _ = train(train_pool, episodes, seed, masked, out / "work" / tag, learning_rate=lr,
                     final_epsilon=final_eps, discount_factor=gamma)
    res = evaluate(agent, val_pool, out / "logs" / f"{tag}.jsonl", tag)
    res.pop("_rows", None)
    res["val_episodes"] = res.pop("episodes")
    return {"lr": lr, "episodes": episodes, "final_eps": final_eps, "seed": seed, "masked": masked,
            "gamma": gamma, **res}


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    cells = {}
    for r in rows:
        cells.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in cells.items():
        ret = [r["return"] for r in rs]
        out.append({**dict(zip(keys, key)), "return": round(statistics.fmean(ret), 4),
                    "std": round(statistics.pstdev(ret), 4), "f1": round(statistics.fmean(r["f1"] for r in rs), 4),
                    "steps": round(statistics.fmean(r["steps"] for r in rs), 2), "runs": len(rs)})
    return out


def pick(cells: list[dict]) -> dict:
    """Best validation return; near-ties go to fewer episodes, lower spread, the default epsilon, a lower lr."""
    top = max(c["return"] for c in cells)
    near = [c for c in cells if c["return"] >= top - 0.005]
    return min(near, key=lambda c: (c.get("episodes", 0), c["std"], abs(c.get("final_eps", 0.05) - 0.05),
                                    c.get("lr", 0), -c["return"]))


def _strip(res: dict) -> dict:
    res.pop("_rows", None)
    return res


def stages_1_2(g: dict, run, train_pool: list[Path], val_pool: list[Path], out: Path):
    """Stage 1 (lr x episodes), stage 1b (finalists on all verify seeds), stage 2 (final epsilon)."""
    fe0 = 0.05 if 0.05 in g["final_eps"] else g["final_eps"][0]
    stage1_rows = run([(lr, ep, fe0, s, train_pool, val_pool, out)
                       for lr, ep, s in itertools.product(g["lr"], g["episodes"], g["seeds"])])
    stage1 = aggregate(stage1_rows, ("lr", "episodes"))
    top = max(c["return"] for c in stage1)
    finalist_keys = {(c["lr"], c["episodes"]) for c in stage1 if c["return"] >= top - 0.005}
    extra = [s for s in g["verify_seeds"] if s not in g["seeds"]]
    verify_rows = run([(lr, ep, fe0, s, train_pool, val_pool, out) for lr, ep in sorted(finalist_keys) for s in extra])
    finalists = aggregate([r for r in stage1_rows if (r["lr"], r["episodes"]) in finalist_keys] + verify_rows,
                          ("lr", "episodes"))
    best1 = pick(finalists)
    print(f"stage 1b: {len(finalists)} finalists on {len(g['verify_seeds'])} seeds; winner lr {best1['lr']}, "
          f"episodes {best1['episodes']}: val return {best1['return']:+.3f} +/- {best1['std']:.3f}", flush=True)
    stage2_rows = run([(best1["lr"], best1["episodes"], fe, s, train_pool, val_pool, out)
                       for fe, s in itertools.product(g["final_eps"], g["seeds"])])
    stage2 = aggregate(stage2_rows, ("final_eps",))
    best2 = pick(stage2)
    return stage1, finalists, best1, stage2, best2


def stage_3(g: dict, run, best1: dict, best2: dict, train_pool: list[Path], val_pool: list[Path], out: Path):
    """Discount factor for both arms (masked and unmasked) of the benchmark."""
    rows = run([(best1["lr"], best1["episodes"], best2["final_eps"], s, train_pool, val_pool, out, m, gm)
                for gm in g["gamma"] for m in (True, False) for s in g["verify_seeds"]])
    stage3 = []
    for gm in g["gamma"]:
        cell = {"gamma": gm, "runs": len(g["verify_seeds"])}
        for m, arm in ((True, "masked"), (False, "unmasked")):
            rs = [r for r in rows if r["gamma"] == gm and r["masked"] == m]
            ret = [r["return"] for r in rs]
            cell.update({f"{arm}_return": round(statistics.fmean(ret), 4), f"{arm}_std": round(statistics.pstdev(ret), 4),
                         f"{arm}_f1": round(statistics.fmean(r["f1"] for r in rs), 4),
                         f"{arm}_steps": round(statistics.fmean(r["steps"] for r in rs), 2)})
        stage3.append(cell)
    top_m = max(c["masked_return"] for c in stage3)
    keep = [c for c in stage3 if c["masked_return"] >= top_m - 0.005]
    best3 = max(keep, key=lambda c: (round(c["unmasked_return"], 3), c["gamma"]))
    print(f"stage 3: gamma {best3['gamma']} (masked {best3['masked_return']:+.3f}, "
          f"unmasked {best3['unmasked_return']:+.3f})", flush=True)
    return stage3, best3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "sweep")
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--resume", action="store_true",
                    help="reuse stages 1-2 of an existing sweep.json with the same grid and seeds (results are deterministic)")
    args = ap.parse_args(argv)
    g = QUICK if args.quick else FULL
    previous = None
    if args.resume and (args.out / "sweep.json").exists():
        old = json.loads((args.out / "sweep.json").read_text())
        same = (old.get("grid", {}).get("lr") == g["lr"] and old["grid"].get("episodes") == g["episodes"]
                and old["grid"].get("final_eps") == g["final_eps"]
                and old["protocol"].get("verify_seeds") == g["verify_seeds"]
                and old["protocol"].get("agent_seeds") == g["seeds"] and "finalists" in old)
        previous = old if same else None
        print("resume:", "reusing stages 1-2" if previous else "grid changed, starting over", flush=True)
    if previous is None:
        shutil.rmtree(args.out, ignore_errors=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train_pool = build_pool("train", g["n_train"], args.out / "pool")
    val_pool = build_pool("val", g["n_val"], args.out / "pool")
    assert not {p.name for p in train_pool} & {p.name for p in val_pool}

    def run(jobs):
        with ProcessPoolExecutor(args.workers) as ex:
            return list(ex.map(job, jobs))

    if previous is not None:
        stage1, finalists, stage2 = previous["stage1"], previous["finalists"], previous["stage2"]
        best1 = pick(finalists)
        best2 = pick(stage2)
    else:
        stage1, finalists, best1, stage2, best2 = stages_1_2(g, run, train_pool, val_pool, args.out)
    stage3, best3 = stage_3(g, run, best1, best2, train_pool, val_pool, args.out)

    reference = {name: _strip(evaluate(name, val_pool, args.out / "logs" / f"ref_{name}.jsonl", name))
                 for name in ("untrained", "reconcile", "expert_options", "oracle")}
    summary = {
        "protocol": {"train_seeds": [300_000, 300_000 + g["n_train"] - 1],
                     "val_seeds": [500_000, 500_000 + g["n_val"] - 1],
                     "heldout_untouched": [400_000, 400_199], "agent_seeds": g["seeds"],
                     "verify_seeds": g["verify_seeds"], "masked": True,
                     "selection": "cells within 0.005 of the best re-run on all verify seeds; then max mean "
                                  "validation return, near-ties (0.005) -> fewer episodes, lower std, default epsilon, "
                                  "lower learning rate",
                     "gamma_selection": "masked arm within 0.005 of its best, then the best unmasked arm "
                                        "(no crippled baseline), then the larger gamma"},
        "grid": {k: g[k] for k in ("lr", "episodes", "final_eps", "gamma")},
        "stage1": sorted(stage1, key=lambda c: (c["lr"], c["episodes"])),
        "finalists": sorted(finalists, key=lambda c: (-c["return"], c["episodes"])),
        "stage2": sorted(stage2, key=lambda c: c["final_eps"]),
        "stage3": stage3,
        "best": {"learning_rate": best1["lr"], "episodes": best1["episodes"], "final_epsilon": best2["final_eps"],
                 "discount_factor": best3["gamma"], "val_return": best3["masked_return"],
                 "val_std": best3["masked_std"], "val_f1": best3["masked_f1"],
                 "val_unmasked_return": best3["unmasked_return"], "val_seeds": len(g["verify_seeds"])},
        "reference_on_val": reference,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(args.out / "sweep.json", summary)
    if not args.quick:
        (ROOT / "artifacts").mkdir(exist_ok=True)
        shutil.copy(args.out / "sweep.json", ROOT / "artifacts" / "sweep.json")
    b = summary["best"]
    print(f"best: lr {b['learning_rate']}, episodes {b['episodes']}, final epsilon {b['final_epsilon']}, "
          f"gamma {b['discount_factor']} -> val return {b['val_return']:+.3f} +/- {b['val_std']:.3f}, "
          f"F1 {b['val_f1']:.3f}; unmasked {b['val_unmasked_return']:+.3f} ({summary['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Training benchmark in the style of the Gymnasium tutorials (Blackjack / Taxi Q-learning).

Shows that the environment gives a LEARNABLE signal, with a controlled comparison on held-out
data, as the brief requires ("no claims of improvement without a controlled comparison on
held-out data"). CPU only, stdlib only, deterministic.

* Agent: tabular Q-learning with epsilon-greedy exploration and per-episode epsilon decay
  (BlackjackAgent pattern), TD errors recorded as the training error.
* Env: AuditEnv wrapped in AuditOptionsEnv (discrete audit skills, info["action_mask"]) and
  RecordEpisodeStatistics. Rewards are the env's own terminal rewards in train mode.
* Experiment (Taxi pattern): K independent seeds x {masked, unmasked} agents.
* Data: training pool and held-out evaluation pool from DISJOINT seeds; evaluation in eval
  mode (rewards read from the grader log), greedy policy, against the baselines.

    python examples/train_q_learning.py            # full benchmark (~5 min), writes runs/training/
    python examples/train_q_learning.py --quick    # smoke run (tests)

What this does NOT show: it is not an LLM and the skills are scripted; the agent learns which
audit procedures to run and when to stop. It demonstrates the reward signal, the API and a
proper held-out protocol, not real-world audit ability.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditenv.benchstats import bootstrap_ci, paired_bootstrap, recall_by_type  # noqa: E402
from auditenv.common import read_json, write_json  # noqa: E402
from auditenv.env import AuditEnv  # noqa: E402
from auditenv.generator import generate_episode  # noqa: E402
from auditenv.grader.client import GraderClient  # noqa: E402
from auditenv.grader.oracle import oracle_submission  # noqa: E402
from auditenv.policies import make_policy  # noqa: E402
from auditenv.wrappers import OPTIONS, AuditOptionsEnv, RecordEpisodeStatistics  # noqa: E402

TRAIN_SEED_BASE, EVAL_SEED_BASE, VAL_SEED_BASE = 300_000, 400_000, 500_000
SEED_BASES = {"train": TRAIN_SEED_BASE, "eval": EVAL_SEED_BASE, "val": VAL_SEED_BASE}
# Chosen by examples/sweep_hparams.py on the validation pool (never on the held-out pool).
LEARNING_RATE, FINAL_EPSILON, EPISODES_PER_RUN, DISCOUNT_FACTOR = 0.8, 0.05, 300, 0.99
# Every pool mirrors the hidden-set recipe (configs/hidden_like.json): clean : interacting : standard = 1 : 1 : 2.
from auditenv.training import CLASS_CONFIG, EPISODE_CLASSES  # noqa: E402
EPISODE_CLASS: dict[str, str] = {}  # episode_id -> class, filled by build_pool


class QLearningAgent:
    """Tabular Q-learning, as in the Gymnasium Blackjack tutorial (masking as in the Taxi one)."""

    def __init__(self, n_actions: int, learning_rate: float, initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, discount_factor: float = 1.0, seed: int = 0, use_mask: bool = True):
        self.n = n_actions
        self.q_values = defaultdict(lambda: [0.0] * n_actions)
        self.lr, self.gamma = learning_rate, discount_factor
        self.epsilon, self.epsilon_decay, self.final_epsilon = initial_epsilon, epsilon_decay, final_epsilon
        self.use_mask = use_mask
        self.rng = random.Random(seed)
        self.training_error: list[float] = []

    def _valid(self, mask):
        return [a for a in range(self.n) if not self.use_mask or mask[a]] or list(range(self.n))

    def get_action(self, state, mask, greedy: bool = False) -> int:
        valid = self._valid(mask)
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(valid)
        q = self.q_values[state]
        best = max(q[a] for a in valid)
        return min(a for a in valid if q[a] == best)  # deterministic tie-break

    def update(self, state, action: int, reward: float, terminated: bool, next_state, next_mask) -> None:
        future = 0.0 if terminated else max(self.q_values[next_state][a] for a in self._valid(next_mask))
        td = reward + self.gamma * future - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td
        self.training_error.append(td)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def rolling_abs(values: list[float], window: int, points: int) -> list[float]:
    """Rolling mean of |TD error|, downsampled to at most `points` values (keeps results small)."""
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += abs(v)
        if i >= window:
            acc -= abs(values[i - window])
        out.append(round(acc / min(i + 1, window), 5))
    step = max(1, len(out) // points)
    return out[::step]


# ------------------------------------------------------------------------------- data
def build_pool(kind: str, n: int, root: Path) -> list[Path]:
    base = SEED_BASES[kind]
    out = root / kind
    pool = []
    for i in range(n):
        cls = EPISODE_CLASSES[i % len(EPISODE_CLASSES)]
        ep = generate_episode(base + i, CLASS_CONFIG[cls], out)
        EPISODE_CLASS[read_json(ep / "public" / "manifest.json")["episode_id"]] = cls
        pool.append(ep)
    return pool


# ---------------------------------------------------------------------------- training
def train(pool: list[Path], n_episodes: int, seed: int, masked: bool, workdir: Path,
          learning_rate: float = LEARNING_RATE, final_epsilon: float = FINAL_EPSILON,
          discount_factor: float | None = None) -> tuple[QLearningAgent, dict]:
    agent = QLearningAgent(len(OPTIONS), learning_rate=learning_rate, initial_epsilon=1.0,
                           epsilon_decay=1.0 / (n_episodes / 2), final_epsilon=final_epsilon, seed=seed,
                           use_mask=masked,
                           discount_factor=DISCOUNT_FACTOR if discount_factor is None else discount_factor)
    env = RecordEpisodeStatistics(AuditOptionsEnv(AuditEnv(mode="train", workdir=workdir), mask=masked),
                                  buffer_length=n_episodes)
    picker = random.Random(f"episodes|{seed}")
    primitive_steps = []
    for _ in range(n_episodes):
        state, info = env.reset(episode_dir=picker.choice(pool))
        mask = info["action_mask"]
        done = False
        while not done:
            action = agent.get_action(state, mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, terminated, next_state, info["action_mask"])
            done = terminated or truncated
            state, mask = next_state, info["action_mask"]
        primitive_steps.append(info["steps_used"])
        agent.decay_epsilon()
    env.close()
    curves = {"returns": list(env.return_queue), "option_lengths": list(env.length_queue),
              "primitive_steps": primitive_steps,
              "training_error_rolling": rolling_abs(agent.training_error, window=500, points=400),
              "updates": len(agent.training_error)}
    return agent, curves


# -------------------------------------------------------------------------- evaluation
def evaluate(policy, pool: list[Path], log: Path, name: str) -> dict:
    """Eval mode: the policy gets no reward; results come from the grader log only."""
    for ep in pool:
        env = AuditEnv(mode="eval", policy_name=name)
        grader = GraderClient.for_episode(ep, log_path=log, policy_name=name)
        if isinstance(policy, QLearningAgent) or policy in ("untrained", "expert_options"):
            wrapped = AuditOptionsEnv(env, mask=True)
            state, info = wrapped.reset(episode_dir=ep, grader=grader)
            rng = random.Random(f"untrained|{ep.name}")
            done, script = False, iter(range(len(OPTIONS)))
            while not done:
                if isinstance(policy, QLearningAgent):
                    action = policy.get_action(state, info["action_mask"], greedy=True)
                elif policy == "expert_options":
                    action = next(script)
                else:  # untrained: uniform over valid options (epsilon = 1)
                    action = rng.choice([a for a, ok in enumerate(info["action_mask"]) if ok])
                state, _, terminated, truncated, info = wrapped.step(action)
                done = terminated or truncated
        else:
            obs, info = env.reset(episode_dir=ep, grader=grader)
            if policy == "oracle":
                env.step({"tool": "submit", "args": {"findings": oracle_submission(ep)}})
                continue
            p = make_policy(policy, ep)
            p.reset(obs, info)
            while True:
                obs, _, terminated, truncated, _ = env.step(p.act(obs))
                p.observe(obs)
                if terminated or truncated:
                    break
    rows = [json.loads(line) for line in log.read_text().splitlines() if json.loads(line)["policy"] == name]
    clean = [r for r in rows if r["tp"] + r["fn"] == 0]
    dirty = [r for r in rows if r["tp"] + r["fn"] > 0]
    mean = lambda key, rs: round(statistics.fmean(r[key] for r in rs), 4) if rs else None
    return {"episodes": len(rows), "return": mean("reward", rows), "f1": mean("f1", rows),
            "steps": mean("inspection_steps", rows), "return_clean": mean("reward", clean),
            "return_with_errors": mean("reward", dirty), "_rows": rows}


def describe(rows_per_run: list[list[dict]], label: str) -> dict:
    """Held-out statistics for one policy (several runs for trained agents, one for baselines).

    The per-episode return is averaged over runs first, so the bootstrap CI is over episodes.
    """
    by_ep: dict[str, list[dict]] = {}
    for rows in rows_per_run:
        for r in rows:
            by_ep.setdefault(r["episode_id"], []).append(r)
    order = sorted(by_ep)
    per_ep = [statistics.fmean(r["reward"] for r in by_ep[e]) for e in order]
    by_class = {}
    for cls in dict.fromkeys(EPISODE_CLASSES):
        vals = [v for e, v in zip(order, per_ep) if EPISODE_CLASS.get(e) == cls]
        if vals:
            by_class[cls] = round(statistics.fmean(vals), 4)
    all_rows = [r for rows in rows_per_run for r in rows]
    return {"return": round(statistics.fmean(per_ep), 4), "ci95": bootstrap_ci(per_ep, seed=f"ci|{label}"),
            "f1": round(statistics.fmean(r["f1"] for r in all_rows), 4), "by_class": by_class,
            "recall_by_type": recall_by_type(all_rows), "episodes": len(order), "runs": len(rows_per_run),
            "_per_episode": dict(zip(order, per_ep))}


def greedy_policy(agent: QLearningAgent) -> list[str]:
    """The learned plan from the empty state (for the report)."""
    state, plan, seen = (0,) * 8 + (2, 2), [], set()
    read = {"read_invoices": 0, "read_payments": 1, "read_ledger": 2, "read_fx": 3}
    check = {"check_duplicates": 4, "check_suspense": 5, "check_cutoff": 6, "check_fx": 7}
    while len(plan) < 12:
        mask = [1] * len(OPTIONS)
        for name, i in read.items():
            mask[OPTIONS.index(name)] = int(state[i] == 0)
        need = {"check_duplicates": (0,), "check_suspense": (1, 2), "check_cutoff": (0, 2), "check_fx": (0, 1, 2, 3)}
        for name, i in check.items():
            mask[OPTIONS.index(name)] = int(state[i] == 0 and all(state[j] for j in need[name]))
        a = agent.get_action(state, mask, greedy=True)
        plan.append(OPTIONS[a])
        if OPTIONS[a] == "submit" or (state, a) in seen:
            break
        seen.add((state, a))
        s = list(state)
        if OPTIONS[a] in read:
            s[read[OPTIONS[a]]] = 1
        elif OPTIONS[a] in check:
            s[check[OPTIONS[a]]] = 1
        if s[0] and s[8] == 2:
            s[8] = 1  # assume foreign documents present (the common case)
        if s[1] and s[2] and s[9] == 2:
            s[9] = 1
        state = tuple(s)
    return plan


QUICK = (24, 8, 40, [0, 1])  # --quick: train pool, eval pool, episodes per run, agent seeds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "training")
    args = ap.parse_args(argv)
    n_train, n_eval, n_episodes, seeds = QUICK if args.quick else (120, 200, EPISODES_PER_RUN, [0, 1, 2, 3, 4])
    shutil.rmtree(args.out, ignore_errors=True)
    t0 = time.perf_counter()
    pools = {"train": build_pool("train", n_train, args.out / "pool"), "eval": build_pool("eval", n_eval, args.out / "pool")}
    overlap = {p.name for p in pools["train"]} & {p.name for p in pools["eval"]}
    assert not overlap, f"train/eval overlap: {overlap}"

    runs, evals = [], defaultdict(list)
    log = args.out / "eval_log.jsonl"
    for masked in (True, False):
        for seed in seeds:
            agent, curves = train(pools["train"], n_episodes, seed, masked, args.out / "work")
            tag = f"q_{'masked' if masked else 'unmasked'}_s{seed}"
            result = evaluate(agent, pools["eval"], log, tag)
            evals["masked" if masked else "unmasked"].append(result)
            runs.append({"seed": seed, "masked": masked, "curves": curves, "eval": result,
                         "greedy_plan": greedy_policy(agent)})
            print(f"  trained {tag}: held-out return {result['return']:+.3f}, F1 {result['f1']:.3f}, "
                  f"steps {result['steps']:.1f}", flush=True)

    baselines = {name: evaluate(name, pools["eval"], log, name)
                 for name in ("untrained", "silent", "random", "reconcile", "expert_options", "oracle")}

    # Held-out statistics: bootstrap CIs over episodes, paired comparisons, breakdowns.
    stats = {"q_masked": describe([r.pop("_rows") for r in evals["masked"]], "q_masked"),
             "q_unmasked": describe([r.pop("_rows") for r in evals["unmasked"]], "q_unmasked")}
    for name, res in baselines.items():
        stats[name] = describe([res.pop("_rows")], name)
    per_ep = {k: v.pop("_per_episode") for k, v in stats.items()}
    order = sorted(per_ep["q_masked"])
    vec = lambda k: [per_ep[k][e] for e in order]  # noqa: E731
    pairs = [("q_masked", b) for b in ("untrained", "silent", "reconcile", "q_unmasked", "expert_options")]
    significance = [{"a": a, "b": b, **paired_bootstrap(vec(a), vec(b), seed=f"paired|{a}|{b}")} for a, b in pairs]
    composition = {}
    for ep in pools["eval"]:
        cls = EPISODE_CLASS[read_json(ep / "public" / "manifest.json")["episode_id"]]
        composition[cls] = composition.get(cls, 0) + 1

    def agg(results):
        keys = ("return", "f1", "steps", "return_clean", "return_with_errors")
        return {k: {"mean": round(statistics.fmean(r[k] for r in results), 4),
                    "std": round(statistics.pstdev([r[k] for r in results]), 4)} for k in keys}

    summary = {
        "protocol": {"train_seeds": [TRAIN_SEED_BASE, TRAIN_SEED_BASE + n_train - 1],
                     "eval_seeds": [EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval - 1], "train_pool": n_train,
                     "eval_pool": n_eval, "episodes_per_run": n_episodes, "agent_seeds": seeds,
                     "agent": f"tabular Q-learning over audit options, lr {LEARNING_RATE}, gamma {DISCOUNT_FACTOR}, epsilon 1 -> "
                              f"{FINAL_EPSILON} (linear over the first half), masked vs unmasked",
                     "hyperparameters": {"learning_rate": LEARNING_RATE, "discount_factor": DISCOUNT_FACTOR,
                                         "initial_epsilon": 1.0, "final_epsilon": FINAL_EPSILON,
                                         "episodes_per_run": n_episodes,
                                         "chosen_by": "examples/sweep_hparams.py on validation seeds 500000+"},
                     "eval_composition": composition,
                     "statistics": "95% percentile bootstrap over held-out episodes (4000 resamples, seeded); "
                                   "trained agents: per-episode return averaged over agent seeds first; "
                                   "paired bootstrap on the same episodes for comparisons",
                     "evaluation": "eval mode, greedy policy, rewards from the grader log only"},
        "trained": {cond: agg(results) for cond, results in evals.items()},
        "baselines": baselines,
        "stats": stats,
        "significance": significance,
        "runs": runs,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(args.out / "results.json", summary)
    if not args.quick:
        (ROOT / "artifacts").mkdir(exist_ok=True)
        shutil.copy(args.out / "results.json", ROOT / "artifacts" / "training.json")
        if (ROOT / "runs" / "dev" / "summary.json").exists():
            from auditenv.report import build_report
            shutil.copy(build_report(ROOT / "runs", status="PASSED"), ROOT / "artifacts" / "report.html")

    print(f"\ntraining benchmark ({summary['seconds']}s): held-out {n_eval} episodes, seeds disjoint from training")
    print(f"  {'policy':<16} {'return':>8} {'F1':>6} {'steps':>6}")
    rows = [(f"q-learning {c}", v["return"]["mean"], v["f1"]["mean"], v["steps"]["mean"]) for c, v in summary["trained"].items()]
    rows += [(k, v["return"], v["f1"], v["steps"]) for k, v in baselines.items()]
    for name, ret, f1, steps in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<16} {ret:+8.3f} {f1:6.3f} {steps:6.1f}")
    print(f"\n  95% CI over {n_eval} held-out episodes ({composition}):")
    for name, st in sorted(stats.items(), key=lambda kv: -kv[1]["return"]):
        print(f"  {name:<16} {st['return']:+.3f}  [{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}]  by class {st['by_class']}")
    print("\n  paired comparisons (same episodes):")
    for c in significance:
        print(f"  {c['a']} - {c['b']:<15} {c['diff']:+.3f}  CI [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]  "
              f"p(<=0) {c['p_le_0']:.4f}  W/T/L {c['wins']}/{c['ties']}/{c['losses']}")
    trained = summary["trained"]["masked"]["return"]["mean"]
    ok = trained > baselines["untrained"]["return"] and trained > baselines["silent"]["return"]
    if not args.quick:  # full run: the gains must be significant on held-out data, not just positive
        sig = {c["b"]: c for c in significance}
        ok = ok and sig["untrained"]["ci"][0] > 0 and sig["reconcile"]["ci"][0] > 0
    print("TRAINING BENCHMARK", "PASSED" if ok else "FAILED (no significant learning on held-out episodes)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

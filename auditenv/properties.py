"""Measured reward properties (task section 05) on the open episodes, through the real grader.

Each property is computed, not asserted from the formula: the same scoring code the grader runs
(grader/reward.py, with the episode's published tolerance and its set of existing evidence ids,
grader/episode.py) is applied to concrete submissions built from the open episodes. make check
fails if any property does not hold; the table goes to the report and artifacts/reward_properties.json.
"""

from __future__ import annotations

import copy
import random
import statistics
from pathlib import Path

from auditenv.common import read_json
from auditenv.generator.validate import load_public
from auditenv.grader.oracle import oracle_submission
from auditenv.grader.episode import load_grading_inputs
from auditenv.grader.reward import score

_INPUTS: dict[Path, tuple] = {}


def _truth(ep: Path) -> dict:
    return read_json(ep / "private" / "truth.json")


def _score(ep: Path, submission: dict, steps: int) -> dict:
    """Exactly what the grader process computes for this episode."""
    if ep not in _INPUTS:
        _INPUTS[ep] = load_grading_inputs(ep)
    truth, tolerance, universe = _INPUTS[ep]
    return score(submission, truth, steps, tolerance, universe)


def _accusation(ep: Path, rng: random.Random, n: int) -> list[dict]:
    """n plausible but unfounded findings built from real public invoices."""
    data = load_public(ep)
    invoices = rng.sample(data["invoices"], min(n, len(data["invoices"])))
    return [{"type": "duplicate_invoice", "affected_ids": [inv["invoice_id"]],
             "claimed_amount_minor": inv["total_minor"], "currency": inv["currency"],
             "evidence_ids": [inv["doc_id"]], "explanation": "looks suspicious"} for inv in invoices]


def measure(episodes: list[Path], baselines: dict) -> list[dict]:
    rng = random.Random("reward-properties")
    clean = [ep for ep in episodes if not _truth(ep)["issues"]]
    dirty = [ep for ep in episodes if _truth(ep)["issues"]]
    mean = statistics.fmean
    rows = []

    def add(prop: str, setup: str, values: dict, holds: bool):
        rows.append({"property": prop, "setup": setup, "values": {k: round(v, 4) for k, v in values.items()},
                     "holds": bool(holds)})

    def sub(ep, findings):
        return {"episode_id": _truth(ep)["episode_id"], "findings": findings}

    oracle = [_score(ep, oracle_submission(ep), 0) for ep in episodes]
    add("Oracle reaches the ceiling on every open episode", f"{len(episodes)} episodes, 0 steps",
        {"min_reward": min(g["reward"] for g in oracle), "min_f1": min(g["f1"] for g in oracle)},
        all(g["reward"] == 1.0 and g["f1"] == 1.0 for g in oracle))

    add("Flag-everything and random exploration score strongly negative", "baselines on train + dev",
        {"flag_all": baselines["flag_all"], "random": baselines["random"]},
        baselines["flag_all"] <= -0.9 and baselines["random"] <= -0.5)

    silence_clean = [_score(ep, sub(ep, []), 0)["reward"] for ep in clean]
    one_false = [_score(ep, sub(ep, _accusation(ep, rng, 1)), 0)["reward"] for ep in clean]
    five_false = [_score(ep, sub(ep, _accusation(ep, rng, 5)), 0)["reward"] for ep in clean]
    add("Silence beats false accusations on clean books", f"{len(clean)} clean episodes",
        {"silence": mean(silence_clean), "one_false_finding": mean(one_false), "five_false_findings": mean(five_false)},
        min(silence_clean) > max(one_false + five_false))

    full = [_score(ep, oracle_submission(ep), 20)["reward"] for ep in dirty]
    silence_dirty = [_score(ep, sub(ep, []), 0)["reward"] for ep in dirty]
    add("A full answer beats silence on books with errors", f"{len(dirty)} dirty episodes, answer after 20 steps",
        {"full_answer": mean(full), "silence": mean(silence_dirty)}, min(full) > max(silence_dirty))

    dup = []
    for ep in dirty:
        f = oracle_submission(ep)["findings"]
        dup.append(_score(ep, sub(ep, f + copy.deepcopy(f)), 0)["reward"])
    add("Duplicated findings give no benefit", "every oracle finding submitted twice",
        {"answer": 1.0, "answer_twice": mean(dup)}, max(dup) < 1.0)

    guessed = [_score(ep, sub(ep, oracle_submission(ep)["findings"] + _accusation(ep, rng, 3)), 0)["reward"]
               for ep in dirty]
    add("Guessed IDs are false positives", "oracle answer + 3 findings on real but innocent invoices",
        {"answer": 1.0, "answer_plus_guesses": mean(guessed)}, max(guessed) < 1.0)

    unsupported = []
    for ep in dirty:
        f = copy.deepcopy(oracle_submission(ep)["findings"])
        for x in f:
            x["evidence_ids"] = []
        unsupported.append(_score(ep, sub(ep, f), 0)["reward"])
    add("Findings without supporting evidence earn nothing", "oracle answer with evidence_ids removed",
        {"answer_without_evidence": mean(unsupported)}, max(unsupported) == -1.0)

    steps = {s: mean(_score(ep, oracle_submission(ep), s)["reward"] for ep in dirty) for s in (0, 25, 50, 100)}
    add("Saving steps separates equal answers only slightly", "same correct answer after 0 / 25 / 50 / 100 steps",
        {f"steps_{s}": v for s, v in steps.items()},
        steps[0] > steps[25] > steps[50] > steps[100] and steps[0] - steps[100] <= 0.2 + 1e-9)

    ghost = []
    for ep in dirty:
        f = copy.deepcopy(oracle_submission(ep)["findings"])
        for x in f:
            x["evidence_ids"] = x["evidence_ids"] + ["NOT-AN-ID"]
        ghost.append(_score(ep, sub(ep, f), 0)["reward"])
    add("Evidence ids that do not exist make a finding a false positive",
        "oracle answer plus one made-up evidence id per finding",
        {"answer": 1.0, "answer_with_made_up_evidence": mean(ghost)}, max(ghost) == -1.0)

    broken = []
    for ep in dirty:
        f = copy.deepcopy(oracle_submission(ep)["findings"])
        f[0]["claimed_amount_minor"] = float(f[0]["claimed_amount_minor"])
        broken.append(_score(ep, sub(ep, f), 0)["reward"])
    add("Invalid structure ends at -1", "a correct answer with one float amount", {"reward": mean(broken)},
        all(r == -1.0 for r in broken))
    return rows


def print_table(rows: list[dict]) -> None:
    print("\nreward properties (measured on the open episodes through the grader)")
    for r in rows:
        vals = ", ".join(f"{k} {v:+.3f}" for k, v in r["values"].items())
        print(f"  {'OK ' if r['holds'] else 'NO '} {r['property']:<62} {vals}")

"""The tie-break rule against a brute force, on EVERY bipartite graph up to 4 issues x 4 findings
(74,963 graphs), then the full matcher against the same brute force on random real submissions.

The rule (task text plus the jury's definitions): the number of true positives is the size of a
maximum one-to-one matching; which pairs are reported is fixed by taking issues in issue_id order
and giving each the smallest finding index that still allows a maximum matching. The brute force
below shares no code with auditenv/grader/matcher.py: it lists every partial one-to-one assignment
in lexicographic order and keeps the first one of the largest size."""

from __future__ import annotations

import itertools
import random

import pytest

from auditenv.grader.matcher import finding_matches, lexicographic_maximum, match

NONE = 10**9  # an unmatched issue sorts after every finding index


def brute_force(adj: list[set[int]]) -> dict[int, int]:
    best: tuple[int, tuple] = (-1, ())
    for choice in itertools.product(*[sorted(a) + [NONE] for a in adj]):
        used = [f for f in choice if f != NONE]
        if len(used) != len(set(used)):
            continue
        if len(used) > best[0]:  # product() walks in lexicographic order: the first of a size wins
            best = (len(used), choice)
    return {i: f for i, f in enumerate(best[1]) if f != NONE}


def all_graphs(max_issues: int, max_findings: int):
    for n_i in range(max_issues + 1):
        for n_f in range(max_findings + 1):
            cells = [(i, f) for i in range(n_i) for f in range(n_f)]
            for mask in range(1 << len(cells)):
                adj = [set() for _ in range(n_i)]
                for bit, (i, f) in enumerate(cells):
                    if mask >> bit & 1:
                        adj[i].add(f)
                yield adj


def test_every_graph_up_to_4_by_4_gets_the_brute_force_answer():
    count = 0
    for adj in all_graphs(4, 4):
        expected = brute_force(adj)
        got = lexicographic_maximum([sorted(a) for a in adj])
        assert got == expected, adj
        count += 1
    assert count == 74963


def test_larger_random_graphs_agree_too():
    rng = random.Random(20261121)
    for _ in range(3000):
        n_i, n_f = rng.randint(0, 6), rng.randint(0, 6)
        p = rng.random()
        adj = [{f for f in range(n_f) if rng.random() < p} for _ in range(n_i)]
        assert lexicographic_maximum([sorted(a) for a in adj]) == brute_force(adj), adj


# --- the full matcher on real-shaped findings -------------------------------------------------

TYPES = ["duplicate_invoice", "fx_conversion", "partial_payment"]
IDS = ["INV-A", "INV-B", "PAY-C", "PAY-D"]
EVIDENCE = ["L-1", "L-2", "DOC-3", "2026-09-29:EUR:KZT"]
UNIVERSE = frozenset(EVIDENCE)  # "L-404" below is cited sometimes and exists nowhere


def _issue(rng, n):
    return {"issue_id": f"ISS-{n:02d}", "type": rng.choice(TYPES),
            "affected_ids": sorted(rng.sample(IDS, rng.randint(1, 2))),
            "evidence_sets": [sorted(rng.sample(EVIDENCE, rng.randint(1, 2))) for _ in range(rng.randint(1, 2))],
            "expected_amount_minor": rng.choice([1000, 1001, 1003, 5000]), "currency": "KZT"}


def _finding(rng, issues):
    if issues and rng.random() < 0.7:  # near a real issue, sometimes slightly off
        t = rng.choice(issues)
        return {"type": t["type"] if rng.random() < 0.9 else rng.choice(TYPES),
                "affected_ids": list(t["affected_ids"]) if rng.random() < 0.85 else [rng.choice(IDS)],
                "evidence_ids": rng.choice(t["evidence_sets"]) + ([rng.choice(EVIDENCE + ["L-404"])] if rng.random() < 0.3 else []),
                "claimed_amount_minor": t["expected_amount_minor"] + rng.choice([0, 0, 1, 2, -3, 400]),
                "currency": "KZT" if rng.random() < 0.95 else "EUR"}
    return {"type": rng.choice(TYPES), "affected_ids": [rng.choice(IDS)], "evidence_ids": [rng.choice(EVIDENCE)],
            "claimed_amount_minor": rng.choice([1000, 5000]), "currency": "KZT"}


@pytest.mark.parametrize("seed", range(4))
def test_full_matcher_equals_brute_force_on_random_submissions(seed):
    rng = random.Random(f"matcher-exhaustive|{seed}")
    for _ in range(400):
        issues = [_issue(rng, n) for n in rng.sample(range(40), rng.randint(0, 4))]
        findings = [_finding(rng, issues) for _ in range(rng.randint(0, 5))]
        ordered = sorted(issues, key=lambda t: t["issue_id"])
        adj = [{fi for fi, f in enumerate(findings) if finding_matches(f, t, 2, UNIVERSE)} for t in ordered]
        expected = sorted((fi, ordered[ti]["issue_id"]) for ti, fi in brute_force(adj).items())
        shuffled = issues[:]
        rng.shuffle(shuffled)  # truth order must not matter
        assert match(findings, shuffled, 2, UNIVERSE) == expected

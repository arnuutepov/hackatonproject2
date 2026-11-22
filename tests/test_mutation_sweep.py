"""The mechanical mutation sweep (auditenv/mutation_sweep.py, `make mutation-sweep`, about 10 min)
is too slow for `make check`. These fast tests keep it honest: the generator is deterministic and
every mutant is a real one-operator change, and the recorded artifacts/mutation_sweep.json was made
from today's grader source (a changed matcher.py or reward.py makes the record stale and fails here)."""

from __future__ import annotations

import ast
import json

from auditenv.mutation_sweep import ROOT, TARGETS, mutants, source_digest


def test_generator_is_deterministic_and_every_mutant_is_a_real_change():
    first, second = mutants(), mutants()
    assert [(m["id"], m["file"], m["what"]) for m in first] == [(m["id"], m["file"], m["what"]) for m in second]
    assert len(first) >= 150
    assert {m["file"] for m in first} == set(TARGETS)
    baselines = {rel: ast.unparse(ast.parse((ROOT / rel).read_text(encoding="utf-8"))) for rel in TARGETS}
    for m in first:
        ast.parse(m["source"])
        assert m["source"] != baselines[m["file"]], m["id"]


def test_recorded_sweep_matches_todays_grader_source():
    record = json.loads((ROOT / "artifacts" / "mutation_sweep.json").read_text(encoding="utf-8"))
    assert record["source_sha256"] == source_digest(), "grader changed: run `make mutation-sweep` again"
    rows = record["mutants"]
    assert [(r["id"], r["what"]) for r in rows] == [(m["id"], m["what"]) for m in mutants()]
    assert record["total"] == len(rows) == record["killed"] + record["survived"]
    assert record["survived"] == len(record["survivors"]) == sum(r["status"] != "killed" for r in rows)
    assert all(r["status"] in ("killed", "SURVIVED") for r in rows)

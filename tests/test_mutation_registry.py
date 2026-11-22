"""The mutation registry (auditenv/mutation.py) must stay applicable: each planted bug names text
that occurs exactly once in today's grader source, so `make mutation` never silently skips one.
The mutants themselves run in `make mutation` (about 2 min, artifacts/mutation.json)."""

from __future__ import annotations

from auditenv.mutation import MUTANTS, ROOT


def test_every_mutant_applies_to_the_current_source_exactly_once():
    assert len({m[0] for m in MUTANTS}) == len(MUTANTS) == 20
    for mid, rel, old, new, what in MUTANTS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert text.count(old) == 1, mid
        assert old != new and what, mid
        assert text.replace(old, new) != text, mid

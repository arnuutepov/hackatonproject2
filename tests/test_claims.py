"""docs/claims.json is the registry of the README's headline claims. Each claim names the tests that
check it, the command that reproduces it, and the artifact values it quotes. This test keeps the
registry honest: every named test exists, every quoted artifact value is what the artifact holds
now, and the README still says the claim in the registered words."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "docs" / "claims.json").read_text(encoding="utf-8"))
CLAIMS = REGISTRY["claims"]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CRITERIA = {"Контракт и изоляция (25)", "Данные и эталон (25)", "Надёжность награды (25)",
            "Парные проверки (15)", "Инженерия (10)"}


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}


def _value(obj, path: str):
    """Dotted keys, plus three reducers: len(), all(key) over list items, max(key) over dict values."""
    for step in path.split("."):
        if step == "len()":
            obj = len(obj)
        elif step.startswith("all(") and step.endswith(")"):
            obj = all(item[step[4:-1]] for item in obj)
        elif step.startswith("max(") and step.endswith(")"):
            key = step[4:-1]
            items = obj.values() if isinstance(obj, dict) else obj
            obj = max(item[key] for item in items if isinstance(item, dict) and key in item)
        else:
            obj = obj[step]
    return obj


def test_registry_shape_and_coverage():
    ids = [c["id"] for c in CLAIMS]
    assert len(ids) == len(set(ids))
    assert {c["criterion"] for c in CLAIMS} == CRITERIA  # every scoring criterion has a claim
    for c in CLAIMS:
        assert c["claim"] and c["command"] and c["tests"], c["id"]


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c["id"])
def test_every_named_test_exists(claim):
    for node in claim["tests"]:
        file, name = node.split("::")
        path = ROOT / file
        assert path.exists(), node
        assert name in _test_names(path), node


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c["id"])
def test_quoted_artifact_values_hold(claim):
    for ev in claim["evidence"]:
        got = _value(json.loads((ROOT / ev["file"]).read_text(encoding="utf-8")), ev["path"])
        want = ev["equals"]
        if isinstance(want, float):
            assert math.isclose(got, want, abs_tol=1e-4), (claim["id"], ev)
        else:
            assert got == want, (claim["id"], ev)


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c["id"])
def test_readme_states_the_claim(claim):
    assert claim["readme"] in README, claim["id"]

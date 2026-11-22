"""Every file the documentation names must exist in a fresh checkout (outputs under runs/ are made
by the commands and skipped). A reader, human or automated, who follows a path
from README, DESIGN, CHANGELOG or docs/ and finds nothing will doubt the rest. Paths are matched
in backticks; names that are not files of this repository are listed with the reason."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "DESIGN.md", ROOT / "CHANGELOG.md", ROOT / "CLAUDE.md", *sorted((ROOT / "docs").glob("*.md"))]
PATH = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|json|md|csv|jsonl|yml|yaml|toml|txt|svg|html|log|cfg|sh))`")
NOT_OURS = {
    "findings.json": "the submission file named by the case, written by an agent",
    "audit_hard.txt": "the team's own notes outside the repository",
    ".github/workflows/docker-check.yml": "a past file, merged into ci.yml; named only in CHANGELOG history",
    "results.json": "written by examples/jury_run.py into its --out folder",
    "results.csv": "written by examples/jury_run.py into its --out folder",
}
SEARCH = ["", "auditenv", "auditenv/grader", "auditenv/generator", "auditenv/env", "auditenv/schemas", "tests",
          "docs", "examples", "artifacts", "artifacts/logs", "configs", "episodes"]


GENERATED = ("runs/",)  # written by `make check` and other commands, absent from a fresh checkout
SKIP_DIRS = {"runs", ".git", "build", "__pycache__", ".pytest_cache", ".mypy_cache"}


def _exists(name: str) -> bool:
    if any((ROOT / base / name).exists() for base in SEARCH):
        return True
    return any(p.name == Path(name).name and not SKIP_DIRS & set(p.relative_to(ROOT).parts)
               for p in ROOT.rglob(Path(name).name))


def test_every_named_file_exists():
    missing = []
    for doc in DOCS:
        for m in PATH.finditer(doc.read_text(encoding="utf-8")):
            name = m.group(1)
            if name not in NOT_OURS and not name.startswith(GENERATED) and not _exists(name):
                missing.append(f"{doc.name}: {name}")
    assert missing == []

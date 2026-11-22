"""Mutation testing of the grader: plant one known bug at a time and check the tests catch it.

A test suite that passes says little unless it would fail on a wrong grader. Each mutant below
is a realistic grader bug (a rule from the case or the jury's definitions done slightly wrong).
For each one, a scratch copy of the repository gets that single change, the grader-focused tests
run on the copy, and the mutant counts as killed if they fail. The working tree is never touched.

    python -m auditenv.run mutation      # writes artifacts/mutation.json; fails if a mutant survives
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M, R, E = "auditenv/grader/matcher.py", "auditenv/grader/reward.py", "auditenv/grader/episode.py"

# (id, file, exact text, replacement, the rule it breaks)
MUTANTS = [
    ("M01", M, 'if finding["type"] != issue["type"]:', "if False:",
     "issue type is not compared"),
    ("M02", M, 'if set(finding["affected_ids"]) != set(issue["affected_ids"]):',
     'if not set(issue["affected_ids"]) <= set(finding["affected_ids"]):',
     "affected_ids: a superset is accepted instead of the exact set"),
    ("M03", M, 'if set(finding["affected_ids"]) != set(issue["affected_ids"]):',
     'if sorted(finding["affected_ids"]) != sorted(issue["affected_ids"]):',
     "affected_ids compared as lists (a repeated id breaks the match)"),
    ("M04", M, 'if finding["currency"] != issue["currency"]:', "if False:",
     "currency is not compared"),
    ("M05", M, '> issue_tolerance(issue, tolerance):', '> issue_tolerance(issue, tolerance) + 1:',
     "amount tolerance one minor unit too wide"),
    ("M06", M, '> issue_tolerance(issue, tolerance):', '>= issue_tolerance(issue, tolerance):',
     "amount tolerance edge excluded"),
    ("M07", M, 'own = issue.get("tolerance_minor")', "own = None",
     "per-issue tolerance_minor of a jury fixture ignored"),
    ("M08", M, "if evidence_universe is not None and not evidence <= evidence_universe:", "if False:",
     "evidence that exists nowhere is accepted"),
    ("M09", M, 'return any(set(es) <= evidence for es in issue["evidence_sets"])',
     'return all(set(es) <= evidence for es in issue["evidence_sets"])',
     "every alternative evidence set required instead of any one"),
    ("M10", M, 'return any(set(es) <= evidence for es in issue["evidence_sets"])',
     'return any(set(es) & evidence for es in issue["evidence_sets"])',
     "one id of an evidence set is enough"),
    ("M11", M, "for fi in adj[ti]:  # ascending", "for fi in reversed(adj[ti]):  # ascending",
     "tie-break takes the largest finding index"),
    ("M12", M, "if len(fixed) + 1 + _max_matching(adj, later, taken) == target:", "if True:",
     "greedy matching instead of a maximum matching"),
    ("M13", M, 'issues = sorted(issues, key=lambda t: t["issue_id"])', "issues = list(issues)",
     "tie-break follows truth order instead of issue_id"),
    ("M14", R, "MAX_ID_LEN = 256", "MAX_ID_LEN = 64",
     "ids longer than 64 code points rejected (paired checks P16-P20 use 80-100)"),
    ("M15", R, 'if not isinstance(findings, list) or len(findings) > MAX_FINDINGS:',
     'if not isinstance(findings, list) or len(findings) > MAX_FINDINGS + 1:',
     "one finding over the limit accepted"),
    ("M16", R, "return max(-1.0, min(1.0, 2 * f1 - 1 - STEP_COST * inspection_steps))",
     "return max(-1.0, min(1.0, 2 * f1 - 1))", "step cost dropped from the reward"),
    ("M17", R, "    return precision, recall, 2 * tp / (2 * tp + fp + fn)",
     "    return precision, recall, 2 * tp / (2 * tp + fn)", "false positives do not lower F1"),
    ("M18", R, '"inspection_steps": inspection_steps, "reward": -1.0,',
     '"inspection_steps": inspection_steps, "reward": 0.0,', "invalid structure scores 0 instead of -1"),
    ("M19", R, "if tp + fp + fn == 0:  # empty truth and empty findings\n        return precision, recall, 1.0",
     "if tp + fp + fn == 0:  # empty truth and empty findings\n        return precision, recall, 0.0",
     "silence on clean books scores as a total failure"),
    ("M20", E, "                ids.add(f\"{row['date']}:{row['source_currency']}:{row['target_currency']}\")",
     "                pass", "fx rate keys DATE:SRC:TGT missing from the evidence universe"),
]

TESTS = ["tests/test_matcher_exhaustive.py", "tests/test_matcher_metamorphic.py", "tests/test_matcher_reward.py",
         "tests/test_jury_fixture_layouts.py", "tests/test_jury_definitions.py",
         "-k", "not paired_check and not closed_scenario and not contract_check"]
IGNORE = shutil.ignore_patterns(".git", "runs", "artifacts", "__pycache__", "*.pyc", ".pytest_cache", "build",
                                "*.egg-info", "*.zip")


def _env(cwd: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(cwd), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}


def _imports_the_copy(cwd: Path) -> bool:
    """The tests must import the mutated copy, not an installed auditenv."""
    proc = subprocess.run([sys.executable, "-c", "import auditenv.grader.matcher as m; print(m.__file__)"],
                          cwd=cwd, capture_output=True, text=True, env=_env(cwd))
    return proc.returncode == 0 and Path(proc.stdout.strip()).resolve().is_relative_to(cwd.resolve())


def _pytest(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS],
                          cwd=cwd, capture_output=True, text=True, timeout=900, env=_env(cwd))
    lines = [x for x in proc.stdout.splitlines() if x.startswith("FAILED") or " passed" in x or " failed" in x]
    return proc.returncode, (lines[0] if lines else proc.stdout[-200:]).strip()


def run(only: list[str] | None = None) -> dict:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="auditenv-mutation-") as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(ROOT, copy, ignore=IGNORE)
        if not _imports_the_copy(copy):
            return {"passed": False, "error": "the scratch copy does not import its own auditenv", "mutants": []}
        code, line = _pytest(copy)
        if code != 0:
            return {"passed": False, "error": f"the unmutated copy fails: {line}", "mutants": []}
        rows: list[dict] = []
        for mid, rel, old, new, what in MUTANTS:
            if only and mid not in only:
                continue
            path = copy / rel
            original = path.read_text(encoding="utf-8")
            if original.count(old) != 1:
                rows.append({"id": mid, "file": rel, "what": what, "status": "not applied",
                             "detail": "the mutated text is not in the file exactly once"})
                continue
            path.write_text(original.replace(old, new), encoding="utf-8")
            try:
                code, line = _pytest(copy)
            finally:
                path.write_text(original, encoding="utf-8")
            rows.append({"id": mid, "file": rel, "what": what, "status": "killed" if code != 0 else "SURVIVED",
                         "first_failure": line if code != 0 else None})
    killed = sum(r["status"] == "killed" for r in rows)
    return {"what": "one realistic grader bug at a time in a scratch copy; killed = the grader tests fail",
            "tests": TESTS, "mutants": rows, "killed": killed, "total": len(rows),
            "passed": killed == len(rows) and bool(rows), "seconds": round(time.time() - started, 1)}

"""Automatic mutation sweep of the grader core: every one-operator change the generator can make.

`auditenv/mutation.py` plants 20 hand-written, realistic grader bugs. This module complements it
with a mechanical sweep, so nobody had to guess which bugs to try: it walks the syntax tree of
the matcher and the reward and makes every single-operator change below, one at a time:

    comparison   ==/!=, </<=, >/>=, </>, in/not in, is/is not swapped
    arithmetic   +/-, * and / swapped, augmented assignments likewise
    logic        and/or swapped, a `not` removed, an `if` condition negated
    constants    True/False swapped, an integer n replaced by n + 1

Each mutant is written into a scratch copy of the repository, the grader-focused tests of
`make mutation` plus tests/test_mutation_survivors.py run on that copy, and the mutant counts as killed when they fail (or hang past
the timeout). A surviving mutant is either a gap in the tests or an equivalent change that cannot
alter any result; the artifact lists every survivor with its line so that a reader can judge.
The idea of a mechanical sweep next to hand-written mutants comes from another team's public
repository (credited in DESIGN §20); the code here is our own.

    python -m auditenv.run mutation-sweep     # about 10 min on 2 CPUs -> artifacts/mutation_sweep.json

Not part of `make check` (too slow); tests/test_mutation_sweep.py checks that the recorded
artifact still matches today's grader source and that the generator is deterministic.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from auditenv.mutation import IGNORE, ROOT, TESTS, _env, _imports_the_copy

TARGETS = ["auditenv/grader/matcher.py", "auditenv/grader/reward.py"]
# The grader tests of `make mutation` plus the tests written for this sweep's first survivors.
SWEEP_TESTS = [*TESTS[:TESTS.index("-k")], "tests/test_mutation_survivors.py", *TESTS[TESTS.index("-k"):]]
TIMEOUT_S = 180

_CMP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
        ast.GtE: ast.Gt, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
_CMP_EXTRA = {ast.Lt: ast.Gt, ast.Gt: ast.Lt}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
_SYM = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
        ast.In: "in", ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not",
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.And: "and", ast.Or: "or"}


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _sites(tree: ast.AST) -> list[tuple[int, str, str]]:
    """(node index in ast.walk order, operator name, description) for every applicable change."""
    skip = _docstring_ids(tree)
    out: list[tuple[int, str, str]] = []
    for i, node in enumerate(ast.walk(tree)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = type(node.ops[0])
            for repl in filter(None, (_CMP.get(op), _CMP_EXTRA.get(op))):
                out.append((i, f"cmp:{repl.__name__}", f"line {line}: {_SYM[op]} -> {_SYM[repl]}"))
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _BIN:
            repl = _BIN[type(node.op)]
            out.append((i, f"bin:{repl.__name__}", f"line {line}: {_SYM[type(node.op)]} -> {_SYM[repl]}"))
        elif isinstance(node, ast.BoolOp):
            repl = ast.Or if isinstance(node.op, ast.And) else ast.And
            out.append((i, "bool", f"line {line}: {_SYM[type(node.op)]} -> {_SYM[repl]}"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((i, "drop_not", f"line {line}: `not` removed"))
        elif isinstance(node, (ast.If, ast.While)):
            out.append((i, "negate_if", f"line {line}: condition negated"))
        elif isinstance(node, ast.Constant) and id(node) not in skip:
            if node.value is True or node.value is False:
                out.append((i, "bool_const", f"line {line}: {node.value} -> {not node.value}"))
            elif type(node.value) is int:
                out.append((i, "int_plus_one", f"line {line}: {node.value} -> {node.value + 1}"))
    return out


def _apply(tree: ast.AST, index: int, kind: str) -> ast.AST:
    tree = copy.deepcopy(tree)
    nodes = list(ast.walk(tree))
    node: Any = nodes[index]
    if kind.startswith("cmp:"):
        node.ops = [getattr(ast, kind[4:])()]
    elif kind.startswith("bin:"):
        node.op = getattr(ast, kind[4:])()
    elif kind == "bool":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "drop_not":
        parent = next(p for p in nodes if any(c is node for c in ast.iter_child_nodes(p)))
        for field, value in ast.iter_fields(parent):
            if value is node:
                setattr(parent, field, node.operand)
            elif isinstance(value, list):
                setattr(parent, field, [node.operand if v is node else v for v in value])
    elif kind == "negate_if":
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    elif kind == "bool_const":
        node.value = not node.value
    elif kind == "int_plus_one":
        node.value = node.value + 1
    return ast.fix_missing_locations(tree)


def mutants(root: Path = ROOT) -> list[dict]:
    """Every mutant of TARGETS, in a fixed order: {id, file, what, source}. Pure and deterministic."""
    rows: list[dict] = []
    for rel in TARGETS:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        baseline = ast.unparse(tree)
        for index, kind, what in _sites(tree):
            source = ast.unparse(_apply(tree, index, kind))
            if source == baseline:
                continue
            rows.append({"id": f"S{len(rows) + 1:03d}", "file": rel, "what": what, "source": source})
    return rows


def source_digest(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for rel in TARGETS:
        h.update(rel.encode())
        h.update((root / rel).read_bytes())
    return h.hexdigest()


def _pytest(cwd: Path) -> tuple[str, str]:
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *SWEEP_TESTS],
                              cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_S, env=_env(cwd))
    except subprocess.TimeoutExpired:
        return "killed", f"timeout after {TIMEOUT_S} s"
    lines = [x for x in proc.stdout.splitlines() if x.startswith("FAILED") or x.startswith("ERROR")
             or " passed" in x or " failed" in x]
    first = (lines[0] if lines else proc.stdout[-200:]).strip()
    return ("killed" if proc.returncode != 0 else "SURVIVED"), first


def run(workers: int = 2, limit: int | None = None) -> dict:
    started = time.time()
    todo = mutants()[:limit]
    with tempfile.TemporaryDirectory(prefix="auditenv-sweep-") as tmp:
        copies = []
        for w in range(max(1, workers)):
            c = Path(tmp) / f"repo{w}"
            shutil.copytree(ROOT, c, ignore=IGNORE)
            if not _imports_the_copy(c):
                return {"passed": False, "error": "a scratch copy does not import its own auditenv"}
            copies.append(c)
        # The mutants are written with ast.unparse; the unmutated, unparsed sources must pass first.
        for rel in TARGETS:
            p = copies[0] / rel
            p.write_text(ast.unparse(ast.parse(p.read_text(encoding="utf-8"))), encoding="utf-8")
        status, line = _pytest(copies[0])
        if status != "SURVIVED":
            return {"passed": False, "error": f"the unmutated, unparsed copy fails: {line}"}
        for rel in TARGETS:
            (copies[0] / rel).write_bytes((ROOT / rel).read_bytes())

        def one(job: tuple[int, dict]) -> dict:
            n, m = job
            c = copies[n % len(copies)]
            path = c / m["file"]
            original = path.read_bytes()
            path.write_text(m["source"], encoding="utf-8")
            try:
                status, line = _pytest(c)
            finally:
                path.write_bytes(original)
            return {"id": m["id"], "file": m["file"], "what": m["what"], "status": status,
                    "first_failure": line if status == "killed" else None}

        # one job queue per copy, so two mutants never share a scratch copy
        lanes = [[(i, m) for i, m in enumerate(todo) if i % len(copies) == w] for w in range(len(copies))]
        with ThreadPoolExecutor(len(copies)) as pool:
            parts = list(pool.map(lambda lane: [one(job) for job in lane], lanes))
    rows = sorted((r for part in parts for r in part), key=lambda r: r["id"])
    killed = sum(r["status"] == "killed" for r in rows)
    return {"what": "every single-operator mutant of the grader core, one at a time in a scratch copy; "
                    "killed = the grader tests fail or hang",
            "targets": TARGETS, "source_sha256": source_digest(), "tests": SWEEP_TESTS,
            "operators": ["comparison swap", "arithmetic swap", "and/or swap", "`not` removed",
                          "if condition negated", "True/False swap", "integer + 1"],
            "total": len(rows), "killed": killed, "survived": len(rows) - killed,
            "survivors": [r for r in rows if r["status"] != "killed"],
            "mutants": rows, "seconds": round(time.time() - started, 1)}

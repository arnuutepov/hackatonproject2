"""The machine-readable list of ID fields (auditenv/schemas/id_fields.json) and the helpers that use it.

The jury's paired checks rename IDs one-to-one in a copy of an episode. Our own version of that
check (jury_checks.jury_transform) does not keep its own list of fields: it reads them from the
schema here, so a field added to the data model without being added to the list shows up in
tests/test_id_fields.py, which renames every episode kind with this list and then searches every
file of the copy, keys included, for any old ID.

Path syntax, one step per dot:
    name        key of an object (a CSV row is an object)
    name[]      the list under that key, one item per element
    name[][]    a list of lists
    name{}      the KEYS of the object under that key are IDs (e.g. {"INV-1": 300})
The last step is the ID itself; earlier steps only walk into the record.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

from auditenv.common import read_json, read_jsonl, write_json, write_jsonl

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "id_fields.json"


@lru_cache(maxsize=1)
def spec() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _steps(path: str) -> list[tuple[str, int, bool]]:
    """'issues[].evidence_sets[][]' -> [('issues', 1, False), ('evidence_sets', 2, False)]."""
    out = []
    for token in path.split("."):
        keys = token.endswith("{}")
        if keys:
            token = token[:-2]
        depth = 0
        while token.endswith("[]"):
            token, depth = token[:-2], depth + 1
        out.append((token, depth, keys))
    return out


def _slots(obj, steps) -> Iterator[tuple[Any, Any, bool]]:
    """Yield (container, key_or_index, is_dict_key) for every place an ID sits."""
    if not steps or not isinstance(obj, dict):
        return
    (name, depth, keys), rest = steps[0], steps[1:]
    if name not in obj:
        return
    level = [(obj, name)]
    for _ in range(depth):
        nxt: list[tuple[Any, Any]] = []
        for parent, key in level:
            value = parent[key]
            if isinstance(value, list):
                nxt.extend((value, i) for i in range(len(value)))
        level = nxt
    for parent, key in level:
        value = parent[key]
        if rest:
            yield from _slots(value, rest)
        elif keys:
            if isinstance(value, dict):
                yield parent, key, True
        else:
            yield parent, key, False


def field_rules(file_key: str) -> dict:
    return spec()["files"].get(file_key, {})


def _record_rules(rule) -> list[dict]:
    files = rule["as"] if isinstance(rule["as"], list) else [rule["as"]]
    return [field_rules(f) for f in files]


def visit(record, rules: dict, fn: Callable[[str, str], str]) -> None:
    """Call fn(namespace, value) on every ID of one record and store what it returns."""
    for path, rule in rules.items():
        for parent, key, is_keys in list(_slots(record, _steps(path))):
            if isinstance(rule, dict):  # full copy of a record of another file
                for sub_rules in _record_rules(rule):
                    visit(parent[key], sub_rules, fn)
            elif is_keys:
                parent[key] = {fn(rule, k): v for k, v in parent[key].items()}
            elif isinstance(parent[key], str) and parent[key] != "":
                parent[key] = fn(rule, parent[key])


def _records(file_key: str, data) -> list:
    """The records a rule set applies to: CSV rows and JSONL lines one by one, a JSON file whole."""
    return data if file_key.endswith((".csv", ".jsonl")) else [data]


def load_files(episode_dir: Path) -> dict:
    """Every file the schema names, read into Python objects (CSV as row dicts with its header)."""
    out: dict[str, Any] = {}
    for file_key in spec()["files"]:
        path = Path(episode_dir) / file_key
        if not path.exists():
            continue
        if file_key.endswith(".csv"):
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                out[file_key] = {"header": reader.fieldnames, "rows": list(reader)}
        elif file_key.endswith(".jsonl"):
            out[file_key] = read_jsonl(path)
        else:
            out[file_key] = read_json(path)
    return out


def _data(file_key: str, files: dict):
    value = files[file_key]
    return value["rows"] if file_key.endswith(".csv") else value


def collect(files: dict) -> dict[str, set[str]]:
    """IDs by namespace, from every field with a definite namespace (truth's `any` fields only
    point at IDs that exist elsewhere)."""
    found: dict[str, set[str]] = {ns: set() for ns in spec()["namespaces"]}

    def note(ns, value):
        if ns in found:
            found[ns].add(value)
        return value

    for file_key in files:
        for record in _records(file_key, _data(file_key, files)):
            visit(record, field_rules(file_key), note)
    return found


def rename(files: dict, mapping: dict[str, str]) -> None:
    """Rewrite every ID field in place. Values missing from the mapping (fx keys) stay."""
    for file_key in files:
        for record in _records(file_key, _data(file_key, files)):
            visit(record, field_rules(file_key), lambda ns, v: mapping.get(v, v))


def write_files(files: dict, episode_dir: Path) -> None:
    for file_key, value in files.items():
        path = Path(episode_dir) / file_key
        path.parent.mkdir(parents=True, exist_ok=True)
        if file_key.endswith(".csv"):
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=value["header"], lineterminator="\n")
                w.writeheader()
                w.writerows(value["rows"])
        elif file_key.endswith(".jsonl"):
            write_jsonl(path, value)
        else:
            write_json(path, value)

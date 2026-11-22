"""auditenv/schemas/id_fields.json is the one list of ID fields; renaming with it must leave no old ID
anywhere in a copy of any episode kind (keys of JSON objects included), and every entry must be real.

The jury's paired checks rename IDs one-to-one. A field missed by the renamer keeps an old ID next
to new ones, and the two halves of a pair stop being the same episode. Before 1.8.4 our own renamer
kept its list in code and missed the invoice IDs used as keys of `allocated` in provenance; this
test found it."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from auditenv import idfields
from auditenv.judge import SCENARIO_NAMES, generate_episode
from auditenv.jury_checks import jury_transform

KINDS = list(SCENARIO_NAMES) + ["clean", "interacting", "level_hard", "fx_partial_boundary", "cutoff_fx"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("idfields")
    out = {}
    for kind in KINDS:
        d = root / kind
        generate_episode(1, kind, str(d))
        out[kind] = d
    return out


def _all_text(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): p.read_text(encoding="utf-8") for p in folder.rglob("*") if p.is_file()}


def _leftovers(folder: Path, olds) -> list[tuple[str, str]]:
    found = []
    for rel, text in _all_text(folder).items():
        found += [(rel, old) for old in olds if old in text]
    return found


def test_every_file_of_an_episode_is_listed_one_way_or_the_other(corpus):
    spec = idfields.spec()
    listed = set(spec["files"]) | set(spec["files_without_ids"])
    for kind, folder in corpus.items():
        assert set(_all_text(folder)) == listed, kind


def test_every_listed_field_occurs_in_some_episode(corpus):
    hits = {(f, p): 0 for f, rules in idfields.spec()["files"].items() for p in rules}
    for folder in corpus.values():
        files = idfields.load_files(folder)
        for file_key in files:
            for record in idfields._records(file_key, idfields._data(file_key, files)):
                for path in idfields.field_rules(file_key):
                    hits[(file_key, path)] += sum(1 for _ in idfields._slots(record, idfields._steps(path)))
    assert [k for k, n in hits.items() if n == 0] == []


@pytest.mark.parametrize("kind", KINDS)
def test_renaming_with_the_list_leaves_no_old_id(corpus, tmp_path, kind):
    src = corpus[kind]
    files = idfields.load_files(src)
    ids = idfields.collect(files)
    spaces = list(ids.values())
    assert all(not (a & b) for i, a in enumerate(spaces) for b in spaces[i + 1:]), "namespaces overlap"
    mapping = {old: f"~{ns}:{n:05d}~" for ns, olds in ids.items() for n, old in enumerate(sorted(olds))}
    idfields.rename(files, mapping)
    idfields.write_files(files, tmp_path)
    for rel in idfields.spec()["files_without_ids"]:
        shutil.copy(src / rel, tmp_path / rel)
    assert _leftovers(tmp_path, mapping) == []
    text = "".join(_all_text(tmp_path).values())
    assert [new for new in mapping.values() if new not in text] == []


@pytest.mark.parametrize("kind,long_ids", [("D10", False), ("D11", True), ("interacting", True)])
def test_our_paired_check_copy_carries_no_old_id(corpus, tmp_path, kind, long_ids):
    dst = tmp_path / "copy"
    mapping = jury_transform(corpus[kind], dst, seed=7, permute=True, rename=True, long_ids=long_ids)
    assert mapping and _leftovers(dst, mapping) == []


def test_paths_reach_list_items_and_object_keys():
    record = {"a": [{"b": ["X", "Y"]}, {"b": []}], "m": {"K1": 1, "K2": 2}, "s": "", "ll": [["P"], ["Q", "R"]]}
    rules = {"a[].b[]": "x", "m{}": "x", "s": "x", "ll[][]": "x", "missing[].z": "x"}
    seen = []
    idfields.visit(record, rules, lambda ns, v: seen.append(v) or v.lower())
    assert sorted(seen) == ["K1", "K2", "P", "Q", "R", "X", "Y"]
    assert record == {"a": [{"b": ["x", "y"]}, {"b": []}], "m": {"k1": 1, "k2": 2}, "s": "", "ll": [["p"], ["q", "r"]]}

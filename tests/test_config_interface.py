"""The config is a public interface: the jury generates hidden episodes via reset(seed, config)."""

from pathlib import Path

import pytest

from auditenv.common import read_json
from auditenv.env import AuditEnv
from auditenv.generator import ConfigError, generate_episode
from auditenv.generator.config import DEFAULT_CONFIG
from auditenv.generator.validate import validate_episode

ROOT = Path(__file__).resolve().parents[1]


def scenarios(ep):
    return [c["scenario"] for c in read_json(ep / "private" / "provenance.json")["changes"] if "scenario" in c]


def test_reset_accepts_partial_dict_and_path(tmp_path):
    env = AuditEnv(workdir=tmp_path)
    obs, info = env.reset(seed=3, config={"force_clean": True})
    ep = next(tmp_path.glob(info["episode_id"]))
    assert read_json(ep / "private" / "truth.json")["issues"] == []
    obs, info = env.reset(seed=3, config="interacting")
    ep = next(tmp_path.glob(info["episode_id"]))
    assert sorted(scenarios(ep)) == ["cutoff_fx", "fx_partial_boundary"]
    obs, info2 = env.reset()  # keeps the last config, continues the seed sequence
    assert sorted(scenarios(next(tmp_path.glob(info2["episode_id"])))) == ["cutoff_fx", "fx_partial_boundary"]


@pytest.mark.parametrize("preset,check", [
    ("clean", lambda t, s: t == [] and s == []),
    ("interacting", lambda t, s: sorted(s) == ["cutoff_fx", "fx_partial_boundary"]),
    ("fx_partial_boundary.json", lambda t, s: s.count("fx_partial_boundary") == 1),
    ("cutoff_fx", lambda t, s: s.count("cutoff_fx") == 1),
    ("errors_only", lambda t, s: s == [] and {i["type"] for i in t} == {
        "duplicate_invoice", "partial_payment", "fx_conversion", "period_cutoff"}),
])
def test_presets_guarantee_composition_for_many_seeds(preset, check, tmp_path):
    for seed in range(12):
        ep = generate_episode(seed, preset, tmp_path)
        assert validate_episode(ep) == []
        truth = read_json(ep / "private" / "truth.json")["issues"]
        assert check(truth, scenarios(ep)), (preset, seed)


def test_hidden_like_recipe_has_five_clean_and_five_interacting(tmp_path):
    from auditenv.run import cmd_generate
    eps = cmd_generate(ROOT / "configs" / "hidden_like.json", tmp_path)
    assert len(eps) == 20
    clean = [e for e in eps if not read_json(e / "private" / "truth.json")["issues"]]
    interacting = [e for e in eps[5:10] if scenarios(e)]
    assert len(clean) == 5 and len(interacting) == 5


def test_same_seed_and_config_reproduce_content(tmp_path):
    a = generate_episode(77, {"interactions": {"cutoff_fx": {"min": 1}}}, tmp_path / "a")
    b = generate_episode(77, {"interactions": {"cutoff_fx": {"min": 1}}}, tmp_path / "b")
    for f in ("ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "manifest.json", "policy.json"):
        assert (a / "public" / f).read_bytes() == (b / "public" / f).read_bytes()


@pytest.mark.parametrize("bad,needle", [
    ({"forse_clean": True}, "unknown config keys"),
    ({"errors": {"duplicate_invoice": {"min": 3, "max": 1}}}, "min <= max"),
    ({"force_clean": True, "interactions": {"cutoff_fx": {"min": 1}}}, "conflicts"),
    ({"interactions": {"cutoff_fx": {"min": 1}}, "foreign_invoice_share": 0}, "foreign_invoice_share"),
    ({"errors": {"fx_conversion": {"enabled": False, "min": 1}}}, "requires enabled"),
    ({"base_currency": "BTC"}, "base_currency"),
    ({"base_currency": "USD"}, "foreign_currencies"),
    ({"ledger_rows": [10, 20]}, "ledger_rows"),
    ({"timezones": ["5 hours"]}, "timezones"),
    ({"errors": {t: {"min": 4, "max": 4} for t in ("duplicate_invoice", "partial_payment", "fx_conversion",
                                                    "period_cutoff")}}, "exceed max_errors"),
])
def test_invalid_configs_fail_loudly(bad, needle, tmp_path):
    with pytest.raises(ConfigError, match=needle):
        AuditEnv(workdir=tmp_path).reset(seed=1, config=bad)


def test_any_base_currency_including_zero_decimal(tmp_path):
    currencies = ["KZT", "USD", "EUR", "JPY"]
    for base in currencies:
        cfg = {"base_currency": base, "foreign_currencies": [c for c in currencies if c != base],
               "clean_probability": 0.0}
        for seed in range(6):
            ep = generate_episode(seed, cfg, tmp_path)
            assert validate_episode(ep) == [], (base, seed)
            assert read_json(ep / "public" / "manifest.json")["base_currency"] == base


def test_config_schema_matches_defaults():
    jsonschema = pytest.importorskip("jsonschema")
    schema = read_json(ROOT / "auditenv" / "schemas" / "config.schema.json")
    jsonschema.validate(DEFAULT_CONFIG, schema)
    for path in list((ROOT / "configs").glob("*.json")) + list((ROOT / "auditenv" / "presets").glob("*.json")):
        if path.name == "jury_matrix.json":  # a matrix of partial configs, one per jury scenario
            for spec in read_json(path)["scenarios"].values():
                jsonschema.validate(spec["config"], schema)
            continue
        jsonschema.validate(read_json(path), schema)


@pytest.mark.parametrize("bad,needle", [
    ({"n_customers": [1009, 1009]}, "n_customers"),            # audit F03: name space could be exhausted
    ({"ledger_rows": [2000, 2000]}, "ledger_rows"),            # audit F07: final size above contract
    ({"n_plain_invoices": [200, 200]}, "n_plain_invoices"),    # audit F07: documents above contract
    ({"max_errors": 0, "clean_probability": 0}, "force_clean"),  # audit F08: unsatisfiable
])
def test_audit_configs_are_rejected_immediately(bad, needle, tmp_path):
    import time
    t0 = time.monotonic()
    with pytest.raises(ConfigError, match=needle):
        generate_episode(0, bad, tmp_path)
    assert time.monotonic() - t0 < 1.0


def test_name_generation_is_bounded_when_the_name_space_is_tiny(tmp_path, monkeypatch):
    from auditenv.generator import company
    monkeypatch.setattr(company, "NAME_A", ["Solo"])
    monkeypatch.setattr(company, "NAME_B", ["Trading"])
    monkeypatch.setattr(company, "NAME_C", ["LLP"])
    ep = generate_episode(1, {"n_customers": [20, 20], "n_vendors": [10, 10]}, tmp_path)
    names = [e["name"] for e in read_json(ep / "public" / "manifest.json")["entities"]]
    assert len(names) == len(set(names)) == 30


def test_final_sizes_respect_the_contract_for_extreme_valid_configs(tmp_path):
    import random
    rng = random.Random(11)
    for i in range(25):
        lo, lr = rng.randint(12, 110), rng.randint(200, 1900)
        cfg = {"n_plain_invoices": [lo, rng.randint(lo, 110)], "ledger_rows": [lr, rng.randint(lr, 1900)],
               "n_twin_orders": [0, 5], "n_refunds": [0, 5], "clean_probability": 0.0}
        m = read_json(generate_episode(i, cfg, tmp_path) / "public" / "manifest.json")
        assert 200 <= m["record_counts"]["ledger"] <= 2000 and 20 <= m["document_count"] <= 200

"""Generator configuration — a public interface: the jury generates hidden episodes with
`env.reset(seed, config)`, where `config` may be

    None                   -> defaults
    {"force_clean": True}  -> a PARTIAL dict, merged over the defaults
    "configs/x.json"       -> a JSON file (also merged over the defaults)

Every resolved config is validated: unknown keys, wrong types or impossible combinations
raise ConfigError with a precise message instead of silently producing a different episode.
Full reference: docs in README ("Конфигурация") and auditenv/schemas/config.schema.json.
Built-in presets (auditenv/presets/*.json) can be passed by name: config="interacting".
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from auditenv.common import CURRENCY_EXPONENT, canonical_json, read_json

ERROR_NAMES = ("duplicate_invoice", "partial_payment", "fx_conversion", "period_cutoff")
SCENARIO_NAMES = ("fx_partial_boundary", "cutoff_fx")
META_KEYS = ("seeds", "split", "description", "preset")
PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"  # package data  # allowed in files, ignored by the generator

DEFAULT_CONFIG: dict = {
    "config_version": "1.0.0",
    "base_currency": "KZT",
    "foreign_currencies": ["USD", "EUR", "JPY"],
    "timezones": ["+05:00", "+06:00", "+03:00", "+00:00"],
    "n_customers": [6, 18],
    "n_vendors": [3, 8],
    "n_plain_invoices": [20, 70],
    # ENV 1.4.0: the default mix leans on evidence analysis that a sweep over references cannot do:
    # more foreign documents, up to 3 FX errors and up to 2 of each interaction scenario; the
    # single non-FX errors stay present (up to 1 each). Calibrated on the hidden-set recipe:
    # simple reconciliation F1 0.84 -> 0.74, oracle 1.0 (DESIGN.md, "Калибровка сложности").
    # Only maxima and shares change: a minimum here would contradict the clean preset.
    "foreign_invoice_share": 0.4,
    "n_boundary_invoices": [1, 3],
    "n_recurring_customers": [1, 3],
    "n_unidentified_receipts": [0, 2],
    "n_customer_advances": [1, 2],
    "n_last_day_invoices": [1, 2],
    "n_twin_orders": [1, 2],
    "n_refunds": [1, 2],
    "n_last_day_night_payments": [1, 2],
    "posting_delay_share": 0.15,
    "foreign_night_payment_share": 0.3,
    "ledger_rows": [300, 900],
    "invoice_outcomes": {"paid_full": 0.6, "split": 0.1, "partial_legit": 0.1, "unpaid": 0.2},
    "clean_probability": 0.25,
    "force_clean": False,
    "errors": {
        "duplicate_invoice": {"enabled": True, "min": 0, "max": 1},
        "partial_payment": {"enabled": True, "min": 0, "max": 1, "missing_share": 0.4},
        "fx_conversion": {"enabled": True, "min": 0, "max": 3},
        "period_cutoff": {"enabled": True, "min": 0, "max": 1},
    },
    "interactions": {
        "fx_partial_boundary": {"enabled": True, "min": 0, "max": 2},
        "cutoff_fx": {"enabled": True, "min": 0, "max": 2},
    },
    "max_errors": 12,
}

RANGE_KEYS = ("n_customers", "n_vendors", "n_plain_invoices", "n_boundary_invoices", "n_recurring_customers",
              "n_unidentified_receipts", "n_customer_advances", "n_last_day_invoices", "n_twin_orders",
              "n_refunds", "n_last_day_night_payments", "ledger_rows")
# Bounds that keep every episode inside the size contract (200-2000 ledger rows, 20-200 documents,
# 0-12 issues) and keep generation finite. The final sizes are also checked after injection.
RANGE_LIMITS = {
    "n_customers": (2, 60), "n_vendors": (1, 30), "n_plain_invoices": (12, 110),
    "n_boundary_invoices": (0, 5), "n_recurring_customers": (0, 5), "n_unidentified_receipts": (0, 5),
    "n_customer_advances": (0, 5), "n_last_day_invoices": (0, 5), "n_twin_orders": (0, 5), "n_refunds": (0, 5),
    "n_last_day_night_payments": (0, 5), "ledger_rows": (200, 1900),
}
SHARE_KEYS = ("foreign_invoice_share", "posting_delay_share", "foreign_night_payment_share", "clean_probability")


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def validate_config(cfg: dict) -> dict:
    """Raise ConfigError on anything the generator would not honour exactly."""
    known = set(DEFAULT_CONFIG) | set(META_KEYS)
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")

    def is_int(v):
        return type(v) is int

    for key in RANGE_KEYS:
        v = cfg[key]
        if not (isinstance(v, list) and len(v) == 2 and all(map(is_int, v)) and 0 <= v[0] <= v[1]):
            raise ConfigError(f"{key} must be [lo, hi] with integers 0 <= lo <= hi")
    for key, (lo, hi) in RANGE_LIMITS.items():
        if not (lo <= cfg[key][0] and cfg[key][1] <= hi):
            raise ConfigError(f"{key} must lie within [{lo}, {hi}] (episode size contract)")
    for key in SHARE_KEYS:
        if not isinstance(cfg[key], (int, float)) or isinstance(cfg[key], bool) or not 0 <= cfg[key] <= 1:
            raise ConfigError(f"{key} must be a number in [0, 1]")
    if not isinstance(cfg["force_clean"], bool):
        raise ConfigError("force_clean must be true or false")
    if cfg["base_currency"] not in CURRENCY_EXPONENT:
        raise ConfigError(f"base_currency must be one of {sorted(CURRENCY_EXPONENT)}")
    fc = cfg["foreign_currencies"]
    if not (isinstance(fc, list) and 2 <= len(fc) == len(set(fc))
            and all(c in CURRENCY_EXPONENT and c != cfg["base_currency"] for c in fc)):
        raise ConfigError("foreign_currencies: >= 2 distinct known currencies, excluding the base currency")
    tz = cfg["timezones"]
    if not (isinstance(tz, list) and tz and all(isinstance(t, str) and re.fullmatch(r"[+-](0\d|1[0-4]):[0-5]\d", t) for t in tz)):
        raise ConfigError("timezones must be a non-empty list like ['+05:00', '-04:00']")
    outcomes = cfg["invoice_outcomes"]
    if set(outcomes) != {"paid_full", "split", "partial_legit", "unpaid"} or \
            any(not isinstance(w, (int, float)) or w < 0 for w in outcomes.values()) or sum(outcomes.values()) <= 0:
        raise ConfigError("invoice_outcomes: non-negative weights for paid_full, split, partial_legit, unpaid")
    if not is_int(cfg["max_errors"]) or not 0 <= cfg["max_errors"] <= 12:
        raise ConfigError("max_errors must be an integer in [0, 12]")

    required = 0
    for table, names, size in (("errors", ERROR_NAMES, 1), ("interactions", SCENARIO_NAMES, 2)):
        if set(cfg[table]) != set(names):
            raise ConfigError(f"{table} must configure exactly {list(names)}")
        for name in names:
            spec = cfg[table][name]
            extra = set(spec) - {"enabled", "min", "max", "missing_share"}
            if extra:
                raise ConfigError(f"{table}.{name}: unknown keys {sorted(extra)}")
            if not isinstance(spec["enabled"], bool) or not is_int(spec["min"]) or not is_int(spec["max"]) \
                    or not 0 <= spec["min"] <= spec["max"]:
                raise ConfigError(f"{table}.{name}: enabled bool, integers 0 <= min <= max")
            if spec["enabled"]:
                required += size * spec["min"]
            elif spec["min"]:
                raise ConfigError(f"{table}.{name}: min > 0 requires enabled = true")
    if not 0 <= cfg["errors"]["partial_payment"].get("missing_share", 0.4) <= 1:
        raise ConfigError("errors.partial_payment.missing_share must be in [0, 1]")
    if required > cfg["max_errors"]:
        raise ConfigError(f"minimum issues requested ({required}) exceed max_errors ({cfg['max_errors']})")
    mins = {n: cfg["errors"][n]["min"] if cfg["errors"][n]["enabled"] else 0 for n in ERROR_NAMES}
    mins.update({n: cfg["interactions"][n]["min"] if cfg["interactions"][n]["enabled"] else 0 for n in SCENARIO_NAMES})
    if (mins["fx_conversion"] or mins["fx_partial_boundary"] or mins["cutoff_fx"]) and cfg["foreign_invoice_share"] == 0:
        raise ConfigError("FX errors/scenarios with min > 0 need foreign_invoice_share > 0")
    if (mins["partial_payment"] or mins["fx_partial_boundary"]) and outcomes["paid_full"] == 0:
        raise ConfigError("partial-payment errors with min > 0 need invoice_outcomes.paid_full > 0")
    if (mins["period_cutoff"] or mins["cutoff_fx"]) and outcomes["unpaid"] == 0:
        raise ConfigError("cut-off errors with min > 0 need invoice_outcomes.unpaid > 0")
    any_enabled = any(cfg["errors"][n]["enabled"] for n in ERROR_NAMES) or \
        any(cfg["interactions"][n]["enabled"] for n in SCENARIO_NAMES)
    can_be_dirty = not cfg["force_clean"] and cfg["clean_probability"] < 1
    if can_be_dirty and cfg["max_errors"] == 0:
        raise ConfigError("max_errors = 0 allows no errors: set force_clean = true for a clean company")
    if can_be_dirty and not any_enabled:
        raise ConfigError("no error type or scenario is enabled: set force_clean = true for a clean company")
    placeable = any(cfg[t][n]["enabled"] and cfg[t][n]["max"] >= 1
                    for t, names in (("errors", ERROR_NAMES), ("interactions", SCENARIO_NAMES)) for n in names)
    if can_be_dirty and not placeable:
        raise ConfigError("every enabled error type and scenario has max = 0, so a dirty episode cannot get "
                          "any error: raise a max or set force_clean = true for a clean company")
    if cfg["force_clean"] and required:
        raise ConfigError("force_clean conflicts with min > 0")
    if required and cfg["clean_probability"] > 0:
        cfg["clean_probability"] = 0.0  # a requested minimum means the episode cannot be clean
    problem = feasibility_problem(cfg, mins)
    if problem:
        raise ConfigError(problem)
    return cfg


# Late-month invoices whose payment would fall after the period end stay unpaid, so the realized
# payment rate is below the nominal one. 0.75 is conservative (measured ~0.8 on default configs).
PAYMENT_RATE_DISCOUNT = 0.75
DOCUMENT_LIMITS = (20, 200)


def _estimated_documents(cfg: dict, pick) -> float:
    """Expected document count of a company sampled at the low (pick=min) or high (pick=max) end of
    every range: invoices of all roles plus payments plus standalone receipts."""
    w = cfg["invoice_outcomes"]
    total = sum(w.values()) or 1.0
    pay_rate = (w["paid_full"] + 2 * w["split"] + w["partial_legit"]) / total * PAYMENT_RATE_DISCOUNT
    plain = pick(cfg["n_plain_invoices"])
    invoices = (plain + min(plain, pick(cfg["n_twin_orders"])) + pick(cfg["n_last_day_invoices"])
                + pick(cfg["n_boundary_invoices"]) + pick(cfg["n_recurring_customers"]) * 3)
    receipts = pick(cfg["n_customer_advances"]) + pick(cfg["n_unidentified_receipts"]) + pick(cfg["n_refunds"])
    duplicates = cfg["errors"]["duplicate_invoice"]["min"] if cfg["errors"]["duplicate_invoice"]["enabled"] else 0
    return invoices * (1 + pay_rate) + receipts + duplicates  # each duplicate is one more invoice document


def feasibility_problem(cfg: dict, mins: dict) -> str | None:
    """Reject combinations the generator cannot satisfy, at config time and with numbers.

    Necessary conditions in expectation, evaluated for the LARGEST company the config allows:
    each requested error needs its own plain invoice of a specific kind (foreign / unpaid /
    paid in one payment). A config that fails them would only burn all generation attempts and
    end in GenerationError; one that passes is generated (stress-tested on extreme configs,
    `python -m auditenv.run stress`). Document counts are checked against the size contract
    for the smallest and largest company the ranges allow.
    """
    lo_docs = _estimated_documents(cfg, lambda r: r[0])
    hi_docs = _estimated_documents(cfg, lambda r: r[1])
    if lo_docs > DOCUMENT_LIMITS[1]:
        return (f"size contract: even the smallest company this config allows has about {lo_docs:.0f} documents "
                f"(limit {DOCUMENT_LIMITS[1]}); lower n_plain_invoices or the distractor counts")
    if hi_docs < DOCUMENT_LIMITS[0]:
        return (f"size contract: even the largest company this config allows has about {hi_docs:.0f} documents "
                f"(minimum {DOCUMENT_LIMITS[0]}); raise n_plain_invoices")
    if cfg["force_clean"] or not any(mins.values()):
        return None
    w = cfg["invoice_outcomes"]
    total = sum(w.values()) or 1.0
    # Late-month invoices stay unpaid (their payment would fall after the period end), so the
    # realized paid share is below the nominal one (measured ~0.73x) and the unpaid share above it.
    paid = w["paid_full"] / total * PAYMENT_RATE_DISCOUNT
    unpaid = w["unpaid"] / total
    f = cfg["foreign_invoice_share"]
    # Distractors consume plain invoices: twin orders turn plain invoices into twins, refunds take
    # base-currency invoices paid in one payment, night payments take foreign ones.
    n_max = cfg["n_plain_invoices"][1]
    n = max(0, n_max - cfg["n_twin_orders"][0])
    paid_base = max(0.0, n * (1 - f) * paid - cfg["n_refunds"][0])
    paid_foreign = max(0.0, n * f * paid - cfg["n_last_day_night_payments"][0])
    needs = [
        ("plain invoices", sum(mins.values()), n),
        ("foreign invoices (fx_conversion, fx_partial_boundary, cutoff_fx)",
         mins["fx_conversion"] + mins["fx_partial_boundary"] + mins["cutoff_fx"], n * f),
        ("unpaid invoices (period_cutoff, cutoff_fx)", mins["period_cutoff"] + mins["cutoff_fx"], n * unpaid),
        ("unpaid foreign invoices (cutoff_fx)", mins["cutoff_fx"], n * f * unpaid),
        ("invoices paid in one payment and not used by refunds or night payments (partial_payment, "
         "fx_partial_boundary)", mins["partial_payment"] + mins["fx_partial_boundary"], paid_base + paid_foreign),
        ("foreign invoices paid in one payment (fx_partial_boundary)", mins["fx_partial_boundary"], paid_foreign),
    ]
    for what, need, expected in needs:
        if need and expected < need:
            return (f"error minimums need {need} {what}, but even at n_plain_invoices max = {n_max} the company is "
                    f"expected to have only {expected:.1f}; raise n_plain_invoices (or foreign_invoice_share / "
                    f"the matching invoice_outcomes share) or lower the minimums")
    return None


def resolve_config(config=None) -> dict:
    """None | partial dict | path -> full validated config."""
    if config is None:
        return validate_config(copy.deepcopy(DEFAULT_CONFIG))
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.is_file():
            return load_config(path)
        preset = PRESET_DIR / (path.name if path.suffix == ".json" else f"{path.name}.json")
        if path.parent == Path(".") and preset.is_file():
            return load_config(preset)
        raise ConfigError(f"config file not found and not a preset name: {config!s} "
                          f"(presets: {sorted(p.stem for p in PRESET_DIR.glob('*.json'))})")
    if isinstance(config, dict):
        return load_config(overrides=config)
    raise ConfigError("config must be None, a dict of overrides or a path to a JSON file")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        cfg = _merge(cfg, read_json(Path(path)))
    if overrides:
        cfg = _merge(cfg, overrides)
    return validate_config(cfg)


def config_hash(cfg: dict) -> str:
    gen_part = {k: v for k, v in cfg.items() if k not in META_KEYS}
    return hashlib.sha256(canonical_json(gen_part).encode()).hexdigest()[:16]

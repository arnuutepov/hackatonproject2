"""Independent episode validator: re-derives every issue from PUBLIC records only.

It does not import generator internals. For each episode it checks:
  1. schema: required files, columns, integer money, explicit nulls;
  2. the ledger balances per currency (errors are double-entry too);
  3. every truth issue is real: its amount is recomputed from records and every
     evidence id exists in public files;
  4. completeness: every discrepancy found in the records is declared in truth
     (so clean episodes are really clean and nothing is unresolvable).
"""

from __future__ import annotations

from typing import Any
import csv
from collections import defaultdict
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

from auditenv.common import LEDGER_COLUMNS, PUBLIC_FILES, canonical_json, decimal_context, read_json, read_jsonl


FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")


def load_public(episode_dir: Path) -> dict:
    """The public files as records. Column ORDER in the CSV files does not matter (a jury copy may be
    rewritten with another header order); every record comes back in the canonical key order."""
    pub = Path(episode_dir) / "public"
    with open(pub / "ledger.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or sorted(reader.fieldnames) != sorted(LEDGER_COLUMNS):
            raise ValueError(f"ledger.csv must have exactly the columns {list(LEDGER_COLUMNS)}")
        ledger = []
        for r in reader:
            row: dict[str, Any] = {k: (r[k] if r[k] != "" else None) for k in LEDGER_COLUMNS}
            row["debit_minor"], row["credit_minor"] = int(row["debit_minor"]), int(row["credit_minor"])
            ledger.append(row)
    with open(pub / "fx.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or sorted(reader.fieldnames) != sorted(FX_COLUMNS):
            raise ValueError(f"fx.csv must have exactly the columns {list(FX_COLUMNS)}")
        fx = [{k: r[k] for k in FX_COLUMNS} for r in reader]
    return {
        "manifest": read_json(pub / "manifest.json"),
        "policy": read_json(pub / "policy.json"),
        "ledger": ledger,
        "invoices": read_jsonl(pub / "invoices.jsonl"),
        "payments": read_jsonl(pub / "payments.jsonl"),
        "fx": fx,
    }


def find_duplicates(data: dict) -> dict[tuple, int]:
    groups = defaultdict(list)
    for inv in data["invoices"]:
        key = (inv["entity_id"], inv["issued_at"], inv["currency"], inv["total_minor"],
               canonical_json(inv["line_items"]))
        groups[key].append(inv)
    return {tuple(sorted(i["invoice_id"] for i in g)): g[0]["total_minor"]
            for g in groups.values() if len(g) > 1}


def find_unallocated(data: dict) -> dict[tuple, int]:
    """Suspense credits of receipts whose bank record names an invoice."""
    referenced = {p["payment_id"] for p in data["payments"] if p["invoice_refs"]}
    out: dict[tuple, int] = defaultdict(int)
    for r in data["ledger"]:
        if r["account"] == "2900" and r["payment_id"] in referenced:
            out[(r["payment_id"],)] += r["credit_minor"] - r["debit_minor"]
    return dict(out)


def _converter(data: dict):
    """Independent re-implementation of the policy: rate of the document's local date, half-even."""
    rates = {(r["date"], r["source_currency"]): Decimal(r["rate"]) for r in data["fx"]}
    base = data["manifest"]["base_currency"]
    exps = data["manifest"]["currency_exponents"]

    def conv(amount: int, currency: str, local_date: str) -> int:
        if currency == base:
            return amount
        scaled = Decimal(amount) * rates[(local_date, currency)] * Decimal(10) ** (exps[base] - exps[currency])
        return int(scaled.quantize(Decimal(1), ROUND_HALF_EVEN))

    return conv


@decimal_context
def find_fx_mismatches(data: dict) -> dict[tuple, int]:
    conv, base = _converter(data), data["manifest"]["base_currency"]
    ar_debit: defaultdict[str, int] = defaultdict(int)
    cash_debit: defaultdict[str, int] = defaultdict(int)
    for r in data["ledger"]:
        if r["account"] == "1100" and r["invoice_id"]:
            ar_debit[r["invoice_id"]] += r["debit_minor"]
        if r["account"] == "1000" and r["payment_id"]:
            cash_debit[r["payment_id"]] += r["debit_minor"]
    out = {}
    for inv in data["invoices"]:
        if inv["currency"] != base and inv["invoice_id"] in ar_debit:
            diff = ar_debit[inv["invoice_id"]] - conv(inv["total_minor"], inv["currency"], inv["issued_at"][:10])
            if diff:
                out[(inv["invoice_id"],)] = abs(diff)
    for pay in data["payments"]:
        if pay["currency"] != base and pay["payment_id"] in cash_debit:
            diff = cash_debit[pay["payment_id"]] - conv(pay["amount_minor"], pay["currency"], pay["timestamp"][:10])
            if diff:
                out[(pay["payment_id"],)] = abs(diff)
    return out


def find_cutoff(data: dict) -> dict[tuple, int]:
    start = datetime.fromisoformat(data["manifest"]["period_start"])
    end = datetime.fromisoformat(data["manifest"]["period_end"])
    ar_debit: defaultdict[str, int] = defaultdict(int)
    for r in data["ledger"]:
        if r["account"] == "1100" and r["invoice_id"]:
            ar_debit[r["invoice_id"]] += r["debit_minor"]
    return {(inv["invoice_id"],): ar_debit[inv["invoice_id"]] for inv in data["invoices"]
            if inv["invoice_id"] in ar_debit and not start <= datetime.fromisoformat(inv["issued_at"]) <= end}


@decimal_context
def verify_evidence(issue: dict, evidence_ids: list[str], data: dict) -> str | None:
    """Independent sufficiency check: can the claim be re-derived from ONLY the cited records
    (plus the manifest: period, base currency, exponents)? Returns a reason if not."""
    m = data["manifest"]
    base, exps = m["base_currency"], m["currency_exponents"]
    rows = {r["row_id"]: r for r in data["ledger"]}
    docs = {d["doc_id"]: ("invoice", d) for d in data["invoices"]}
    docs.update({d["doc_id"]: ("payment", d) for d in data["payments"]})
    fx = {f"{r['date']}:{r['source_currency']}:{r['target_currency']}": r for r in data["fx"]}
    cited_rows = [rows[e] for e in evidence_ids if e in rows]
    cited_docs = [docs[e] for e in evidence_ids if e in docs]
    cited_fx = [fx[e] for e in evidence_ids if e in fx]
    ids, amount, kind = issue["affected_ids"], issue["expected_amount_minor"], issue["type"]

    if kind == "duplicate_invoice":
        invs = [d for k, d in cited_docs if k == "invoice" and d["invoice_id"] in ids]
        if sorted(d["invoice_id"] for d in invs) != sorted(set(ids)):
            return "both invoice documents must be cited"
        keys = {(d["entity_id"], d["issued_at"], d["currency"], d["total_minor"],
                 canonical_json(d["line_items"])) for d in invs}
        if len(keys) != 1 or invs[0]["total_minor"] != amount or invs[0]["currency"] != issue["currency"]:
            return "cited invoices are not identical or amount differs"
        return None

    if kind == "partial_payment":
        pays = [d for k, d in cited_docs if k == "payment" and d["payment_id"] == ids[0]]
        susp = [r for r in cited_rows if r["account"] == "2900" and r["payment_id"] == ids[0]]
        if not pays or not pays[0]["invoice_refs"]:
            return "payment document naming an invoice must be cited"
        if not susp or sum(r["credit_minor"] - r["debit_minor"] for r in susp) != amount:
            return "suspense rows must be cited and sum to the amount"
        return None

    if kind == "fx_conversion":
        doc = next((d for k, d in cited_docs if d.get("invoice_id" if k == "invoice" else "payment_id") == ids[0]
                    and (k == "invoice" or "invoice_refs" in d)), None)
        if doc is None:
            return "the converted document must be cited"
        is_inv = "invoice_id" in doc and "line_items" in doc
        local_date = (doc["issued_at"] if is_inv else doc["timestamp"])[:10]
        rate_row = next((r for r in cited_fx if r["date"] == local_date and r["source_currency"] == doc["currency"]
                         and r["target_currency"] == base), None)
        booked = next((r for r in cited_rows if r["debit_minor"] and (
            (is_inv and r["account"] == "1100" and r["invoice_id"] == ids[0]) or
            (not is_inv and r["account"] == "1000" and r["payment_id"] == ids[0]))), None)
        if rate_row is None or booked is None:
            return "the booked row and the fx record of the document's local date must be cited"
        value = Decimal(doc["total_minor"] if is_inv else doc["amount_minor"]) * Decimal(rate_row["rate"]) \
            * Decimal(10) ** (exps[base] - exps[doc["currency"]])
        correct = int(value.quantize(Decimal(1), ROUND_HALF_EVEN))
        if abs(booked["debit_minor"] - correct) != amount:
            return "cited records do not reproduce the amount"
        return None

    if kind == "period_cutoff":
        start, end = datetime.fromisoformat(m["period_start"]), datetime.fromisoformat(m["period_end"])
        inv = next((d for k, d in cited_docs if k == "invoice" and d["invoice_id"] == ids[0]), None)
        ar = next((r for r in cited_rows if r["account"] == "1100" and r["invoice_id"] == ids[0]
                   and r["debit_minor"]), None)
        if inv is None or ar is None:
            return "invoice document and its receivable row must be cited"
        if start <= datetime.fromisoformat(inv["issued_at"]) <= end:
            return "cited invoice is dated inside the period"
        if not start <= datetime.fromisoformat(ar["timestamp"]) <= end or ar["debit_minor"] != amount:
            return "cited row is not booked in this period for the amount"
        return None
    return "unknown issue type"


def reconstruct_pre_error(data: dict, changes: list[dict]) -> dict:
    """Undo every recorded change (newest first) to recover the books before any error."""
    ledger = {r["row_id"]: dict(r) for r in data["ledger"]}
    docs = {d["doc_id"]: ("invoice", dict(d)) for d in data["invoices"]}
    docs.update({d["doc_id"]: ("payment", dict(d)) for d in data["payments"]})
    for change in reversed(changes):
        for row_id in change.get("ledger_added", []):
            ledger.pop(row_id, None)
        for row in change.get("ledger_removed", []):
            ledger[row["row_id"]] = dict(row)
        for doc_id in change.get("documents_added", []):
            docs.pop(doc_id, None)
        for doc in change.get("documents_before", []):
            kind = "invoice" if "line_items" in doc else "payment"
            docs[doc["doc_id"]] = (kind, dict(doc))
    return {**data, "ledger": list(ledger.values()),
            "invoices": [d for k, d in docs.values() if k == "invoice"],
            "payments": [d for k, d in docs.values() if k == "payment"]}


@decimal_context
def check_pre_error_state(data: dict, truth: dict, provenance: dict) -> list[str]:
    """Restore the books before injection and check (1) accounting equalities hold there,
    (2) no discrepancy exists there, (3) every change is linked to declared issues and is
    visible in the public records."""
    problems = []
    changes = [c for c in provenance.get("changes", []) if c.get("op") != "clean_episode"]
    issue_ids = {t["issue_id"] for t in truth["issues"]}
    linked = set()
    for c in changes:
        ids = set(c.get("issue_ids", [])) | ({c["issue_id"]} if "issue_id" in c else set())
        if not ids or not ids <= issue_ids:
            problems.append(f"provenance change not linked to declared issues: {sorted(ids)}")
        if not any(c.get(k) for k in ("ledger_removed", "ledger_added", "documents_before", "documents_added")):
            problems.append(f"provenance change {sorted(ids)} records no record-level diff")
        linked |= ids
    if linked != issue_ids:
        problems.append(f"issues without a recorded change: {sorted(issue_ids - linked)}")

    later_removed: set[str] = set()
    public_rows = {r["row_id"] for r in data["ledger"]}
    for c in reversed(changes):
        for row_id in c.get("ledger_added", []):
            if row_id not in public_rows and row_id not in later_removed:
                problems.append(f"added row {row_id} is not in the public ledger")
        later_removed |= {r["row_id"] for r in c.get("ledger_removed", [])}

    before = reconstruct_pre_error(data, changes)
    balance: defaultdict[str, int] = defaultdict(int)
    for r in before["ledger"]:
        balance[r["currency"]] += r["debit_minor"] - r["credit_minor"]
    if any(balance.values()):
        problems.append(f"pre-error ledger unbalanced: {dict(balance)}")
    for name, finder in (("duplicates", find_duplicates), ("unallocated receipts", find_unallocated),
                         ("fx mismatches", find_fx_mismatches), ("cut-off", find_cutoff)):
        found = finder(before)
        if found:
            problems.append(f"pre-error state already has {name}: {found}")
    credits: defaultdict[str, int] = defaultdict(int)
    debits: defaultdict[str, int] = defaultdict(int)
    for r in before["ledger"]:
        if r["account"] == "1100" and r["invoice_id"]:
            debits[r["invoice_id"]] += r["debit_minor"]
            credits[r["invoice_id"]] += r["credit_minor"]
    for inv_id, credit in credits.items():
        if credit > debits[inv_id]:
            problems.append(f"pre-error receivable of {inv_id} over-credited")
    return problems


@decimal_context
def validate_episode(episode_dir: str | Path) -> list[str]:
    episode_dir = Path(episode_dir)
    problems: list[str] = []
    for name in PUBLIC_FILES:
        if not (episode_dir / "public" / name).exists():
            problems.append(f"missing public/{name}")
    if problems:
        return problems
    data = load_public(episode_dir)
    truth = read_json(episode_dir / "private" / "truth.json")
    m = data["manifest"]

    if truth["episode_id"] != m["episode_id"]:
        problems.append("truth/manifest episode_id mismatch")
    counts = m["record_counts"]
    for table in ("ledger", "invoices", "payments", "fx"):
        if counts[table] != len(data[table]):
            problems.append(f"manifest count mismatch for {table}")

    balance: defaultdict[str, int] = defaultdict(int)
    for r in data["ledger"]:
        balance[r["currency"]] += r["debit_minor"] - r["credit_minor"]
        if (r["debit_minor"] == 0) == (r["credit_minor"] == 0):
            problems.append(f"row {r['row_id']} must have exactly one of debit/credit")
    for cur, diff in balance.items():
        if diff != 0:
            problems.append(f"ledger unbalanced in {cur} by {diff}")

    fx_keys = [f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in data["fx"]]
    if len(fx_keys) != len(set(fx_keys)):
        problems.append("fx.csv has repeated date:source:target keys (fx record ids must be unique)")
    for p in data["payments"]:
        refs = p["invoice_refs"]
        if refs is not None and sum(ref["amount_minor"] for ref in refs) != p["amount_minor"]:
            problems.append(f"payment {p['payment_id']} refs do not sum to amount")

    public_ids = ({r["row_id"] for r in data["ledger"]} | {i["doc_id"] for i in data["invoices"]}
                  | {p["doc_id"] for p in data["payments"]}
                  | {f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in data["fx"]})
    detected = {"duplicate_invoice": find_duplicates(data), "partial_payment": find_unallocated(data),
                "fx_conversion": find_fx_mismatches(data), "period_cutoff": find_cutoff(data)}
    start = datetime.fromisoformat(m["period_start"])
    end = datetime.fromisoformat(m["period_end"])
    for r in data["ledger"]:
        if not start <= datetime.fromisoformat(r["timestamp"]) <= end:
            problems.append(f"ledger row {r['row_id']} posted outside the period")
    base = m["base_currency"]
    payments = {p["payment_id"]: p for p in data["payments"]}

    declared: dict[str, dict] = {t: {} for t in detected}
    for issue in truth["issues"]:
        # affected_ids is a SET (matcher rule): order inside the truth list is irrelevant, so a jury
        # copy whose truth lists were not re-sorted after renaming validates the same.
        key = tuple(sorted(set(issue["affected_ids"])))
        declared.setdefault(issue["type"], {})[key] = issue["expected_amount_minor"]
        if len(key) != len(issue["affected_ids"]):
            problems.append(f"{issue['issue_id']}: affected_ids repeat an id")
        if not issue["evidence_sets"]:
            problems.append(f"{issue['issue_id']}: no evidence set")
        for ev in issue["evidence_sets"]:
            missing = [e for e in ev if e not in public_ids]
            if missing:
                problems.append(f"{issue['issue_id']}: evidence not in public records {missing}")
            elif (reason := verify_evidence(issue, ev, data)) is not None:
                problems.append(f"{issue['issue_id']}: evidence set insufficient: {reason}")
        if issue["type"] != "duplicate_invoice" and issue["currency"] != base:
            problems.append(f"{issue['issue_id']}: amount must be in base currency")
        if issue["type"] == "partial_payment":
            pay = payments.get(key[0])
            if pay is None or not pay["invoice_refs"]:
                problems.append(f"{issue['issue_id']}: payment has no remittance reference")

    for error_type, found in detected.items():
        if found != declared.get(error_type, {}):
            problems.append(f"{error_type}: records show {found}, truth declares {declared.get(error_type)}")

    provenance_path = episode_dir / "private" / "provenance.json"
    if provenance_path.exists():
        problems += check_pre_error_state(data, truth, read_json(provenance_path))
    else:
        problems.append("missing private/provenance.json")
    return problems

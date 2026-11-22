"""Second, fully independent reconciler of one episode (hard-jury audit, item 2).

It is written from the published rules only (policy.json, README "Правила сверки") and imports
NOTHING from auditenv: no generator, no validator, no grader (a test checks the imports). It does
what a jury member who does not trust the team's validator would do:

  1. read public/ and private/{truth,provenance}.json;
  2. check the books as published: one positive side per row, base currency, every row in the
     period, the ledger balances in total and per transaction;
  3. roll back every recorded change (newest first) to the books BEFORE any error;
  4. re-post every document of that pre-error state by the published posting rules (FX at the
     rate of the document's local date in the manifest timezone, half-even to the minor unit,
     line rounding residue on the last line, the completing payment settles the receivable at its
     booked amount, the FX difference on 7900) and require the pre-error books to match exactly,
     balance, and contain no discrepancy;
  5. for every truth issue, derive the expected amount from the CHANGE (pre-error vs published
     rows) and from the published rules, and compare it with the truth amount, currency, affected
     ids and the minimal evidence set;
  6. scan the published books for discrepancies with its own rules and require the scan to equal
     the truth (nothing hidden, nothing invented).

    python -m auditenv.independent EPISODE_DIR [EPISODE_DIR ...]      # JSON report, exit 1 on problems

Decimal arithmetic runs at 60 significant digits, so a precision loss in the generator would show
up as a mismatch here.
"""

from __future__ import annotations

import csv
import json
import re
from typing import Any
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from pathlib import Path

CTX = Context(prec=60, rounding=ROUND_HALF_EVEN)
TYPES = ("duplicate_invoice", "partial_payment", "fx_conversion", "period_cutoff")
AR, CASH, SUSPENSE, REVENUE, FX_DIFF = "1100", "1000", "2900", "4000", "7900"
EXPENSES = ("6100", "6200", "6300", "6400")


# ------------------------------------------------------------------------------------ loading
def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(episode_dir) -> dict:
    root = Path(episode_dir)
    pub, priv = root / "public", root / "private"
    with open(pub / "ledger.csv", encoding="utf-8", newline="") as f:
        ledger = []
        for raw in csv.DictReader(f):
            row: dict[str, Any] = {k: (v if v != "" else None) for k, v in raw.items()}
            row["debit_minor"], row["credit_minor"] = int(row["debit_minor"]), int(row["credit_minor"])
            ledger.append(row)
    with open(pub / "fx.csv", encoding="utf-8", newline="") as f:
        fx = list(csv.DictReader(f))
    with open(pub / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(pub / "policy.json", encoding="utf-8") as f:
        policy = json.load(f)
    data = {"manifest": manifest, "policy": policy, "ledger": ledger, "fx": fx,
            "invoices": _jsonl(pub / "invoices.jsonl"), "payments": _jsonl(pub / "payments.jsonl")}
    with open(priv / "truth.json", encoding="utf-8") as f:
        data["truth"] = json.load(f)
    prov = priv / "provenance.json"
    data["provenance"] = json.loads(prov.read_text(encoding="utf-8")) if prov.exists() else None
    return data


# ------------------------------------------------------------------------------------- rules
def manifest_tz(manifest: dict) -> timezone:
    """'UTC+05:45' / 'UTC-03:30' / 'UTC+00:00' -> tzinfo."""
    m = re.fullmatch(r"UTC([+-])(\d{2}):(\d{2})", manifest["timezone"])
    if not m:
        raise ValueError(f"unparseable manifest timezone {manifest['timezone']!r}")
    sign = 1 if m.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))


class Books:
    """Published rules over one set of records (published or pre-error)."""

    def __init__(self, data: dict, ledger: list[dict], invoices: list[dict], payments: list[dict]):
        self.inv: Any  # JSON records by id: dynamic on purpose
        self.pay: Any
        m = data["manifest"]
        self.base, self.exp, self.tz = m["base_currency"], m["currency_exponents"], manifest_tz(m)
        self.start, self.end = datetime.fromisoformat(m["period_start"]), datetime.fromisoformat(m["period_end"])
        self.rates = {(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"]) for r in data["fx"]}
        self.ledger, self.invoices, self.payments = ledger, invoices, payments
        self.inv = {i["invoice_id"]: i for i in invoices}
        self.pay = {p["payment_id"]: p for p in payments}

    def local_date(self, ts: str) -> str:
        """The date part of the timestamp in the manifest timezone (never the UTC date)."""
        return datetime.fromisoformat(ts).astimezone(self.tz).date().isoformat()

    def in_period(self, ts: str) -> bool:
        return self.start <= datetime.fromisoformat(ts) <= self.end

    def fx_key(self, date: str, currency: str) -> str:
        return f"{date}:{currency}:{self.base}"

    def conv(self, amount: int, currency: str, date: str, rate_currency: str | None = None) -> int | None:
        if currency == self.base and rate_currency is None:
            return amount
        rate = self.rates.get((date, rate_currency or currency, self.base))
        if rate is None:
            return None
        with localcontext(CTX):
            value = Decimal(amount) * rate * Decimal(10) ** (self.exp[self.base] - self.exp[currency])
            return int(value.quantize(Decimal(1), ROUND_HALF_EVEN))

    # rows of one document
    def invoice_rows(self, invoice_id: str) -> list[dict]:
        return [r for r in self.ledger if r["invoice_id"] == invoice_id and r["payment_id"] is None]

    def payment_rows(self, payment_id: str) -> list[dict]:
        return [r for r in self.ledger if r["payment_id"] == payment_id]

    def ar_debit(self, invoice_id: str) -> int:
        return sum(r["debit_minor"] for r in self.invoice_rows(invoice_id) if r["account"] == AR)

    def cash_debit(self, payment_id: str) -> int:
        return sum(r["debit_minor"] for r in self.payment_rows(payment_id) if r["account"] == CASH)

    def suspense_credit(self, payment_id: str) -> int:
        return sum(r["credit_minor"] - r["debit_minor"] for r in self.payment_rows(payment_id) if r["account"] == SUSPENSE)

    # discrepancy scan with this module's own rules
    def scan(self) -> dict[tuple, int]:
        """(type, sorted affected ids) -> amount for every discrepancy the published rules reveal."""
        found: dict[tuple, int] = {}
        groups = defaultdict(list)
        for inv in self.invoices:
            key = (inv["entity_id"], inv["issued_at"], inv["currency"], inv["total_minor"],
                   json.dumps(inv["line_items"], sort_keys=True))
            groups[key].append(inv)
        for g in groups.values():
            if len(g) > 1:
                found[("duplicate_invoice", tuple(sorted(i["invoice_id"] for i in g)))] = g[0]["total_minor"]
        for inv in self.invoices:
            booked = self.ar_debit(inv["invoice_id"])
            if not booked:
                continue
            if inv["currency"] != self.base:
                correct = self.conv(inv["total_minor"], inv["currency"], self.local_date(inv["issued_at"]))
                if correct is None or booked != correct:
                    found[("fx_conversion", (inv["invoice_id"],))] = abs(booked - (correct or 0))
            if not self.in_period(inv["issued_at"]):
                found[("period_cutoff", (inv["invoice_id"],))] = booked
        for pay in self.payments:
            if pay["amount_minor"] <= 0:
                continue
            if pay["currency"] != self.base:
                cash = self.cash_debit(pay["payment_id"])
                correct = self.conv(pay["amount_minor"], pay["currency"], self.local_date(pay["timestamp"]))
                if correct is None or cash != correct:
                    found[("fx_conversion", (pay["payment_id"],))] = abs(cash - (correct or 0))
            if pay["invoice_refs"] and self.suspense_credit(pay["payment_id"]) > 0:
                found[("partial_payment", (pay["payment_id"],))] = self.suspense_credit(pay["payment_id"])
        return found


def transaction_key(row: dict):
    if row["payment_id"]:
        return ("payment", row["payment_id"])
    if row["invoice_id"]:
        return ("invoice", row["invoice_id"])
    return ("expense", row["timestamp"], row["entity_id"])


def check_rows(books: Books, label: str) -> list[str]:
    """Format and double entry of a set of books."""
    problems = []
    total = 0
    per_tx: defaultdict[tuple, int] = defaultdict(int)
    for r in books.ledger:
        if r["debit_minor"] < 0 or r["credit_minor"] < 0 or (r["debit_minor"] > 0) == (r["credit_minor"] > 0):
            problems.append(f"{label}: row {r['row_id']} must have exactly one positive side")
        if r["currency"] != books.base:
            problems.append(f"{label}: row {r['row_id']} not in the base currency")
        if not books.in_period(r["timestamp"]):
            problems.append(f"{label}: row {r['row_id']} posted outside the period")
        total += r["debit_minor"] - r["credit_minor"]
        per_tx[transaction_key(r)] += r["debit_minor"] - r["credit_minor"]
    if total:
        problems.append(f"{label}: ledger unbalanced by {total}")
    bad = [k for k, v in per_tx.items() if v]
    if bad:
        problems.append(f"{label}: {len(bad)} transactions unbalanced, e.g. {bad[:3]}")
    return problems


def check_postings(books: Books, label: str) -> list[str]:
    """Re-post every document by the published rules and compare row by row (pre-error books)."""
    problems = []
    credited: defaultdict[str, int] = defaultdict(int)   # invoice -> AR credited so far (base)
    allocated: defaultdict[str, int] = defaultdict(int)  # invoice -> references so far (document currency)
    booked: dict[str, int] = {}
    for inv in books.invoices:
        iid = inv["invoice_id"]
        rows = books.invoice_rows(iid)
        if not books.in_period(inv["issued_at"]):
            problems.append(f"{label}: invoice {iid} dated outside the period")
        date = books.local_date(inv["issued_at"])
        total = books.conv(inv["total_minor"], inv["currency"], date)
        converted = [books.conv(li["amount_minor"], inv["currency"], date) for li in inv["line_items"]]
        lines = [x for x in converted if x is not None]
        if total is None or len(lines) != len(converted):
            problems.append(f"{label}: invoice {iid}: no fx rate for {books.fx_key(date, inv['currency'])}")
            continue
        lines[-1] += total - sum(lines)
        ar = [r["debit_minor"] for r in rows if r["account"] == AR]
        rev = [r["credit_minor"] for r in rows if r["account"] == REVENUE]
        if ar != [total]:
            problems.append(f"{label}: invoice {iid}: AR debit {ar} != {total} (rate of {date})")
        if sorted(rev) != sorted(lines):
            problems.append(f"{label}: invoice {iid}: revenue lines {sorted(rev)} != {sorted(lines)}")
        if any(datetime.fromisoformat(r["timestamp"]) < datetime.fromisoformat(inv["issued_at"]) for r in rows):
            problems.append(f"{label}: invoice {iid} booked before its document date")
        booked[iid] = total
    if sum(li["amount_minor"] for i in books.invoices for li in i["line_items"]) != sum(i["total_minor"] for i in books.invoices):
        problems.append(f"{label}: an invoice total differs from the sum of its lines")

    for pay in sorted(books.payments, key=lambda p: (p["timestamp"], p["payment_id"])):
        pid, amount, cur = pay["payment_id"], pay["amount_minor"], pay["currency"]
        rows = books.payment_rows(pid)
        date = books.local_date(pay["timestamp"])
        if amount < 0:  # refund: debit 4000, credit 1000, receivable untouched
            value = books.conv(-amount, cur, date)
            refund_rows = sorted((r["account"], r["debit_minor"], r["credit_minor"]) for r in rows)
            if refund_rows != sorted([(REVENUE, value, 0), (CASH, 0, value)]):
                problems.append(f"{label}: refund {pid} rows {refund_rows}")
            continue
        cash = books.conv(amount, cur, date)
        if cash is None:
            problems.append(f"{label}: payment {pid}: no fx rate for {books.fx_key(date, cur)}")
            continue
        expect: defaultdict[tuple, int] = defaultdict(int)
        expect[(CASH, "d")] += cash
        refs = pay["invoice_refs"] or []
        ar_total = 0
        for ref in refs:
            iid, part = ref["invoice_id"], ref["amount_minor"]
            inv = books.inv.get(iid)
            if inv is None or iid not in booked:
                problems.append(f"{label}: payment {pid} references unknown invoice {iid}")
                continue
            if allocated[iid] + part == inv["total_minor"]:
                credit = booked[iid] - credited[iid]          # the completing payment settles exactly
            else:
                with localcontext(CTX):
                    credit = int((Decimal(booked[iid]) * part / inv["total_minor"]).quantize(Decimal(1), ROUND_HALF_EVEN))
            allocated[iid] += part
            credited[iid] += credit
            ar_total += credit
            expect[(AR, "c", iid)] += credit
        rest = amount - sum(r["amount_minor"] for r in refs)
        suspense = (books.conv(rest, cur, date) or 0) if rest > 0 else 0
        if suspense:
            expect[(SUSPENSE, "c")] += suspense
        diff = cash - ar_total - suspense
        if diff > 0:
            expect[(FX_DIFF, "c")] += diff
        elif diff < 0:
            expect[(FX_DIFF, "d")] += -diff
        got: defaultdict[tuple, int] = defaultdict(int)
        for r in rows:
            side = "d" if r["debit_minor"] else "c"
            key = (r["account"], side, r["invoice_id"]) if r["account"] == AR else (r["account"], side)
            got[key] += r["debit_minor"] or r["credit_minor"]
        if dict(got) != dict(expect):
            problems.append(f"{label}: payment {pid}: rows {dict(got)} != rules {dict(expect)}")
        if cur == books.base and ((FX_DIFF, "c") in got or (FX_DIFF, "d") in got):
            problems.append(f"{label}: base-currency payment {pid} has an FX difference")
    for iid, inv in books.inv.items():
        if allocated[iid] > inv["total_minor"]:
            problems.append(f"{label}: invoice {iid} referenced above its total")
        if allocated[iid] == inv["total_minor"] and credited[iid] != booked.get(iid):
            problems.append(f"{label}: fully paid invoice {iid} not settled at its booked amount")
    for r in books.ledger:
        if r["account"] in EXPENSES and (r["invoice_id"] or r["payment_id"] or not r["debit_minor"]):
            problems.append(f"{label}: expense row {r['row_id']} malformed")
    return problems


# ----------------------------------------------------------------------------- roll-back
def pre_error(data: dict) -> tuple[list, list, list, list[str]]:
    """Undo every recorded change, newest first (README: provenance diff)."""
    problems = []
    ledger = {r["row_id"]: dict(r) for r in data["ledger"]}
    docs = {d["doc_id"]: dict(d) for d in data["invoices"] + data["payments"]}
    changes = [c for c in (data["provenance"] or {}).get("changes", []) if c.get("op") != "clean_episode"]
    for c in reversed(changes):
        for row_id in c.get("ledger_added", []):
            if ledger.pop(row_id, None) is None:
                problems.append(f"rollback: added row {row_id} missing")
        for row in c.get("ledger_removed", []):
            ledger[row["row_id"]] = dict(row)
        for doc_id in c.get("documents_added", []):
            if docs.pop(doc_id, None) is None:
                problems.append(f"rollback: added document {doc_id} missing")
        for doc in c.get("documents_before", []):
            docs[doc["doc_id"]] = dict(doc)
    invoices = [d for d in docs.values() if "line_items" in d]
    payments = [d for d in docs.values() if "line_items" not in d]
    return list(ledger.values()), invoices, payments, problems


def _change_of(changes: list[dict], issue_id: str) -> dict | None:
    for c in changes:
        if c.get("issue_id") == issue_id or issue_id in c.get("issue_ids", []):
            return c
    return None


def check_issues(data: dict, post: Books, pre: Books) -> list[str]:
    """Every truth issue follows from its recorded change and the published rules."""
    problems = []
    changes = [c for c in (data["provenance"] or {}).get("changes", []) if c.get("op") != "clean_episode"]
    public_ids = ({r["row_id"] for r in post.ledger} | {d["doc_id"] for d in post.invoices + post.payments}
                  | {f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in data["fx"]})
    for t in data["truth"]["issues"]:
        iid, kind, ids = t["issue_id"], t["type"], sorted(set(t["affected_ids"]))
        label = f"{iid} ({kind})"
        change = _change_of(changes, iid)
        if change is None:
            problems.append(f"{label}: no recorded change")
            continue
        if not any(change.get(k) for k in ("ledger_added", "ledger_removed", "documents_added", "documents_before")):
            problems.append(f"{label}: change has no record-level diff")
        for ev in t["evidence_sets"]:
            if not set(ev) <= public_ids:
                problems.append(f"{label}: evidence {sorted(set(ev) - public_ids)} not in public data")
        evidence = {frozenset(ev) for ev in t["evidence_sets"]}
        if kind != "duplicate_invoice" and t["currency"] != post.base:
            problems.append(f"{label}: amount must be in the base currency")

        if kind == "duplicate_invoice":
            invs = [post.inv.get(i) for i in ids]
            if len(ids) != 2 or None in invs:
                problems.append(f"{label}: needs two published invoices")
                continue
            a, b = invs
            same = all(a[k] == b[k] for k in ("entity_id", "issued_at", "currency", "total_minor", "line_items"))
            added = set(change.get("documents_added", []))
            if not same or len(added & {a["doc_id"], b["doc_id"]}) != 1:
                problems.append(f"{label}: not two identical invoices, exactly one of them added by the change")
            if sum(1 for i in ids if i in pre.inv) != 1:
                problems.append(f"{label}: the pre-error books must hold exactly one of the two invoices")
            if (t["expected_amount_minor"], t["currency"]) != (a["total_minor"], a["currency"]):
                problems.append(f"{label}: amount/currency != invoice total in its currency")
            if frozenset({a["doc_id"], b["doc_id"]}) not in evidence:
                problems.append(f"{label}: minimal evidence (both doc_ids) missing")
            if post.ar_debit(a["invoice_id"]) == 0 or post.ar_debit(b["invoice_id"]) == 0:
                problems.append(f"{label}: both invoices must be booked")

        elif kind == "partial_payment":
            pay = post.pay.get(ids[0])
            if len(ids) != 1 or pay is None or not pay["invoice_refs"] or pay["amount_minor"] <= 0:
                problems.append(f"{label}: needs one receipt that names an invoice")
                continue
            got = post.suspense_credit(pay["payment_id"])
            if pre.suspense_credit(pay["payment_id"]) != 0:
                problems.append(f"{label}: suspense credit already before the error")
            if got <= 0 or got != t["expected_amount_minor"]:
                problems.append(f"{label}: suspense credit {got} != expected {t['expected_amount_minor']}")
            ar_pre = sum(r["credit_minor"] for r in pre.payment_rows(pay["payment_id"]) if r["account"] == AR)
            ar_post = sum(r["credit_minor"] for r in post.payment_rows(pay["payment_id"]) if r["account"] == AR)
            if ar_post >= ar_pre:
                problems.append(f"{label}: the change did not reduce the allocation ({ar_pre} -> {ar_post})")
            if pay["currency"] == post.base and ar_pre - ar_post != got:
                problems.append(f"{label}: moved from receivable {ar_pre - ar_post} != suspense {got}")
            rows_2900 = [r["row_id"] for r in post.payment_rows(pay["payment_id"]) if r["account"] == SUSPENSE]
            if frozenset({pay["doc_id"], *rows_2900}) not in evidence:
                problems.append(f"{label}: minimal evidence (payment doc_id + 2900 row) missing")

        elif kind == "fx_conversion":
            doc_id = ids[0] if len(ids) == 1 else None
            inv, pay = post.inv.get(doc_id), post.pay.get(doc_id)
            if inv is None and pay is None:
                problems.append(f"{label}: needs one published invoice or payment")
                continue
            if inv is not None:
                amount, cur, date = inv["total_minor"], inv["currency"], post.local_date(inv["issued_at"])
                booked_rows = [r for r in post.invoice_rows(inv["invoice_id"]) if r["account"] == AR]
                pre_booked = pre.ar_debit(inv["invoice_id"]) if inv["invoice_id"] in pre.inv else None
                pre_doc = pre.inv.get(inv["invoice_id"])
                doc = inv
            else:
                amount, cur, date = pay["amount_minor"], pay["currency"], post.local_date(pay["timestamp"])
                booked_rows = [r for r in post.payment_rows(pay["payment_id"]) if r["account"] == CASH]
                pre_booked = pre.cash_debit(pay["payment_id"]) if pay["payment_id"] in pre.pay else None
                pre_doc = pre.pay.get(pay["payment_id"])
                doc = pay
            if cur == post.base:
                problems.append(f"{label}: a base-currency document cannot have an fx error")
                continue
            correct = post.conv(amount, cur, date)
            booked = sum(r["debit_minor"] for r in booked_rows)
            if correct is None or abs(booked - correct) != t["expected_amount_minor"] or booked == correct:
                problems.append(f"{label}: |booked {booked} - correct {correct}| != expected {t['expected_amount_minor']}")
            # The wrong figure is explained by a rate that exists in fx.csv (another date or pair).
            explained = any(post.conv(amount, cur, d, c) == booked for (d, c, tgt) in post.rates if tgt == post.base)
            if not explained:
                problems.append(f"{label}: booked {booked} matches no published rate")
            # Before the error the document was converted correctly (at its own local date then).
            if pre_doc is not None and pre_booked is not None:
                pre_date = pre.local_date(pre_doc["issued_at"] if inv is not None else pre_doc["timestamp"])
                if pre_booked != pre.conv(amount, cur, pre_date):
                    problems.append(f"{label}: wrong conversion already before the error")
            minimal = frozenset({doc["doc_id"], *[r["row_id"] for r in booked_rows], post.fx_key(date, cur)})
            if minimal not in evidence:
                problems.append(f"{label}: minimal evidence (doc_id, booked row, fx key of the local date) missing")

        elif kind == "period_cutoff":
            inv = post.inv.get(ids[0]) if len(ids) == 1 else None
            if inv is None:
                problems.append(f"{label}: needs one published invoice")
                continue
            ar_rows = [r for r in post.invoice_rows(inv["invoice_id"]) if r["account"] == AR]
            if post.in_period(inv["issued_at"]):
                problems.append(f"{label}: the invoice is dated inside the period")
            if not ar_rows or not all(post.in_period(r["timestamp"]) for r in ar_rows):
                problems.append(f"{label}: the receivable is not booked in this period")
            if sum(r["debit_minor"] for r in ar_rows) != t["expected_amount_minor"]:
                problems.append(f"{label}: receivable booked != expected {t['expected_amount_minor']}")
            pre_doc = pre.inv.get(inv["invoice_id"])
            if pre_doc is None or not pre.in_period(pre_doc["issued_at"]):
                problems.append(f"{label}: before the error the invoice must be dated inside the period")
            if frozenset({inv["doc_id"], *[r["row_id"] for r in ar_rows]}) not in evidence:
                problems.append(f"{label}: minimal evidence (doc_id + 1100 row) missing")
        else:
            problems.append(f"{label}: unknown type")
    return problems


# -------------------------------------------------------------------------------- driver
def reconcile(episode_dir) -> dict:
    data = load(episode_dir)
    m = data["manifest"]
    problems: list[str] = []
    tz = manifest_tz(m)
    offset = tz.utcoffset(None)
    stamps = ([r["timestamp"] for r in data["ledger"]] + [i["issued_at"] for i in data["invoices"]]
              + [p["timestamp"] for p in data["payments"]] + [m["period_start"], m["period_end"]])
    foreign_offsets = sorted({s[19:] for s in stamps if datetime.fromisoformat(s).utcoffset() != offset})
    if foreign_offsets:  # not an error by itself (local date is computed), but worth knowing
        problems.append(f"timestamps not written in the manifest timezone: {foreign_offsets[:3]}")
    for table in ("ledger", "invoices", "payments", "fx"):
        if m["record_counts"][table] != len(data[table]):
            problems.append(f"manifest record_counts[{table}] != file")
    keys = [f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in data["fx"]]
    if len(keys) != len(set(keys)):
        problems.append("fx keys DATE:SRC:TGT are not unique")
    for p in data["payments"]:
        if p["invoice_refs"] is not None and sum(r["amount_minor"] for r in p["invoice_refs"]) > p["amount_minor"] >= 0:
            problems.append(f"payment {p['payment_id']}: references exceed the amount")

    post = Books(data, data["ledger"], data["invoices"], data["payments"])
    problems += check_rows(post, "published")
    ledger, invoices, payments, rb = pre_error(data)
    problems += rb
    pre = Books(data, ledger, invoices, payments)
    problems += check_rows(pre, "pre-error")
    problems += check_postings(pre, "pre-error")
    leftover = pre.scan()
    if leftover:
        problems.append(f"pre-error books already contain discrepancies: {sorted(leftover)[:3]}")
    problems += check_issues(data, post, pre)

    declared = {(t["type"], tuple(sorted(set(t["affected_ids"])))): t["expected_amount_minor"]
                for t in data["truth"]["issues"]}
    found = post.scan()
    for key in sorted(set(found) | set(declared)):
        if found.get(key) != declared.get(key):
            problems.append(f"scan vs truth for {key}: records {found.get(key)}, truth {declared.get(key)}")
    if len(declared) != len(data["truth"]["issues"]):
        problems.append("two truth issues with the same type and affected ids")
    return {"episode": str(episode_dir), "issues": len(data["truth"]["issues"]),
            "types": sorted(t["type"] for t in data["truth"]["issues"]),
            "base_currency": m["base_currency"], "timezone": m["timezone"],
            "ledger_rows": len(data["ledger"]), "problems": problems}


def main(argv: list[str]) -> int:
    reports = [reconcile(p) for p in argv]
    print(json.dumps(reports, indent=1))
    return 1 if any(r["problems"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

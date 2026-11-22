"""Step 1 of generation: a consistent hidden history of one company for one period.

Nothing here is wrong yet. Every invoice and payment is posted with double entry in
the base currency (foreign documents are converted at the rate of their local date),
so the books balance and every receivable equals what was booked minus allocations.

Postings are repostable: an error is introduced later by un-posting a document and
posting it again with one specific mistake (wrong rate date, wrong allocation, wrong
period). That keeps every injected error causal, double-entry and recorded.
"""

from __future__ import annotations

import calendar
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal

from auditenv.common import CURRENCY_EXPONENT, IdFactory, decimal_context

NAME_A = ["Altyn", "Nomad", "Steppe", "Kazyna", "Samruk", "Aral", "Irtysh", "Tengri", "Baikonur",
          "Ulytau", "Zhetysu", "Kaspi", "Orda", "Saryarka", "Qyran", "Atameken", "Tulpar", "Shanyrak"]
NAME_B = ["Trading", "Systems", "Logistics", "Foods", "Energy", "Digital", "Build", "Pharma", "Textile",
          "Motors", "Analytics", "Agro", "Media", "Retail"]
NAME_C = ["LLP", "JSC", "Group", "Partners"]

CATALOG = [  # (description, unit price range in KZT major units, quantity range)
    ("Consulting hours", (15_000, 45_000), (1, 40)),
    ("Cloud hosting plan", (40_000, 250_000), (1, 3)),
    ("Hardware lease", (90_000, 600_000), (1, 5)),
    ("Training seat", (25_000, 80_000), (1, 25)),
    ("Data integration package", (300_000, 1_500_000), (1, 1)),
    ("Support incident", (8_000, 30_000), (1, 15)),
]
EXPENSE_ACCOUNTS = ["6100", "6200", "6300", "6400"]
# Reference value of one major unit in KZT. Rates between any two currencies are derived from
# these (with noise), so the base currency can be any of them. JPY has 0 decimals (exponent 0).
REF_KZT = {"KZT": Decimal("1"), "USD": Decimal("500"), "EUR": Decimal("540"), "JPY": Decimal("3.3")}
RATE_QUANT = Decimal("0.00000001")


def with_cents(amount_minor: int, currency: str, rng: random.Random, share: float = 0.5) -> int:
    """Half of all amounts carry random minor units, in legit records and errors alike, so the
    granularity of an amount (whole vs. fractional) never hints at an error."""
    exp = CURRENCY_EXPONENT[currency]
    if exp and rng.random() < share:
        return amount_minor + rng.randint(1, 10 ** exp - 1)
    return amount_minor


def money(kzt_major: int, currency: str) -> int:
    """A KZT-scale major amount expressed as whole major units of `currency`, in minor units."""
    major = max(1, int((Decimal(kzt_major) / REF_KZT[currency]).quantize(Decimal(1), ROUND_HALF_EVEN)))
    return major * 10 ** CURRENCY_EXPONENT[currency]
FX_MARGIN_DAYS = 7


@dataclass
class World:
    seed: int
    base_currency: str
    tz: timezone
    period_start: datetime
    period_end: datetime
    entities: dict[str, dict] = field(default_factory=dict)
    invoices: list[dict] = field(default_factory=list)
    payments: list[dict] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)
    foreign: list[str] = field(default_factory=list)
    fx: list[dict] = field(default_factory=list)
    fx_index: dict[tuple[str, str], Decimal] = field(default_factory=dict)
    # Generator-only bookkeeping, never exported to public files.
    inv_meta: dict[str, dict] = field(default_factory=dict)
    pay_meta: dict[str, dict] = field(default_factory=dict)
    truth: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    attempt: int = 0  # generation attempt that produced this world (provenance: generation_attempt)

    def ts(self, dt: datetime) -> str:
        return dt.astimezone(self.tz).isoformat(timespec="seconds")

    def invoice(self, invoice_id: str) -> dict:
        return next(i for i in self.invoices if i["invoice_id"] == invoice_id)

    def payment(self, payment_id: str) -> dict:
        return next(p for p in self.payments if p["payment_id"] == payment_id)

    def row(self, row_id: str) -> dict:
        return next(r for r in self.ledger if r["row_id"] == row_id)

    @property
    def last_day(self) -> str:
        return self.period_end.date().isoformat()


# ----------------------------------------------------------------- conversion
@decimal_context
def convert(world: World, amount_minor: int, currency: str, date: str, rate_currency: str | None = None) -> int:
    """Document amount -> base-currency minor units at the rate of `date`, half-even."""
    if currency == world.base_currency:
        return amount_minor
    rate = world.fx_index[(date, rate_currency or currency)]
    scale = Decimal(10) ** (CURRENCY_EXPONENT[world.base_currency] - CURRENCY_EXPONENT[currency])
    return int((Decimal(amount_minor) * rate * scale).quantize(Decimal(1), ROUND_HALF_EVEN))


def shift_date(date: str, days: int) -> str:
    return (datetime.fromisoformat(date) + timedelta(days=days)).date().isoformat()


# -------------------------------------------------------------------- posting
def _row(ids: IdFactory, ts: str, account: str, debit: int, credit: int, currency: str,
         entity_id=None, invoice_id=None, payment_id=None) -> dict:
    assert (debit > 0) != (credit > 0), (account, debit, credit)
    return {
        "row_id": ids.new("L-", 7), "timestamp": ts, "account": account,
        "debit_minor": debit, "credit_minor": credit, "currency": currency,
        "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id,
    }


def post_invoice(world: World, ids: IdFactory, inv: dict, posting_ts: str | None = None,
                 rate_date: str | None = None, rate_currency: str | None = None) -> list[str]:
    """AR debit for the booked total, revenue credit per line. Returns row ids (AR row first).

    The posting timestamp defaults to the one used last time (legit posting delays survive
    a repost), else to the document date."""
    meta = world.inv_meta[inv["invoice_id"]]
    ts, base = posting_ts or meta.get("posting_ts") or inv["issued_at"], world.base_currency
    rate_date = rate_date or inv["issued_at"][:10]
    booked = convert(world, inv["total_minor"], inv["currency"], rate_date, rate_currency)
    lines = [convert(world, l["amount_minor"], inv["currency"], rate_date, rate_currency) for l in inv["line_items"]]
    lines[-1] += booked - sum(lines)  # rounding residue goes to the last line
    assert lines[-1] > 0
    rows = [_row(ids, ts, "1100", booked, 0, base, inv["entity_id"], inv["invoice_id"])]
    rows += [_row(ids, ts, "4000", 0, amount, base, inv["entity_id"], inv["invoice_id"]) for amount in lines]
    world.ledger.extend(rows)
    meta.update({"rows": [r["row_id"] for r in rows], "booked": booked, "posting_ts": ts})
    return meta["rows"]


def post_payment(world: World, ids: IdFactory, pay: dict, alloc: dict[str, int] | None = None,
                 rate_date: str | None = None) -> dict:
    """Cash debit; AR credit per allocated invoice; rest to suspense 2900; FX difference to 7900.

    `alloc` (document currency) defaults to the remittance advice in invoice_refs.
    """
    if alloc is None:
        alloc = {ref["invoice_id"]: ref["amount_minor"] for ref in (pay["invoice_refs"] or [])}
    ts, cur, base = pay["timestamp"], pay["currency"], world.base_currency
    rate_date = rate_date or ts[:10]
    cash = convert(world, pay["amount_minor"], cur, rate_date)
    rows = [_row(ids, ts, "1000", cash, 0, base, pay["entity_id"], None, pay["payment_id"])]
    credits: dict[str, int] = {}
    for invoice_id, amount in alloc.items():
        inv, meta = world.invoice(invoice_id), world.inv_meta[invoice_id]
        if meta["alloc"] + amount == inv["total_minor"]:
            credit = meta["booked"] - meta["credited"]  # settles the receivable exactly
        else:
            credit = int((Decimal(meta["booked"]) * amount / inv["total_minor"]).quantize(Decimal(1), ROUND_HALF_EVEN))
        meta["alloc"] += amount
        meta["credited"] += credit
        credits[invoice_id] = credit
        rows.append(_row(ids, ts, "1100", 0, credit, base, pay["entity_id"], invoice_id, pay["payment_id"]))
    unallocated = pay["amount_minor"] - sum(alloc.values())
    suspense_rows = []
    if unallocated > 0:
        amount = convert(world, unallocated, cur, rate_date)
        suspense_rows.append(_row(ids, ts, "2900", 0, amount, base, pay["entity_id"], None, pay["payment_id"]))
    rows += suspense_rows
    diff = cash - sum(credits.values()) - sum(r["credit_minor"] for r in suspense_rows)
    if diff:
        rows.append(_row(ids, ts, "7900", max(0, -diff), max(0, diff), base, pay["entity_id"], None, pay["payment_id"]))
    world.ledger.extend(rows)
    meta = {"rows": [r["row_id"] for r in rows], "cash_row": rows[0]["row_id"], "alloc": dict(alloc),
            "credits": credits, "suspense_rows": [r["row_id"] for r in suspense_rows], "cash": cash}
    world.pay_meta[pay["payment_id"]] = meta
    return meta


def _remove_rows(world: World, row_ids: list[str]) -> None:
    drop = set(row_ids)
    world.ledger[:] = [r for r in world.ledger if r["row_id"] not in drop]


def unpost_payment(world: World, payment_id: str) -> dict:
    meta = world.pay_meta.pop(payment_id)
    _remove_rows(world, meta["rows"])
    for invoice_id, credit in meta["credits"].items():
        world.inv_meta[invoice_id]["credited"] -= credit
        world.inv_meta[invoice_id]["alloc"] -= meta["alloc"][invoice_id]
    return meta


def unpost_invoice(world: World, invoice_id: str) -> None:
    meta = world.inv_meta[invoice_id]
    assert meta["credited"] == 0 and meta["alloc"] == 0, "unpost payments first"
    _remove_rows(world, meta["rows"])


# ------------------------------------------------------------------- building
def build_world(seed: int, cfg: dict, rng: random.Random, ids: IdFactory) -> World:
    sign_hours = rng.choice(cfg["timezones"])
    hours, minutes = int(sign_hours[:3]), int(sign_hours[0] + sign_hours[4:])
    tz = timezone(timedelta(hours=hours, minutes=minutes))
    year, month = 2026, rng.randint(1, 11)
    days_in_month = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, tzinfo=tz)
    end = datetime(year, month, days_in_month, 23, 59, 59, tzinfo=tz)
    world = World(seed=seed, base_currency=cfg["base_currency"], tz=tz, period_start=start, period_end=end)
    base = cfg["base_currency"]

    def rand_time(day_lo: int, day_hi: int) -> datetime:
        return start + timedelta(days=rng.randint(day_lo, day_hi), hours=rng.randint(9, 18),
                                 minutes=rng.randint(0, 59))

    # FX table first: daily rates for the period plus a margin on both sides.
    world.foreign = list(cfg["foreign_currencies"])
    for currency in world.foreign:
        ratio = REF_KZT[currency] / REF_KZT[base]
        rate = (ratio * Decimal(str(rng.uniform(0.95, 1.05)))).quantize(RATE_QUANT, ROUND_HALF_EVEN)
        for day in range(-FX_MARGIN_DAYS, days_in_month + FX_MARGIN_DAYS):
            date = (start + timedelta(days=day)).date().isoformat()
            rate = (rate * Decimal(str(1 + rng.uniform(-0.004, 0.004)))).quantize(RATE_QUANT, ROUND_HALF_EVEN)
            world.fx.append({"date": date, "source_currency": currency, "target_currency": base, "rate": str(rate)})
            world.fx_index[(date, currency)] = rate

    used_names: set[str] = set()

    def new_name() -> str:
        """Bounded: a few random draws, then a deterministic numbered variant (never loops forever)."""
        for _ in range(20):
            name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.choice(NAME_C)}"
            if name not in used_names:
                used_names.add(name)
                return name
        base_name, n = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.choice(NAME_C)}", 2
        while f"{base_name} {n}" in used_names:
            n += 1
        used_names.add(f"{base_name} {n}")
        return f"{base_name} {n}"

    customers = []
    for _ in range(rng.randint(*cfg["n_customers"])):
        cid = ids.new("CUS-", 4)
        world.entities[cid] = {"entity_id": cid, "name": new_name(), "kind": "customer"}
        customers.append(cid)
    vendors = []
    for _ in range(rng.randint(*cfg["n_vendors"])):
        vid = ids.new("VEN-", 4)
        world.entities[vid] = {"entity_id": vid, "name": new_name(), "kind": "vendor"}
        vendors.append(vid)

    def add_invoice(entity_id: str, issued: datetime, lines: list[dict], role: str, currency: str,
                    allow_delay: bool = True) -> dict:
        for n, line in enumerate(lines, 1):
            line["line_no"] = n
        inv = {
            "doc_id": ids.new("DOC-", 8), "invoice_id": ids.new("INV-", 6), "entity_id": entity_id,
            "issued_at": world.ts(issued), "due_at": world.ts(issued + timedelta(days=rng.choice([14, 30]))),
            "currency": currency, "total_minor": sum(l["amount_minor"] for l in lines), "line_items": lines,
        }
        world.invoices.append(inv)
        # Legit posting delay: booked a few days after the document date, still inside the period.
        posted = issued
        if allow_delay and rng.random() < cfg["posting_delay_share"]:
            delayed = issued + timedelta(days=rng.randint(1, 3), hours=rng.randint(0, 3))
            if delayed <= end:
                posted = delayed
        world.inv_meta[inv["invoice_id"]] = {"role": role, "outcome": None, "payments": [], "issued": issued,
                                             "posted": posted, "alloc": 0, "credited": 0}
        post_invoice(world, ids, inv, posting_ts=world.ts(posted))
        return inv

    def plain_lines(currency: str) -> list[dict]:
        order_ref = ids.new("ORD-", 6)
        lines = []
        for desc, (p_lo, p_hi), (q_lo, q_hi) in rng.sample(CATALOG, rng.randint(1, 3)):
            qty = rng.randint(q_lo, q_hi)
            unit = with_cents(money(rng.randint(p_lo, p_hi), currency), currency, rng)
            lines.append({"description": f"{desc} (order {order_ref})", "quantity": qty,
                          "unit_price_minor": unit, "amount_minor": qty * unit})
        return lines

    def pick_currency() -> str:
        return rng.choice(world.foreign) if rng.random() < cfg["foreign_invoice_share"] else base

    for _ in range(rng.randint(*cfg["n_plain_invoices"])):
        cur = pick_currency()
        add_invoice(rng.choice(customers), rand_time(0, days_in_month - 1), plain_lines(cur), "plain", cur)

    # Twin orders: two separate orders billed to the same customer at the same time for the same
    # amount, with different order references. Same look as a duplicate until the documents are read.
    plain_now = [i for i in world.invoices if world.inv_meta[i["invoice_id"]]["role"] == "plain"]
    for src in rng.sample(plain_now, min(len(plain_now), rng.randint(*cfg["n_twin_orders"]))):
        order_ref = ids.new("ORD-", 6)
        lines = [{**l, "description": l["description"].rsplit("(order ", 1)[0] + f"(order {order_ref})"}
                 for l in src["line_items"]]
        twin = add_invoice(src["entity_id"], world.inv_meta[src["invoice_id"]]["issued"], lines, "twin",
                           src["currency"], allow_delay=False)
        world.inv_meta[src["invoice_id"]]["role"] = "twin"
        assert twin["total_minor"] == src["total_minor"] and twin["issued_at"] == src["issued_at"]

    # Legit invoices issued and booked on the last day in business hours: the same look as a
    # cut-off error's posting, but the document date is inside the period (distractor).
    for _ in range(rng.randint(*cfg["n_last_day_invoices"])):
        cur = pick_currency()
        add_invoice(rng.choice(customers), rand_time(days_in_month - 1, days_in_month - 1),
                    plain_lines(cur), "late", cur, allow_delay=False)

    # Boundary invoices: legitimately inside the period, but near midnight on the first or
    # last day, so a naive UTC conversion puts them in the neighbouring period (distractor).
    for _ in range(rng.randint(*cfg["n_boundary_invoices"])):
        if rng.random() < 0.5:
            issued = end - timedelta(minutes=rng.randint(1, 50))
        else:
            issued = start + timedelta(minutes=rng.randint(1, 50))
        add_invoice(rng.choice(customers), issued, plain_lines(base), "boundary", base)

    # Recurring invoices: legitimately repeated amounts (distractor for duplicates).
    for cid in rng.sample(customers, min(len(customers), rng.randint(*cfg["n_recurring_customers"]))):
        unit = with_cents(money(rng.randint(50_000, 400_000), base), base, rng)
        first = rand_time(0, 6)
        for week in range(rng.randint(3, 4)):
            issued = first + timedelta(days=7 * week)
            if issued > end:
                break
            lines = [{"description": f"Managed support subscription, week {issued.isocalendar()[1]}",
                      "quantity": 1, "unit_price_minor": unit, "amount_minor": unit}]
            add_invoice(cid, issued, lines, "recurring", base)

    # Payment outcomes (split and partial payments are legitimate distractors).
    def new_payment(inv: dict, when: datetime, amount: int) -> dict:
        if inv["currency"] != base and rng.random() < cfg["foreign_night_payment_share"]:
            # Legit night receipt (foreign banks): converted at the local date's rate (distractor).
            night = when.replace(hour=rng.randint(0, 4), minute=rng.randint(0, 59))
            if night > world.inv_meta[inv["invoice_id"]]["posted"]:
                when = night
        pay = {
            "doc_id": ids.new("DOC-", 8), "payment_id": ids.new("PAY-", 6), "timestamp": world.ts(when),
            "currency": inv["currency"], "amount_minor": amount, "entity_id": inv["entity_id"],
            "invoice_refs": [{"invoice_id": inv["invoice_id"], "amount_minor": amount}],
        }
        world.payments.append(pay)
        post_payment(world, ids, pay)
        world.inv_meta[inv["invoice_id"]]["payments"].append(pay["payment_id"])
        return pay

    outcomes = list(cfg["invoice_outcomes"].items())
    for inv in list(world.invoices):
        meta = world.inv_meta[inv["invoice_id"]]
        outcome = rng.choices([o for o, _ in outcomes], weights=[w for _, w in outcomes])[0]
        if meta["role"] == "recurring" and outcome in ("split", "partial_legit"):
            outcome = "paid_full"
        pay_at = meta["posted"] + timedelta(days=rng.randint(1, 12), hours=rng.randint(0, 5))
        if outcome != "unpaid" and pay_at > end:
            outcome = "unpaid"
        meta["outcome"] = outcome
        total = inv["total_minor"]
        if outcome == "paid_full":
            new_payment(inv, pay_at, total)
        elif outcome == "split":
            first_part = max(1, min(total - 1, int(total * rng.uniform(0.3, 0.7))))
            second_at = pay_at + timedelta(days=rng.randint(1, 4))
            if second_at > end:
                meta["outcome"] = "paid_full"
                new_payment(inv, pay_at, total)
            else:
                new_payment(inv, pay_at, first_part)
                new_payment(inv, second_at, total - first_part)
        elif outcome == "partial_legit":
            new_payment(inv, pay_at, max(1, int(total * rng.uniform(0.3, 0.7))))

    # Legit foreign receipts on the last day in the early local hours (UTC date = previous day),
    # converted correctly at the local date's rate: same look as fx_partial_boundary (distractor).
    last = world.last_day
    offset = end.utcoffset()
    assert offset is not None  # the period bounds always carry the manifest timezone
    offset_h = int(offset.total_seconds() // 3600)
    night_candidates = [i for i in world.invoices
                        if world.inv_meta[i["invoice_id"]]["role"] == "plain" and i["currency"] != base
                        and world.inv_meta[i["invoice_id"]]["outcome"] == "paid_full"
                        and len(world.inv_meta[i["invoice_id"]]["payments"]) == 1
                        and world.inv_meta[i["invoice_id"]]["posted"].date().isoformat() < last]
    for inv in rng.sample(night_candidates, min(len(night_candidates), rng.randint(*cfg["n_last_day_night_payments"]))):
        meta = world.inv_meta[inv["invoice_id"]]
        pay = world.payment(meta["payments"][0])
        unpost_payment(world, pay["payment_id"])
        pay["timestamp"] = world.ts(end.replace(hour=rng.randint(0, max(0, offset_h - 1)), minute=rng.randint(0, 59), second=0))
        post_payment(world, ids, pay)
        meta["role"] = "night"

    # Customer advances: a known customer pays before any invoice exists (invoice_refs null).
    # Suspense 2900 is the correct posting: same look as a misallocated receipt (distractor).
    totals = {(i["entity_id"], i["total_minor"]) for i in world.invoices}
    for _ in range(rng.randint(*cfg["n_customer_advances"])):
        cid = rng.choice(customers)
        amount = with_cents(money(rng.randint(5, 80) * 10_000, base), base, rng)
        if (cid, amount) in totals:
            continue
        pay = {"doc_id": ids.new("DOC-", 8), "payment_id": ids.new("PAY-", 6),
               "timestamp": world.ts(rand_time(0, days_in_month - 1)), "currency": base,
               "amount_minor": amount, "entity_id": cid, "invoice_refs": None}
        world.payments.append(pay)
        post_payment(world, ids, pay)

    # Refunds for returned goods: a negative payment referencing a settled invoice, posted as
    # debit 4000 (sales returns) / credit 1000. The invoice's receipts no longer sum to its total,
    # which looks like a misallocated payment to a naive check (distractor).
    refund_candidates = [i for i in world.invoices
                         if world.inv_meta[i["invoice_id"]]["role"] == "plain" and i["currency"] == base
                         and world.inv_meta[i["invoice_id"]]["outcome"] == "paid_full"
                         and len(world.inv_meta[i["invoice_id"]]["payments"]) == 1]
    for inv in rng.sample(refund_candidates, min(len(refund_candidates), rng.randint(*cfg["n_refunds"]))):
        paid_at = datetime.fromisoformat(world.payment(world.inv_meta[inv["invoice_id"]]["payments"][0])["timestamp"])
        when = paid_at + timedelta(days=rng.randint(1, 5), hours=rng.randint(1, 4))
        if when > end:
            continue
        returned = rng.choice(inv["line_items"])
        amount = returned["unit_price_minor"] * max(1, returned["quantity"] // 2)
        refund = {"doc_id": ids.new("DOC-", 8), "payment_id": ids.new("PAY-", 6), "timestamp": world.ts(when),
                  "currency": base, "amount_minor": -amount, "entity_id": inv["entity_id"],
                  "invoice_refs": [{"invoice_id": inv["invoice_id"], "amount_minor": -amount}]}
        world.payments.append(refund)
        ts = refund["timestamp"]
        world.ledger.append(_row(ids, ts, "4000", amount, 0, base, inv["entity_id"], inv["invoice_id"], refund["payment_id"]))
        world.ledger.append(_row(ids, ts, "1000", 0, amount, base, inv["entity_id"], None, refund["payment_id"]))
        world.inv_meta[inv["invoice_id"]]["role"] = "refund"

    # Unidentified receipts: no remittance advice, so suspense 2900 is the correct posting.
    for _ in range(rng.randint(*cfg["n_unidentified_receipts"])):
        pay = {"doc_id": ids.new("DOC-", 8), "payment_id": ids.new("PAY-", 6),
               "timestamp": world.ts(rand_time(0, days_in_month - 1)), "currency": base,
               "amount_minor": with_cents(money(rng.randint(10_000, 500_000), base), base, rng),
               "entity_id": None, "invoice_refs": None}
        world.payments.append(pay)
        post_payment(world, ids, pay)

    # Operating expenses: realistic noise that makes the ledger large enough.
    target_rows = rng.randint(*cfg["ledger_rows"])
    while len(world.ledger) < target_rows:
        vid = rng.choice(vendors)
        ts = world.ts(rand_time(0, days_in_month - 1))
        amount = with_cents(money(rng.randint(20_000, 900_000), base), base, rng)
        world.ledger.append(_row(ids, ts, rng.choice(EXPENSE_ACCOUNTS), amount, 0, base, vid))
        world.ledger.append(_row(ids, ts, "1000", 0, amount, base, vid))

    check_invariants(world)
    return world


def check_invariants(world: World) -> None:
    """Accounting equalities that must hold on the clean history (before injection)."""
    balance: dict[str, int] = {}
    for r in world.ledger:
        balance[r["currency"]] = balance.get(r["currency"], 0) + r["debit_minor"] - r["credit_minor"]
    assert all(v == 0 for v in balance.values()), f"ledger not balanced: {balance}"
    for inv in world.invoices:
        rows = [r for r in world.ledger if r["account"] == "1100" and r["invoice_id"] == inv["invoice_id"]]
        meta = world.inv_meta[inv["invoice_id"]]
        assert sum(r["debit_minor"] for r in rows) == meta["booked"] == convert(
            world, inv["total_minor"], inv["currency"], inv["issued_at"][:10])
        allocated = sum(ref["amount_minor"] for p in world.payments for ref in (p["invoice_refs"] or [])
                        if ref["invoice_id"] == inv["invoice_id"] and p["amount_minor"] > 0)
        assert allocated == meta["alloc"] <= inv["total_minor"]
        assert sum(r["credit_minor"] for r in rows) == meta["credited"] <= meta["booked"]
    for p in world.payments:
        if p["invoice_refs"] is not None:
            assert sum(ref["amount_minor"] for ref in p["invoice_refs"]) == p["amount_minor"]
        if p["amount_minor"] < 0:
            continue  # refund: cash out, no receivable movement
        assert world.row(world.pay_meta[p["payment_id"]]["cash_row"])["debit_minor"] == convert(
            world, p["amount_minor"], p["currency"], p["timestamp"][:10])

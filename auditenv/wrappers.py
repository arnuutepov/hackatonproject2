"""Gymnasium-style wrappers (no gymnasium dependency).

    Wrapper                  delegation base: unwrapped, get_wrapper_attr, spaces, metadata
    RecordEpisodeStatistics  like gymnasium.wrappers.RecordEpisodeStatistics: return_queue,
                             length_queue, time_queue and info["episode"] = {"r", "l", "t"}
    AuditOptionsEnv          discrete "options" view of the audit task for tabular RL (training
                             tutorials): each option runs one audit skill as several REAL
                             env.step calls; info["action_mask"] marks valid options (Taxi tutorial)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal

from auditenv.common import canonical_json


class Wrapper:
    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def np_random_seed(self):
        return self.env.np_random_seed

    def get_wrapper_attr(self, name: str):
        if name in self.__dict__ or hasattr(type(self), name):
            return getattr(self, name)
        return self.env.get_wrapper_attr(name)

    def has_wrapper_attr(self, name: str) -> bool:
        return name in self.__dict__ or hasattr(type(self), name) or self.env.has_wrapper_attr(name)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


class RecordEpisodeStatistics(Wrapper):
    """Episode return, length (wrapper steps) and wall time, as in the Gymnasium tutorials."""

    def __init__(self, env, buffer_length: int = 100):
        super().__init__(env)
        self.return_queue: deque = deque(maxlen=buffer_length)
        self.length_queue: deque = deque(maxlen=buffer_length)
        self.time_queue: deque = deque(maxlen=buffer_length)
        self.episode_count = 0
        self._ret, self._len, self._t0 = 0.0, 0, 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ret, self._len, self._t0 = 0.0, 0, time.perf_counter()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += reward
        self._len += 1
        if terminated or truncated:
            elapsed = round(time.perf_counter() - self._t0, 6)
            info = {**info, "episode": {"r": self._ret, "l": self._len, "t": elapsed}}
            self.return_queue.append(self._ret)
            self.length_queue.append(self._len)
            self.time_queue.append(elapsed)
            self.episode_count += 1
        return obs, reward, terminated, truncated, info


# ----------------------------------------------------------------------------- options
OPTIONS = ("read_invoices", "read_payments", "read_ledger", "read_fx",
           "check_duplicates", "check_suspense", "check_cutoff", "check_fx", "submit")
TABLE_OF = {"read_invoices": "invoices", "read_payments": "payments", "read_ledger": "ledger", "read_fx": "fx"}
REQUIRES = {"check_duplicates": {"invoices"}, "check_suspense": {"payments", "ledger"},
            "check_cutoff": {"invoices", "ledger"}, "check_fx": {"invoices", "payments", "ledger", "fx"}}
UNKNOWN = 2


class AuditOptionsEnv(Wrapper):
    """Semi-MDP view for tabular agents. Observation = small discrete state tuple:

        (read invoices, payments, ledger, fx, ran dup, suspense, cutoff, fx checks,
         foreign documents present?, receipts naming an invoice in suspense?)

    with 0/1 bits and 2 = not yet known. Every option executes real primitive actions through
    the wrapped AuditEnv, so step costs, limits, grading and rewards are exactly the env's.
    The skills only use public records returned by the tools.

    NOT an interface for language models. The audit logic lives in these hand-written skills, so
    an agent over options only learns WHICH procedure to run and WHEN to stop: its +0.956 shows
    that the reward is learnable and generalizes to held-out episodes, not that a model can audit.
    A language model must act through the primitive tools (auditenv.llm); use these skills only
    to produce warm-up transcripts (examples/export_sft.py).

    mask=True:  info["action_mask"] marks valid options; agents should pick among them.
    mask=False: an invalid option (e.g. a check before its tables are read) is sent to the env as
                an unknown action: a bounded error that wastes one step (the env's own contract).
    """

    n_actions = len(OPTIONS)

    def __init__(self, env, mask: bool = True):
        super().__init__(env)
        self.mask_enabled = mask
        self.action_space = {"type": "integer", "minimum": 0, "maximum": len(OPTIONS) - 1, "options": list(OPTIONS)}

    # --------------------------------------------------------------- api
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._manifest = obs["manifest"]
        self._tables: dict[str, list] = {}
        self._docs: dict[str, dict] = {}
        self._checks: set[str] = set()
        self._findings: dict[tuple, dict] = {}
        return self._state(), {**info, "action_mask": self.action_mask()}

    def action_mask(self) -> list[int]:
        mask = []
        for name in OPTIONS:
            if name in TABLE_OF:
                ok = TABLE_OF[name] not in self._tables
            elif name in REQUIRES:
                ok = REQUIRES[name] <= set(self._tables) and name not in self._checks
            else:
                ok = True
            mask.append(int(ok))
        return mask

    def step(self, action: int):
        name = OPTIONS[int(action)]
        reward, terminated, truncated, info, inner = 0.0, False, False, {}, 0
        if not self.action_mask()[int(action)]:
            # invalid option: one wasted primitive step (unknown action -> bounded error)
            _, reward, terminated, truncated, info = self.env.step({"tool": f"option:{name}"})
            inner = 1
        elif name in TABLE_OF:
            reward, terminated, truncated, info, inner = self._read(TABLE_OF[name])
        elif name == "submit":
            findings = list(self._findings.values())[:30]
            _, reward, terminated, truncated, info = self.env.step(
                {"tool": "submit", "args": {"findings": {"episode_id": self._manifest["episode_id"], "findings": findings}}})
            inner = 1
        else:
            reward, terminated, truncated, info, inner = getattr(self, f"_{name}")()
            self._checks.add(name)
        info = {**info, "option": name, "inner_steps": inner, "action_mask": self.action_mask()}
        return self._state(), reward, terminated, truncated, info

    # ----------------------------------------------------------- helpers
    def _state(self) -> tuple:
        read = tuple(int(t in self._tables) for t in ("invoices", "payments", "ledger", "fx"))
        checks = tuple(int(c in self._checks) for c in ("check_duplicates", "check_suspense", "check_cutoff", "check_fx"))
        base = self._manifest["base_currency"]
        foreign = UNKNOWN if "invoices" not in self._tables else \
            int(any(i["currency"] != base for i in self._tables["invoices"]))
        suspense = UNKNOWN
        if {"payments", "ledger"} <= set(self._tables):
            refs = {p["payment_id"] for p in self._tables["payments"] if p["invoice_refs"]}
            suspense = int(any(r["account"] == "2900" and r["payment_id"] in refs for r in self._tables["ledger"]))
        return read + checks + (foreign, suspense)

    def _primitive(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def _read(self, table: str):
        rows, cursor, inner = [], None, 0
        reward, terminated, truncated, info = 0.0, False, False, {}
        while True:
            obs, reward, terminated, truncated, info = self._primitive(
                {"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}})
            inner += 1
            if terminated or truncated or obs["error"]:
                break
            rows += obs["result"]["rows"]
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
        self._tables[table] = rows
        return reward, terminated, truncated, info, inner

    def _add(self, finding: dict) -> None:
        self._findings.setdefault((finding["type"], tuple(sorted(finding["affected_ids"]))), finding)

    def _check_duplicates(self):
        groups = defaultdict(list)
        for inv in self._tables["invoices"]:
            groups[(inv["entity_id"], inv["issued_at"], inv["currency"], inv["total_minor"])].append(inv)
        inner, reward, terminated, truncated, info = 0, 0.0, False, False, {}
        for group in (g for g in groups.values() if len(g) > 1):
            docs = []
            for inv in group:
                obs, reward, terminated, truncated, info = self._primitive(
                    {"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})
                inner += 1
                if terminated or truncated:
                    return reward, terminated, truncated, info, inner
                if obs["result"]:
                    docs.append(obs["result"]["document"])
            by_lines = defaultdict(list)
            for d in docs:
                by_lines[canonical_json(d["line_items"])].append(d)
            for same in (s for s in by_lines.values() if len(s) > 1):
                self._add({"type": "duplicate_invoice", "affected_ids": sorted(d["invoice_id"] for d in same),
                           "claimed_amount_minor": same[0]["total_minor"], "currency": same[0]["currency"],
                           "evidence_ids": sorted(d["doc_id"] for d in same)[:10],
                           "explanation": "identical invoice booked twice"})
        return reward, terminated, truncated, info, inner

    def _check_suspense(self):
        refs = {p["payment_id"]: p for p in self._tables["payments"] if p["invoice_refs"]}
        rows = defaultdict(list)
        for r in self._tables["ledger"]:
            if r["account"] == "2900" and r["payment_id"] in refs:
                rows[r["payment_id"]].append(r)
        for pid, rs in rows.items():
            self._add({"type": "partial_payment", "affected_ids": [pid],
                       "claimed_amount_minor": sum(r["credit_minor"] - r["debit_minor"] for r in rs),
                       "currency": rs[0]["currency"], "evidence_ids": [refs[pid]["doc_id"]] + [r["row_id"] for r in rs][:9],
                       "explanation": "receipt names an invoice but sits in suspense"})
        return 0.0, False, False, {}, 0

    def _check_cutoff(self):
        start = datetime.fromisoformat(self._manifest["period_start"])
        end = datetime.fromisoformat(self._manifest["period_end"])
        ar = {}
        for r in self._tables["ledger"]:
            if r["account"] == "1100" and r["invoice_id"] and r["debit_minor"]:
                ar.setdefault(r["invoice_id"], r)
        for inv in self._tables["invoices"]:
            row = ar.get(inv["invoice_id"])
            if row and not start <= datetime.fromisoformat(inv["issued_at"]) <= end:
                self._add({"type": "period_cutoff", "affected_ids": [inv["invoice_id"]],
                           "claimed_amount_minor": row["debit_minor"], "currency": row["currency"],
                           "evidence_ids": [inv["doc_id"], row["row_id"]],
                           "explanation": "invoice dated outside the period is booked in it"})
        return 0.0, False, False, {}, 0

    def _check_fx(self):
        base, exps = self._manifest["base_currency"], self._manifest["currency_exponents"]
        rates = {(r["date"], r["source_currency"]): r["rate"] for r in self._tables["fx"]}

        def convert(amount, currency, date):
            rate = rates.get((date, currency))
            if rate is None:
                return None
            value = Decimal(amount) * Decimal(rate) * Decimal(10) ** (exps[base] - exps[currency])
            return int(value.quantize(Decimal(1), ROUND_HALF_EVEN))

        ar, cash = {}, {}
        for r in self._tables["ledger"]:
            if r["account"] == "1100" and r["invoice_id"] and r["debit_minor"]:
                ar.setdefault(r["invoice_id"], r)
            if r["account"] == "1000" and r["payment_id"] and r["debit_minor"]:
                cash.setdefault(r["payment_id"], r)
        candidates = [(inv, ar.get(inv["invoice_id"]), inv["invoice_id"], inv["total_minor"], inv["issued_at"])
                      for inv in self._tables["invoices"]]
        candidates += [(p, cash.get(p["payment_id"]), p["payment_id"], p["amount_minor"], p["timestamp"])
                       for p in self._tables["payments"]]
        for doc, row, target, amount, ts in candidates:
            if row is None or doc["currency"] == base:
                continue
            date = ts[:10]
            correct = convert(amount, doc["currency"], date)
            if correct is None or correct == row["debit_minor"]:
                continue
            self._add({"type": "fx_conversion", "affected_ids": [target],
                       "claimed_amount_minor": abs(row["debit_minor"] - correct), "currency": row["currency"],
                       "evidence_ids": [doc["doc_id"], row["row_id"], f"{date}:{doc['currency']}:{base}"],
                       "explanation": "booked amount differs from the local-date conversion"})
        return 0.0, False, False, {}, 0

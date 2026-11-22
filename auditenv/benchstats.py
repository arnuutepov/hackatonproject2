"""Benchmark statistics: bootstrap confidence intervals and paired comparisons (stdlib only).

Used by examples/train_q_learning.py. Everything is deterministic: resampling uses a seeded
random.Random, so the same per-episode results always give the same intervals.

* bootstrap_ci: percentile CI of the mean over episodes.
* paired_bootstrap: the same episodes under two policies; CI of the mean difference, the share
  of resamples where the difference is <= 0 (a one-sided bootstrap p-value) and episode-level
  wins / ties / losses.
"""

from __future__ import annotations

import random
import statistics

N_BOOT = 4000
ALPHA = 0.05
TIE = 1e-9


def _quantiles(sorted_values: list[float], alpha: float) -> tuple[float, float]:
    n = len(sorted_values)
    lo = sorted_values[max(0, int(n * alpha / 2))]
    hi = sorted_values[min(n - 1, int(n * (1 - alpha / 2)) - 1)]
    return lo, hi


def _boot_means(values: list[float], n_boot: int, seed: str) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    idx = range(n)
    return sorted(statistics.fmean(values[i] for i in rng.choices(idx, k=n)) for _ in range(n_boot))


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, alpha: float = ALPHA, seed: str = "ci") -> list[float]:
    """Percentile bootstrap CI of the mean. A single value or constant input gives a zero-width CI."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if len(set(values)) == 1:
        return [round(values[0], 4), round(values[0], 4)]
    lo, hi = _quantiles(_boot_means(list(values), n_boot, seed), alpha)
    return [round(lo, 4), round(hi, 4)]


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = N_BOOT, alpha: float = ALPHA,
                     seed: str = "paired") -> dict:
    """Compare policy A to policy B on the same episodes (a[i] and b[i] are the same episode)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired_bootstrap needs two equal, non-empty lists")
    diff = [x - y for x, y in zip(a, b)]
    if len(set(diff)) == 1:
        means = [diff[0]] * n_boot
    else:
        means = _boot_means(diff, n_boot, seed)
    lo, hi = _quantiles(means, alpha)
    return {"diff": round(statistics.fmean(diff), 4), "ci": [round(lo, 4), round(hi, 4)],
            "p_le_0": round(sum(1 for m in means if m <= 0) / len(means), 4),
            "wins": sum(1 for d in diff if d > TIE), "ties": sum(1 for d in diff if abs(d) <= TIE),
            "losses": sum(1 for d in diff if d < -TIE), "episodes": len(diff)}


def recall_by_type(rows: list[dict]) -> dict:
    """Pool grader-log rows: recall per error type = sum tp / sum (tp + fn)."""
    tp: dict[str, int] = {}
    total: dict[str, int] = {}
    for r in rows:
        for etype, c in (r.get("diagnostics") or {}).get("by_type", {}).items():
            tp[etype] = tp.get(etype, 0) + c.get("tp", 0)
            total[etype] = total.get(etype, 0) + c.get("tp", 0) + c.get("fn", 0)
    return {t: (round(tp[t] / total[t], 4) if total[t] else None) for t in sorted(total)}

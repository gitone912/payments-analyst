"""
Driver analysis for a change in a rate metric between two periods.

Overall rate R = sum_i w_i * r_i  (w = share of denominator, r = segment rate).
The change splits exactly into
    mix_i  = (w_i^B - w_i^A) * (rbar_i - R^A)        # traffic moved toward/away from segment i
    rate_i = (r_i^B - r_i^A) * wbar_i                # segment i itself got better/worse
with rbar, wbar = averages of the two periods. sum(mix_i) + sum(rate_i) == R^B - R^A
(the "- R^A" term in mix sums to zero because shares sum to 1 - it just makes each
segment's mix effect readable: moving share into a below-average segment is negative).

Each segment's rate change also gets a two-proportion z-score so small, noisy segments
aren't reported as drivers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class Decomposition:
    dimension: str
    rate_a: float
    rate_b: float
    delta: float
    mix_total: float
    rate_total: float
    like_for_like_b: float      # current rates at baseline mix
    segments: pd.DataFrame      # per-segment detail

    @property
    def mix_share(self) -> float:
        return self.mix_total / self.delta if self.delta else 0.0


def decompose(df: pd.DataFrame, dimension: str) -> Decomposition:
    """df columns: segment, period ('baseline'|'current'), denom, numer"""
    p = df.pivot_table(index="segment", columns="period", values=["denom", "numer"],
                       aggfunc="sum", fill_value=0)
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    for c in ("denom_baseline", "denom_current", "numer_baseline", "numer_current"):
        if c not in p:
            p[c] = 0
    p = p.astype(float)
    na, nb = p.denom_baseline.sum(), p.denom_current.sum()
    Ra = p.numer_baseline.sum() / na
    Rb = p.numer_current.sum() / nb
    p["w_a"] = p.denom_baseline / na
    p["w_b"] = p.denom_current / nb
    p["r_a"] = (p.numer_baseline / p.denom_baseline).where(p.denom_baseline > 0)
    p["r_b"] = (p.numer_current / p.denom_current).where(p.denom_current > 0)
    # segment that only exists in one period: no rate effect, all mix
    p["r_a"] = p.r_a.fillna(p.r_b)
    p["r_b"] = p.r_b.fillna(p.r_a)
    rbar = (p.r_a + p.r_b) / 2
    wbar = (p.w_a + p.w_b) / 2
    p["mix_effect"] = (p.w_b - p.w_a) * (rbar - Ra)
    p["rate_effect"] = (p.r_b - p.r_a) * wbar
    p["total_effect"] = p.mix_effect + p.rate_effect

    # two-proportion z for the segment's rate change
    def z(row):
        n1, n2 = row.denom_baseline, row.denom_current
        if n1 < 30 or n2 < 30:
            return float("nan")
        pooled = (row.numer_baseline + row.numer_current) / (n1 + n2)
        se = math.sqrt(max(pooled * (1 - pooled), 1e-12) * (1 / n1 + 1 / n2))
        return (row.r_b - row.r_a) / se
    p["z_rate"] = p.apply(z, axis=1)
    p = p.sort_values("total_effect").reset_index()
    lfl = float((p.w_a * p.r_b).sum())
    return Decomposition(dimension, Ra, Rb, Rb - Ra, float(p.mix_effect.sum()),
                         float(p.rate_effect.sum()), lfl, p)


def overall_change_ci(na: float, sa: float, nb: float, sb: float) -> tuple[float, float, float]:
    """95% CI for the difference in proportions (current - baseline)."""
    ra, rb = sa / na, sb / nb
    se = math.sqrt(ra * (1 - ra) / na + rb * (1 - rb) / nb)
    d = rb - ra
    return d, d - 1.96 * se, d + 1.96 * se

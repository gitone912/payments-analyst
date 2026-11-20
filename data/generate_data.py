"""
Synthetic payments dataset generator.

Creates a DuckDB file with 5 related tables and *deliberately planted* scenarios
so that every failure/exception case in the brief can be demonstrated
deterministically (fixed seed).

Window: 2026-08-31 (Mon) .. 2026-10-25 (Sun) = 8 full weeks.

PLANTED SCENARIOS (the "ground truth" the evals check against)
---------------------------------------------------------------
1. MIX SHIFT (main story)  - From 2026-10-19 merchant "QuickTix" (ticketing) runs a
   festival sale. Its traffic is mostly UPI via "Delta Co-op Bank", a segment with a
   structurally low success rate (~70%). Segment rates do NOT change, but the share
   of volume does -> overall success rate falls. (Simpson's-paradox style.)
2. GENUINE RATE DROP - From 2026-10-21, "Coastal Bank" NetBanking degrades
   (issuer downtime): success rate falls ~11 pts. Smaller, real driver.
3. DUPLICATES - 2026-10-20..22 an ingestion retry re-inserted ~16% of FAILED rows
   with the same txn_id (later ingested_at). Raw counts overstate failures.
4. MISSING DATA - 2026-10-14 pipeline outage: only transactions before 12:00 landed.
5. MISSING DIMENSION - 2026-10-08..10 ~6% of UPI rows have NULL bank_id
   (wallet rows have NULL bank_id legitimately).
6. SOURCE DISAGREEMENT - `settlements` (finance system) resolves PENDING txns later
   and is built from de-duplicated data, so its success counts differ from
   `transactions`.
7. PII - transactions.customer_vpa is PII; only the risk_ops role may read it.
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

SEED = 42
START = date(2026, 8, 31)
DAYS = 56
SALE_START = date(2026, 10, 19)
COASTAL_NB_DEGRADE = date(2026, 10, 21)
DUP_DAYS = {date(2026, 10, 20), date(2026, 10, 21), date(2026, 10, 22)}
GAP_DAY = date(2026, 10, 14)
NULL_BANK_DAYS = {date(2026, 10, 8), date(2026, 10, 9), date(2026, 10, 10)}

BANKS = [
    # bank_id, name, tier
    ("B01", "Northstar Bank", "large_private"),
    ("B02", "Coastal Bank", "large_private"),
    ("B03", "Unity Public Bank", "public_sector"),
    ("B04", "Delta Co-op Bank", "cooperative"),
    ("B05", "Summit Federal", "mid_private"),
    ("B06", "Riverside Bank", "small_finance"),
]
BANK_W = np.array([0.27, 0.22, 0.20, 0.05, 0.16, 0.10])

METHODS = [("UPI", "UPI"), ("CARD", "Card"), ("NB", "NetBanking"), ("WALLET", "Wallet")]
BASE_SR = {"UPI": 0.880, "CARD": 0.900, "NB": 0.885, "WALLET": 0.935}
# bank x method adjustment (Delta Co-op UPI is structurally weak)
BANK_ADJ = {
    "B01": 0.015, "B02": 0.010, "B03": -0.010, "B04": -0.040,
    "B05": 0.005, "B06": -0.020,
}
DELTA_UPI_EXTRA = -0.140

CATEGORIES = {
    "ecommerce": 0.00, "food_delivery": 0.005, "travel": -0.010,
    "gaming": -0.015, "utilities": 0.010, "ticketing": -0.005, "education": 0.005,
}
CAT_METHOD_MIX = {  # UPI, CARD, NB, WALLET
    "ecommerce":     [0.55, 0.28, 0.07, 0.10],
    "food_delivery": [0.65, 0.18, 0.02, 0.15],
    "travel":        [0.35, 0.45, 0.15, 0.05],
    "gaming":        [0.70, 0.10, 0.05, 0.15],
    "utilities":     [0.50, 0.15, 0.30, 0.05],
    "ticketing":     [0.60, 0.30, 0.05, 0.05],
    "education":     [0.45, 0.25, 0.25, 0.05],
}
CITY_TIERS = ["tier1", "tier2", "tier3"]

FAIL_REASONS = ["BANK_DECLINED", "TIMEOUT", "INSUFFICIENT_FUNDS", "USER_DROPPED", "RISK_BLOCKED"]
FAIL_P = [0.30, 0.25, 0.20, 0.18, 0.07]


def build_merchants(rng: np.random.Generator) -> pd.DataFrame:
    names = [
        "ShopKart", "DailyBasket", "Zipply", "UrbanThreads", "GadgetHub", "BookNook",
        "FreshFeast", "BiryaniBox", "TiffinTime", "QuickBite",
        "TripWise", "RailEase", "SkyHop",
        "PlayArena", "LudoLeague", "QuestCraft",
        "QuickTix", "ShowSeat", "MatchPass",
        "PowerPay", "WaterWorks", "GasGrid", "BroadNet",
        "LearnLoop", "ExamAce", "CodeCamp",
        "HomeHaven", "PetPantry", "StyleStreet", "KidsKorner",
        "MealMate", "CafeCrate", "BusBuddy", "StayNest",
        "CardClash", "FantasyXI", "CinemaCue", "MobiRecharge", "SkillSprout", "TutorTap",
    ]
    cats = (
        ["ecommerce"] * 6 + ["food_delivery"] * 4 + ["travel"] * 3 + ["gaming"] * 3
        + ["ticketing"] * 3 + ["utilities"] * 4 + ["education"] * 3
        + ["ecommerce"] * 4 + ["food_delivery"] * 2 + ["travel"] * 2
        + ["gaming"] * 2 + ["ticketing"] * 1 + ["utilities"] * 1 + ["education"] * 2
    )
    n = len(names)
    rows = []
    for i, (nm, cat) in enumerate(zip(names, cats)):
        rows.append({
            "merchant_id": f"M{i+1:03d}",
            "merchant_name": nm,
            "category": cat,
            "city_tier": CITY_TIERS[rng.choice(3, p=[0.55, 0.30, 0.15])],
            "onboarded_on": START - timedelta(days=int(rng.integers(90, 1500))),
            "kam_portfolio": "PORTFOLIO_A" if cat in ("ecommerce", "food_delivery") else "PORTFOLIO_B",
        })
    df = pd.DataFrame(rows)
    # Zipf-ish volume weights; QuickTix modest at baseline
    w = 1.0 / np.arange(1, n + 1) ** 0.9
    rng.shuffle(w)
    df["vol_w"] = w / w.sum()
    qt = df.index[df.merchant_name == "QuickTix"][0]
    df.loc[qt, "vol_w"] = 0.012
    df["vol_w"] = df["vol_w"] / df["vol_w"].sum()
    return df


def gen_day(rng, d: date, merchants: pd.DataFrame, n: int, override_merchant=None,
            method_p=None, bank_p=None) -> pd.DataFrame:
    if override_merchant is None:
        m_idx = rng.choice(len(merchants), size=n, p=merchants.vol_w.values)
    else:
        m_idx = np.full(n, override_merchant)
    m = merchants.iloc[m_idx].reset_index(drop=True)

    if method_p is None:
        mix = np.array([CAT_METHOD_MIX[c] for c in m.category])
        u = rng.random(n)[:, None]
        meth_i = (u > mix.cumsum(axis=1)).sum(axis=1)
    else:
        meth_i = rng.choice(4, size=n, p=method_p)
    meth = np.array([METHODS[i][0] for i in meth_i])

    bp = BANK_W if bank_p is None else np.array(bank_p)
    bank_i = rng.choice(len(BANKS), size=n, p=bp / bp.sum())
    bank = np.array([BANKS[i][0] for i in bank_i], dtype=object)
    bank[meth == "WALLET"] = None

    # success probability
    p = np.array([BASE_SR[x] for x in meth])
    p += np.array([BANK_ADJ.get(b, 0.0) if b else 0.0 for b in bank])
    p += np.array([CATEGORIES[c] for c in m.category])
    p[(bank == "B04") & (meth == "UPI")] += DELTA_UPI_EXTRA
    coastal_nb = (bank == "B02") & (meth == "NB")
    if d >= COASTAL_NB_DEGRADE:
        p[coastal_nb] -= 0.11
    p += rng.normal(0, 0.004)  # day-level noise
    p = np.clip(p, 0.05, 0.99)

    u = rng.random(n)
    pending = rng.random(n) < 0.015
    status = np.where(u < p, "SUCCESS", "FAILED").astype(object)
    status[pending] = "PENDING"

    reasons = rng.choice(FAIL_REASONS, size=n, p=FAIL_P).astype(object)
    reasons[(bank == "B04") & (meth == "UPI") & (rng.random(n) < 0.6)] = "TIMEOUT"
    if d >= COASTAL_NB_DEGRADE:
        reasons[coastal_nb & (rng.random(n) < 0.7)] = "BANK_DOWNTIME"
    reasons[status != "FAILED"] = None

    # diurnal hours
    hour_p = np.array([1, .6, .4, .3, .3, .5, 1, 2, 3.5, 4.5, 5, 5.5, 5.8, 5.5, 5, 5,
                       5.2, 5.6, 6.2, 6.8, 6.5, 5, 3.5, 2])
    hour = rng.choice(24, size=n, p=hour_p / hour_p.sum())
    secs = rng.integers(0, 3600, size=n)
    base = datetime(d.year, d.month, d.day)
    created = [base + timedelta(hours=int(h), seconds=int(s)) for h, s in zip(hour, secs)]

    amt_scale = {"ecommerce": 1200, "food_delivery": 380, "travel": 4200, "gaming": 250,
                 "utilities": 900, "ticketing": 750, "education": 2600}
    amount = np.round(rng.lognormal(0, 0.6, n) * np.array([amt_scale[c] for c in m.category]), 2)

    vpa_users = rng.integers(10000, 99999, size=n)
    handles = rng.choice(["okaxis", "oksbi", "ybl", "paytm", "ibl"], size=n)
    vpa = np.array([f"user{u}@{h}" for u, h in zip(vpa_users, handles)], dtype=object)
    vpa[meth != "UPI"] = None

    return pd.DataFrame({
        "created_at": created,
        "merchant_id": m.merchant_id.values,
        "bank_id": bank,
        "method_id": meth,
        "amount_inr": amount,
        "status": status,
        "failure_reason": reasons,
        "customer_vpa": vpa,
    })


def main(out_path: str) -> None:
    rng = np.random.default_rng(SEED)
    merchants = build_merchants(rng)
    qt_idx = int(merchants.index[merchants.merchant_name == "QuickTix"][0])

    frames = []
    for k in range(DAYS):
        d = START + timedelta(days=k)
        dow = d.weekday()
        n = int(rng.normal(4800, 120) * (1.12 if dow >= 5 else 1.0))
        day = gen_day(rng, d, merchants, n)
        if d >= SALE_START:
            # festival sale: extra QuickTix traffic, UPI-heavy, Delta Co-op-heavy
            extra = int(rng.normal(1500, 60))
            sale = gen_day(rng, d, merchants, extra, override_merchant=qt_idx,
                           method_p=[0.88, 0.08, 0.02, 0.02],
                           bank_p=[0.10, 0.08, 0.10, 0.60, 0.07, 0.05])
            day = pd.concat([day, sale], ignore_index=True)
        if d == GAP_DAY:
            day = day[pd.to_datetime(day.created_at).dt.hour < 12]
        if d in NULL_BANK_DAYS:
            mask = (day.method_id == "UPI") & (rng.random(len(day)) < 0.06)
            day.loc[mask, "bank_id"] = None
        frames.append(day)

    tx = pd.concat(frames, ignore_index=True).sort_values("created_at").reset_index(drop=True)
    tx.insert(0, "txn_id", [f"T{i:08d}" for i in range(1, len(tx) + 1)])
    lag = rng.integers(5, 300, size=len(tx))
    tx["ingested_at"] = pd.to_datetime(tx.created_at) + pd.to_timedelta(lag, unit="s")

    # --- planted duplicates (ingestion retry of failed rows) -------------------
    tx_day = pd.to_datetime(tx.created_at).dt.date
    cand = tx[(tx_day.isin(DUP_DAYS)) & (tx.status == "FAILED")]
    dups = cand.sample(frac=0.16, random_state=SEED)  # ~2% of those days' rows
    dups = dups.copy()
    dups["ingested_at"] = dups["ingested_at"] + pd.Timedelta(hours=6)
    tx_raw = pd.concat([tx, dups], ignore_index=True).sort_values(["created_at", "txn_id"])

    # --- settlements (finance source of truth, different logic) ----------------
    # Built from de-duplicated txns; ~65% of PENDING later resolve to SUCCESS.
    st = tx.copy()
    resolve = (st.status == "PENDING") & (rng.random(len(st)) < 0.65)
    st.loc[resolve, "status"] = "SUCCESS"
    st["settlement_date"] = pd.to_datetime(st.created_at).dt.date
    settlements = (
        st.groupby(["settlement_date", "merchant_id"])
        .agg(attempted_count=("txn_id", "count"),
             success_count=("status", lambda s: int((s == "SUCCESS").sum())),
             settled_amount_inr=("amount_inr",
                                 lambda a: float(a[st.loc[a.index, "status"] == "SUCCESS"].sum())))
        .reset_index()
    )
    settlements["settled_amount_inr"] = settlements["settled_amount_inr"].round(2)

    banks = pd.DataFrame(BANKS, columns=["bank_id", "bank_name", "bank_tier"])
    methods = pd.DataFrame(METHODS, columns=["method_id", "method_name"])
    methods["is_bank_routed"] = methods.method_id != "WALLET"
    merchants_out = merchants.drop(columns=["vol_w"])

    if os.path.exists(out_path):
        os.remove(out_path)
    con = duckdb.connect(out_path)
    con.execute("CREATE SCHEMA raw")
    for name, df in [("transactions", tx_raw), ("merchants", merchants_out), ("banks", banks),
                     ("payment_methods", methods), ("settlements", settlements)]:
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE raw.{name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    con.execute("ALTER TABLE raw.transactions ALTER created_at TYPE TIMESTAMP")
    con.execute("ALTER TABLE raw.transactions ALTER ingested_at TYPE TIMESTAMP")
    con.execute("ALTER TABLE raw.settlements ALTER settlement_date TYPE DATE")
    con.execute("ALTER TABLE raw.merchants ALTER onboarded_on TYPE DATE")
    # clean, de-duplicated view used by the analyst by default
    con.execute("""
        CREATE VIEW raw.transactions_clean AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY txn_id ORDER BY ingested_at) AS rn
            FROM raw.transactions) WHERE rn = 1
    """)
    con.close()

    print(f"Wrote {out_path}")
    print(f"  transactions (raw): {len(tx_raw):,}  (unique: {len(tx):,}, dup rows: {len(dups):,})")
    print(f"  merchants: {len(merchants_out)}  banks: {len(banks)}  settlements: {len(settlements):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "payments.duckdb"))
    main(ap.parse_args().out)

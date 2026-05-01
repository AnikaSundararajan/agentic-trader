"""
Compustat quarterly fundamentals via CCM link.
All data is lagged to report date (rdq) or datadate + 90 days if rdq is null.
No feature from this module may be used before its as_of_date.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

RAW_DIR = Path(__file__).parent / "raw"


@lru_cache(maxsize=1)
def _load_fundq() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "compustat_fundq.parquet")
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"] = pd.to_datetime(df["rdq"])
    # as_of_date: the first date this data is usable (point-in-time safe)
    # Use rdq if available, else datadate + 90 days
    df["as_of_date"] = df["rdq"].fillna(df["datadate"] + pd.Timedelta(days=90))
    # Guard: rdq occasionally precedes datadate due to Compustat data errors.
    # Enforce as_of_date >= datadate so we never use data before the period ends.
    df["as_of_date"] = df[["as_of_date", "datadate"]].max(axis=1)
    return df


def get_fundamentals_as_of(permno: int, as_of_date: pd.Timestamp) -> pd.Series | None:
    """
    Return the most recent quarterly fundamental row for a PERMNO
    where as_of_date <= as_of_date. Returns None if no data available.
    """
    df = _load_fundq()
    eligible = df[(df["permno"] == permno) & (df["as_of_date"] <= as_of_date)]
    if eligible.empty:
        return None
    return eligible.sort_values("as_of_date").iloc[-1]


def get_fundamentals_panel(permnos: list[int], as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Return one row per PERMNO: the most recent fundamentals available as of a date.
    Used to build cross-sectional features for a single trading day.
    """
    df = _load_fundq()
    eligible = df[(df["permno"].isin(permnos)) & (df["as_of_date"] <= as_of_date)]

    if eligible.empty:
        return pd.DataFrame()

    # Keep latest available row per permno
    idx = eligible.groupby("permno")["as_of_date"].idxmax()
    return eligible.loc[idx].reset_index(drop=True)


def compute_derived_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived accounting ratios from raw Compustat columns.
    All inputs are already point-in-time lagged — no additional lag needed here.
    """
    out = df.copy()

    # Profitability
    out["gross_margin"] = (out["saleq"] - out["cogsq"]) / out["saleq"].replace(0, np.nan)
    out["net_margin"] = out["niq"] / out["saleq"].replace(0, np.nan)
    out["roe"] = out["niq"] / out["ceqq"].replace(0, np.nan)
    out["roa"] = out["niq"] / out["atq"].replace(0, np.nan)

    # Leverage
    out["debt_to_equity"] = (out["dlttq"].fillna(0) + out["dlcq"].fillna(0)) / out["ceqq"].replace(0, np.nan)
    out["debt_to_assets"] = (out["dlttq"].fillna(0) + out["dlcq"].fillna(0)) / out["atq"].replace(0, np.nan)

    # Liquidity
    out["current_ratio"] = out["actq"] / out["lctq"].replace(0, np.nan)

    # Cash flow
    out["fcf"] = out["oancfy"] - out["capxq"]
    out["fcf_yield"] = out["fcf"] / (out["prccq"] * out["cshoq"]).replace(0, np.nan)

    # Accruals (Sloan 1996): earnings quality signal
    # accruals = (net income - operating cash flow) / avg assets
    out["accruals"] = (out["ibq"] - out["oancfy"]) / out["atq"].replace(0, np.nan)

    # Piotroski F-Score components (simplified 5-factor version)
    out["f_roa_pos"] = (out["roa"] > 0).astype(float)
    out["f_cfo_pos"] = (out["oancfy"] > 0).astype(float)
    out["f_accrual"] = (out["oancfy"] > out["ibq"]).astype(float)
    out["f_leverage_dec"] = np.nan  # requires prior-period data; filled in feature_store
    out["f_current_inc"] = np.nan  # requires prior-period data; filled in feature_store
    out["piotroski_partial"] = out[["f_roa_pos", "f_cfo_pos", "f_accrual"]].sum(axis=1)

    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_lag_correctness():
    """
    Verify that as_of_date is always >= datadate (report can't precede period end)
    and that rdq takes precedence over the 90-day fallback.
    """
    df = _load_fundq()

    # as_of_date must be after datadate
    bad_lag = df[df["as_of_date"] < df["datadate"]]
    assert bad_lag.empty, f"{len(bad_lag)} rows where as_of_date < datadate"

    # Rows with rdq should use max(rdq, datadate) — rdq occasionally precedes datadate
    has_rdq = df[df["rdq"].notna()].copy()
    has_rdq["expected_as_of"] = has_rdq[["rdq", "datadate"]].max(axis=1)
    mismatch = has_rdq[has_rdq["as_of_date"] != has_rdq["expected_as_of"]]
    assert mismatch.empty, f"{len(mismatch)} rows where rdq exists but as_of_date != max(rdq, datadate)"

    # Rows without rdq should use datadate + 90
    no_rdq = df[df["rdq"].isna()].copy()
    no_rdq["expected_as_of"] = no_rdq["datadate"] + pd.Timedelta(days=90)
    mismatch2 = no_rdq[no_rdq["as_of_date"] != no_rdq["expected_as_of"]]
    assert mismatch2.empty, f"{len(mismatch2)} rows where rdq is null but as_of_date != datadate+90"

    print(f"[PASS] lag correctness: {len(df):,} rows checked")


def _test_panel_no_future_data():
    """Spot-check: no row returned by get_fundamentals_panel has as_of_date > query date."""
    import random
    df = _load_fundq()
    permnos = df["permno"].dropna().unique().tolist()
    sample = random.sample(list(permnos), min(50, len(permnos)))

    test_dates = [pd.Timestamp("2010-06-30"), pd.Timestamp("2015-12-31"), pd.Timestamp("2020-03-31")]
    for dt in test_dates:
        panel = get_fundamentals_panel(sample, dt)
        if panel.empty:
            continue
        future_rows = panel[panel["as_of_date"] > dt]
        assert future_rows.empty, f"Future data in panel for {dt}: {len(future_rows)} rows"

    print("[PASS] no future data in panel lookups")


if __name__ == "__main__":
    _test_lag_correctness()
    _test_panel_no_future_data()

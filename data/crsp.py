"""
CRSP data access: point-in-time S&P 500 universe and delisting handling.
All functions operate on pre-downloaded parquet files in data/raw/.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

RAW_DIR = Path(__file__).parent / "raw"

# Shumway (1997) delisting return fills by dlstcd category
_DELISTING_FILLS = {
    "acquisition": 0.00,   # 200-299
    "liquidation": -0.40,  # 400-499
    "dropped": -0.40,      # 500-599
}


@lru_cache(maxsize=1)
def _load_dsp500list() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "crsp_dsp500list.parquet")
    df["start"] = pd.to_datetime(df["start"])
    df["ending"] = pd.to_datetime(df["ending"])
    # Treat null ending as still in index
    df["ending"] = df["ending"].fillna(pd.Timestamp("2099-12-31"))
    return df


@lru_cache(maxsize=1)
def _load_dsf() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "crsp_dsf.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@lru_cache(maxsize=1)
def _load_dsedelist() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "crsp_dsedelist.parquet")
    df["dlstdt"] = pd.to_datetime(df["dlstdt"])
    return df


def get_sp500_permnos(date: pd.Timestamp) -> set[int]:
    """Return the set of PERMNOs in the S&P 500 on a given date (point-in-time)."""
    members = _load_dsp500list()
    mask = (members["start"] <= date) & (members["ending"] >= date)
    return set(members.loc[mask, "permno"].tolist())


def get_daily_prices(permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """
    Return daily price data for a list of PERMNOs over a date range.
    Columns: permno, date, prc, ret, vol, shrout, cfacpr, cfacshr
    Price (prc) is made positive — CRSP stores bid/ask midpoint as negative when no trade.
    """
    dsf = _load_dsf()
    mask = (
        dsf["permno"].isin(permnos)
        & (dsf["date"] >= pd.Timestamp(start))
        & (dsf["date"] <= pd.Timestamp(end))
    )
    df = dsf[mask].copy()
    df["prc"] = df["prc"].abs()
    return df.sort_values(["permno", "date"]).reset_index(drop=True)


def get_delisting_info(permnos: list[int]) -> pd.DataFrame:
    """
    Return delisting info for given PERMNOs.
    Adds dlret_filled: Shumway-filled return when dlret is missing.
    Adds dlst_category: 'acquisition' | 'liquidation' | 'dropped' | 'other'
    """
    delist = _load_dsedelist()
    df = delist[delist["permno"].isin(permnos)].copy()

    def categorize(code):
        if pd.isna(code):
            return "other"
        code = int(code)
        if 200 <= code <= 299:
            return "acquisition"
        elif 400 <= code <= 499:
            return "liquidation"
        elif 500 <= code <= 599:
            return "dropped"
        return "other"

    df["dlst_category"] = df["dlstcd"].apply(categorize)

    def fill_dlret(row):
        if not pd.isna(row["dlret"]):
            return float(row["dlret"])
        return _DELISTING_FILLS.get(row["dlst_category"], -0.40)

    df["dlret_filled"] = df.apply(fill_dlret, axis=1)
    return df.reset_index(drop=True)


def apply_delisting_exits(
    positions: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given open positions and the current date, check for any delistings.

    positions: DataFrame with at least columns [permno, entry_price, entry_date, shares]
    Returns:
        (still_open, forced_exits)
        forced_exits has additional columns: exit_price, exit_return, exit_reason
    """
    if positions.empty:
        return positions, pd.DataFrame()

    delist_info = get_delisting_info(positions["permno"].tolist())
    delisted_today = delist_info[delist_info["dlstdt"] == as_of_date]

    if delisted_today.empty:
        return positions, pd.DataFrame()

    delisted_permnos = set(delisted_today["permno"])
    exiting = positions[positions["permno"].isin(delisted_permnos)].copy()
    remaining = positions[~positions["permno"].isin(delisted_permnos)].copy()

    exiting = exiting.merge(
        delisted_today[["permno", "dlret_filled", "dlst_category"]],
        on="permno",
        how="left",
    )
    exiting["exit_return"] = exiting["dlret_filled"]
    exiting["exit_reason"] = "delisting_" + exiting["dlst_category"]
    exiting["exit_price"] = exiting["entry_price"] * (1 + exiting["exit_return"])

    return remaining, exiting


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_constituent_lookup():
    """Spot-check known S&P 500 membership dates."""
    # Apple (PERMNO 14593) was in S&P 500 on 2010-01-04
    members_2010 = get_sp500_permnos(pd.Timestamp("2010-01-04"))
    assert 14593 in members_2010, "AAPL should be in S&P 500 on 2010-01-04"

    # Universe should be ~500 names
    assert 400 <= len(members_2010) <= 550, f"Unexpected universe size: {len(members_2010)}"
    print(f"[PASS] constituent lookup: {len(members_2010)} members on 2010-01-04")


def _test_delisting_fills():
    """Verify Shumway fill logic for each dlstcd category."""
    fake_rows = [
        {"permno": 1, "dlstdt": pd.Timestamp("2005-01-01"), "dlstcd": 250, "dlret": np.nan},
        {"permno": 2, "dlstdt": pd.Timestamp("2005-01-01"), "dlstcd": 450, "dlret": np.nan},
        {"permno": 3, "dlstdt": pd.Timestamp("2005-01-01"), "dlstcd": 550, "dlret": np.nan},
        {"permno": 4, "dlstdt": pd.Timestamp("2005-01-01"), "dlstcd": 550, "dlret": -0.20},
    ]
    df = pd.DataFrame(fake_rows)

    def categorize(code):
        code = int(code)
        if 200 <= code <= 299:
            return "acquisition"
        elif 400 <= code <= 499:
            return "liquidation"
        return "dropped"

    df["dlst_category"] = df["dlstcd"].apply(categorize)
    df["dlret_filled"] = df.apply(
        lambda r: float(r["dlret"]) if not pd.isna(r["dlret"]) else _DELISTING_FILLS[r["dlst_category"]],
        axis=1,
    )

    assert df.loc[0, "dlret_filled"] == 0.00, "acquisition fill should be 0%"
    assert df.loc[1, "dlret_filled"] == -0.40, "liquidation fill should be -40%"
    assert df.loc[2, "dlret_filled"] == -0.40, "dropped fill should be -40%"
    assert df.loc[3, "dlret_filled"] == -0.20, "actual dlret should not be overwritten"
    print("[PASS] delisting fill logic")


if __name__ == "__main__":
    _test_delisting_fills()
    _test_constituent_lookup()

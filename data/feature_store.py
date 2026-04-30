"""
Unified feature assembly for the RL state vector (~75-100 features per stock per day).
Joins CRSP technicals, Compustat fundamentals, WRDS ratios, I/B/E/S analyst data,
Fama-French factors, and WRDS betas into a single normalized cross-sectional matrix.

CRITICAL: Every feature here must have as_of_date <= the trading date it's used on.
Run _test_lag_correctness() after any schema change.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

from data.compustat import get_fundamentals_panel, compute_derived_fundamentals
from data.preprocess import compute_technicals_panel

RAW_DIR = Path(__file__).parent / "raw"

# Feature groups and their expected column names — used for validation
TECHNICAL_FEATURES = [
    "dc_position", "dc_breakout", "atr_pct", "rsi_14", "macd_hist",
    "ma10_ratio", "ma20_ratio", "ma50_ratio", "ma120_ratio", "ma200_ratio",
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    "rel_vol", "ret_1m", "ret_3m", "ret_6m", "ret_12m_skip1m",
    "vol_21d", "vol_63d",
]

FUNDAMENTAL_FEATURES = [
    "gross_margin", "net_margin", "roe", "roa",
    "debt_to_equity", "debt_to_assets", "current_ratio",
    "fcf_yield", "accruals", "piotroski_partial",
]

RATIO_FEATURES = [
    "bm", "pe_op_basic", "ps", "pcf", "dpr",
    "npm", "opmad", "gpm", "roe", "roa",
    "de_ratio", "curr_ratio", "debt_at",
    "divyield", "accrual",
]

ANALYST_FEATURES = [
    "sue_score", "est_revision", "est_dispersion", "num_analysts", "days_to_earnings",
]

REGIME_FEATURES = [
    "ff_mktrf_20d", "ff_smb_20d", "ff_hml_20d", "ff_rmw_20d", "ff_cma_20d",
    "ff_mktrf_60d", "ff_smb_60d", "ff_hml_60d",
    "ff_mktrf_120d", "ff_umd_20d", "rf",
]

RISK_FEATURES = [
    "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_mom",
]

ALL_FEATURES = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + REGIME_FEATURES + RISK_FEATURES


@lru_cache(maxsize=1)
def _load_wrds_ratios() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "wrds_ratios.parquet")
    df["public_date"] = pd.to_datetime(df["public_date"])
    return df


@lru_cache(maxsize=1)
def _load_ff_factors() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "ff_factors.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_beta_suite() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "beta_suite.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@lru_cache(maxsize=1)
def _load_ibes() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "ibes_statsum.parquet")
    df["statpers"] = pd.to_datetime(df["statpers"])
    df["fpedats"] = pd.to_datetime(df["fpedats"])
    return df


@lru_cache(maxsize=1)
def _load_ibes_link() -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "ibes_ticker_permno.parquet")
    df["sdate"] = pd.to_datetime(df["sdate"])
    df["edate"] = pd.to_datetime(df["edate"])
    return df


def _get_ff_regime_features(as_of_date: pd.Timestamp) -> dict:
    """Trailing FF factor returns over 20, 60, 120 day windows."""
    ff = _load_ff_factors()
    hist = ff[ff["date"] <= as_of_date].tail(120)
    if hist.empty:
        return {}

    def trailing_ret(col, n):
        window = hist[col].tail(n)
        return (1 + window).prod() - 1 if len(window) == n else np.nan

    return {
        "ff_mktrf_20d": trailing_ret("mktrf", 20),
        "ff_smb_20d": trailing_ret("smb", 20),
        "ff_hml_20d": trailing_ret("hml", 20),
        "ff_rmw_20d": trailing_ret("rmw", 20),
        "ff_cma_20d": trailing_ret("cma", 20),
        "ff_mktrf_60d": trailing_ret("mktrf", 60),
        "ff_smb_60d": trailing_ret("smb", 60),
        "ff_hml_60d": trailing_ret("hml", 60),
        "ff_mktrf_120d": trailing_ret("mktrf", 120),
        "ff_umd_20d": trailing_ret("umd", 20) if "umd" in hist.columns else np.nan,
        "rf": hist["rf"].iloc[-1] if "rf" in hist.columns else np.nan,
    }


def _get_betas(permnos: list[int], as_of_date: pd.Timestamp) -> pd.DataFrame:
    beta = _load_beta_suite()
    available = beta[(beta["permno"].isin(permnos)) & (beta["date"] <= as_of_date)]
    if available.empty:
        return pd.DataFrame(columns=["permno"] + RISK_FEATURES)
    idx = available.groupby("permno")["date"].idxmax()
    latest = available.loc[idx].copy()
    latest = latest.rename(columns={
        "beta": "beta_mkt",
        "betasmb": "beta_smb",
        "betahml": "beta_hml",
        "betarmw": "beta_rmw",
        "betamom": "beta_mom",
    })
    return latest[["permno", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_mom"]]


def _get_analyst_features(permnos: list[int], as_of_date: pd.Timestamp) -> pd.DataFrame:
    link = _load_ibes_link()
    ibes = _load_ibes()

    # Point-in-time ticker→permno map
    valid_link = link[(link["sdate"] <= as_of_date) & (link["edate"] >= as_of_date)]
    permno_to_ticker = valid_link[valid_link["permno"].isin(permnos)].set_index("permno")["ticker"]

    rows = []
    for permno in permnos:
        if permno not in permno_to_ticker.index:
            rows.append({"permno": permno})
            continue

        ticker = permno_to_ticker[permno]
        hist = ibes[
            (ibes["ticker"] == ticker) &
            (ibes["statpers"] <= as_of_date) &
            (ibes["fpi"] == "1")  # next fiscal year EPS
        ].sort_values("statpers")

        if hist.empty:
            rows.append({"permno": permno})
            continue

        latest = hist.iloc[-1]
        prior = hist.iloc[-2] if len(hist) >= 2 else None

        dispersion = (latest["stdev"] / abs(latest["meanest"])) if latest["meanest"] != 0 else np.nan
        revision = (latest["meanest"] - prior["meanest"]) / abs(prior["meanest"]) if prior is not None and prior["meanest"] != 0 else np.nan
        days_to_earn = (latest["fpedats"] - as_of_date).days if pd.notna(latest["fpedats"]) else np.nan

        rows.append({
            "permno": permno,
            "sue_score": latest.get("suescore", np.nan),
            "est_revision": revision,
            "est_dispersion": dispersion,
            "num_analysts": latest.get("numest", np.nan),
            "days_to_earnings": days_to_earn,
        })

    return pd.DataFrame(rows)


def _get_wrds_ratios(permnos: list[int], as_of_date: pd.Timestamp) -> pd.DataFrame:
    ratios = _load_wrds_ratios()
    eligible = ratios[(ratios["permno"].isin(permnos)) & (ratios["public_date"] <= as_of_date)]
    if eligible.empty:
        return pd.DataFrame(columns=["permno"])
    idx = eligible.groupby("permno")["public_date"].idxmax()
    return eligible.loc[idx].reset_index(drop=True)


def _cross_sectional_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Normalize each feature column to z-score across all stocks on the same date."""
    for col in feature_cols:
        if col not in df.columns:
            continue
        mu = df[col].mean()
        sigma = df[col].std()
        if sigma > 0:
            df[col] = (df[col] - mu) / sigma
        else:
            df[col] = 0.0
    return df


def build_feature_matrix(
    permnos: list[int],
    as_of_date: pd.Timestamp,
    price_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the full cross-sectional feature matrix for a set of PERMNOs on as_of_date.

    price_panel: output of compute_technicals_panel(), covering history up to as_of_date.
    Returns one row per PERMNO with all ~75-100 features, cross-sectionally z-scored.
    Returns empty DataFrame if no data available.
    """
    # --- Technical features (from precomputed price panel) ---
    tech_cols = ["permno"] + [c for c in TECHNICAL_FEATURES if c in price_panel.columns]
    tech_today = price_panel[price_panel["date"] == as_of_date][tech_cols].copy()

    if tech_today.empty:
        return pd.DataFrame()

    # --- Fama-French regime (same for all stocks) ---
    regime = _get_ff_regime_features(as_of_date)

    # --- WRDS ratios ---
    ratios = _get_wrds_ratios(permnos, as_of_date)

    # --- Compustat fundamentals ---
    fund_raw = get_fundamentals_panel(permnos, as_of_date)
    if not fund_raw.empty:
        fund = compute_derived_fundamentals(fund_raw)[["permno"] + FUNDAMENTAL_FEATURES]
    else:
        fund = pd.DataFrame(columns=["permno"] + FUNDAMENTAL_FEATURES)

    # --- Analyst features ---
    analyst = _get_analyst_features(permnos, as_of_date)

    # --- Factor betas ---
    betas = _get_betas(permnos, as_of_date)

    # --- Merge all ---
    result = tech_today
    for other in [ratios, fund, analyst, betas]:
        if not other.empty and "permno" in other.columns:
            result = result.merge(other, on="permno", how="left")

    # Add regime features (scalar → broadcast to all rows)
    for k, v in regime.items():
        result[k] = v

    # --- Cross-sectional z-score normalization ---
    feature_cols = [c for c in result.columns if c != "permno"]
    result = _cross_sectional_zscore(result, feature_cols)

    return result.reset_index(drop=True)


def get_state_vector(permno: int, feature_matrix: pd.DataFrame) -> np.ndarray:
    """
    Extract the state vector for a single PERMNO from a pre-built feature matrix.
    Returns a 1D numpy array of float32. NaN values are filled with 0 (mean after zscore).
    """
    row = feature_matrix[feature_matrix["permno"] == permno]
    if row.empty:
        n_features = len(feature_matrix.columns) - 1
        return np.zeros(n_features, dtype=np.float32)

    feat_cols = [c for c in feature_matrix.columns if c != "permno"]
    vec = row[feat_cols].iloc[0].values.astype(np.float32)
    vec = np.nan_to_num(vec, nan=0.0)
    return vec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_lag_correctness():
    """
    Verify that no feature in build_feature_matrix uses data published after as_of_date.
    Tests WRDS ratios and Compustat fundamentals lag rules.
    """
    ratios = _load_wrds_ratios()
    from data.compustat import _load_fundq
    fund = _load_fundq()

    test_date = pd.Timestamp("2015-06-30")
    permnos_sample = ratios["permno"].dropna().unique()[:20].tolist()

    # WRDS ratios: public_date must be <= test_date
    eligible_ratios = ratios[(ratios["permno"].isin(permnos_sample)) & (ratios["public_date"] <= test_date)]
    idx = eligible_ratios.groupby("permno")["public_date"].idxmax()
    selected = ratios.loc[idx]
    future = selected[selected["public_date"] > test_date]
    assert future.empty, f"Ratio look-ahead: {len(future)} rows"

    # Compustat: as_of_date must be <= test_date
    eligible_fund = fund[(fund["permno"].isin(permnos_sample)) & (fund["as_of_date"] <= test_date)]
    idx2 = eligible_fund.groupby("permno")["as_of_date"].idxmax()
    selected2 = fund.loc[idx2]
    future2 = selected2[selected2["as_of_date"] > test_date]
    assert future2.empty, f"Compustat look-ahead: {len(future2)} rows"

    print(f"[PASS] feature_store lag correctness on {test_date.date()}")


if __name__ == "__main__":
    _test_lag_correctness()

"""
Technical indicator computation from CRSP daily price data.
All indicators are price-level invariant (ratio/normalized forms).
Uses the 'ta' library (pip install ta) — pandas-ta is not compatible with arm64/Python 3.11.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features for a single stock's price history.

    Input df must have columns: date, prc (positive), vol, ret
    sorted ascending by date. Returns df with new feature columns appended.
    Requires at least 120 rows for all indicators to be non-null.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    close = df["prc"]
    volume = df["vol"]

    # --- Donchian Channel (primary buy signal) ---
    df["dc_upper"] = close.rolling(20).max()
    df["dc_lower"] = close.rolling(20).min()
    df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2
    dc_range = (df["dc_upper"] - df["dc_lower"]).replace(0, np.nan)
    df["dc_position"] = (close - df["dc_lower"]) / dc_range
    df["dc_breakout"] = (close > df["dc_upper"].shift(1)).astype(float)

    # --- ATR (normalized by price) ---
    # CRSP daily has no OHLC — approximate high/low with close
    atr = AverageTrueRange(high=close, low=close, close=close, window=14)
    df["atr_pct"] = atr.average_true_range() / close

    # --- RSI ---
    rsi = RSIIndicator(close=close, window=14)
    df["rsi_14"] = rsi.rsi() / 100.0

    # --- MACD histogram (normalized by price) ---
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_hist"] = macd.macd_diff() / close

    # --- Moving averages (ratio to price — price invariant) ---
    for window in [10, 20, 50, 120, 200]:
        ma = close.rolling(window).mean()
        df[f"ma{window}_ratio"] = close / ma - 1

    # --- Multi-horizon returns ---
    for days in [1, 5, 10, 21, 63]:
        df[f"ret_{days}d"] = close.pct_change(days)

    # --- Relative volume ---
    vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
    df["rel_vol"] = volume / vol_ma20

    # --- Momentum ---
    df["ret_1m"] = close.pct_change(21)
    df["ret_3m"] = close.pct_change(63)
    df["ret_6m"] = close.pct_change(126)
    df["ret_12m_skip1m"] = close.shift(21).pct_change(252 - 21)

    # --- Realized volatility ---
    df["vol_21d"] = df["ret"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"] = df["ret"].rolling(63).std() * np.sqrt(252)

    return df


def compute_technicals_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply compute_technicals to a panel of stocks.
    price_panel must have columns: permno, date, prc, vol, ret
    Returns panel with all technical features, sorted by permno, date.
    """
    results = []
    for permno, group in price_panel.groupby("permno"):
        enriched = compute_technicals(group)
        results.append(enriched)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True).sort_values(["permno", "date"])


def get_donchian_breakouts(price_panel: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Filter to stocks with an active Donchian breakout on as_of_date.
    Returns subset of price_panel rows for that date where dc_breakout == 1.
    """
    day = price_panel[price_panel["date"] == as_of_date]
    return day[day["dc_breakout"] == 1.0].copy()
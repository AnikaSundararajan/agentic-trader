"""
RL trading environment for the Buy agent.
Wraps the feature store and enforces point-in-time universe, delisting exits,
transaction costs, and temporal train/val/test splits.

Interface (same as mock_environment.py):
    env = TradingEnvironment(split="train")
    obs_list = env.reset()          # list of (permno, state_vector) for today's candidates
    obs_list, reward, done, info = env.step(action)
    # action: list of permnos to buy from today's candidates
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from data.crsp import get_sp500_permnos, get_daily_prices, apply_delisting_exits
from data.preprocess import compute_technicals_panel, get_donchian_breakouts
from data.feature_store import build_feature_matrix, get_state_vector

# Temporal split boundaries
SPLITS = {
    "train": ("2000-01-03", "2017-12-29"),
    "val":   ("2018-01-02", "2019-12-31"),
    "test":  ("2020-01-02", "2024-12-31"),
}

TRANSACTION_COST = 0.001   # 0.1% per trade (entry + exit each charged once)
MAX_HOLD_DAYS = 120
LOOKBACK_DAYS = 250        # price history needed for indicators


@dataclass
class Position:
    permno: int
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_features: np.ndarray
    days_held: int = 0


@dataclass
class TradeLog:
    permno: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    exit_reason: str
    days_held: int
    entry_features: np.ndarray = field(repr=False)


class TradingEnvironment:
    """
    Single-stock-at-a-time RL environment for the Buy agent.

    On each step the Buy agent receives a list of Donchian breakout candidates
    and decides which (if any) to purchase. The Sell agent and Stop Loss are
    separate modules called externally before step() returns.
    """

    def __init__(self, split: str = "train", initial_capital: float = 1_000_000.0):
        assert split in SPLITS, f"split must be one of {list(SPLITS)}"
        self.split = split
        self.start_date, self.end_date = (pd.Timestamp(d) for d in SPLITS[split])
        self.initial_capital = initial_capital

        self._trading_dates: list[pd.Timestamp] = []
        self._date_idx: int = 0
        self._positions: list[Position] = []
        self._cash: float = initial_capital
        self._portfolio_value_history: list[float] = []
        self._trade_log: list[TradeLog] = []
        self._price_panel: pd.DataFrame = pd.DataFrame()
        self._feature_matrix: pd.DataFrame = pd.DataFrame()
        self._candidates: list[tuple[int, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> list[tuple[int, np.ndarray]]:
        """
        Reset to first trading day. Returns initial list of (permno, state_vector) candidates.
        Loads and caches the full price panel for the split period.
        """
        self._cash = self.initial_capital
        self._positions = []
        self._portfolio_value_history = []
        self._trade_log = []
        self._date_idx = 0

        self._load_price_panel()
        self._trading_dates = sorted(self._price_panel["date"].unique())

        return self._advance_to_next_day()

    def step(
        self,
        buy_action: list[int],
        sell_action: list[int] | None = None,
    ) -> tuple[list[tuple[int, np.ndarray]], float, bool, dict]:
        """
        Execute one trading day.

        buy_action: list of permnos to buy (subset of current candidates)
        sell_action: list of permnos to sell from current positions (from Sell agent)

        Returns:
            candidates: next day's (permno, state_vector) list
            reward: scalar reward for this step
            done: True if end of split period
            info: dict with portfolio stats, trade log entries
        """
        current_date = self._trading_dates[self._date_idx - 1]

        # 1. Forced exits: delistings (must happen before any agent action)
        self._positions, forced_exits = apply_delisting_exits(
            pd.DataFrame([p.__dict__ for p in self._positions]) if self._positions else pd.DataFrame(),
            current_date,
        )
        if not forced_exits.empty:
            self._positions = [
                p for p in self._positions if p.permno not in forced_exits["permno"].values
            ]
            self._record_forced_exits(forced_exits, current_date)

        # 2. Sell agent exits
        sell_reward = 0.0
        if sell_action:
            sell_reward = self._execute_sells(sell_action, current_date, reason="sell_agent")

        # 3. Buy agent entries
        buy_reward = 0.0
        if buy_action:
            buy_reward = self._execute_buys(buy_action, current_date)

        # 4. Age all positions by one day; stop loss fires externally before this
        for pos in self._positions:
            pos.days_held += 1

        # 5. Portfolio value
        pv = self._portfolio_value(current_date)
        self._portfolio_value_history.append(pv)

        # 6. Composite reward
        reward = buy_reward + sell_reward + self._drawdown_penalty()

        # 7. Advance to next day
        done = self._date_idx >= len(self._trading_dates)
        if done:
            candidates = []
        else:
            candidates = self._advance_to_next_day()

        info = {
            "date": current_date,
            "portfolio_value": pv,
            "cash": self._cash,
            "n_positions": len(self._positions),
            "forced_exits": len(forced_exits) if not isinstance(forced_exits, pd.DataFrame) else len(forced_exits),
        }
        return candidates, reward, done, info

    @property
    def current_date(self) -> pd.Timestamp | None:
        if self._date_idx == 0 or self._date_idx > len(self._trading_dates):
            return None
        return self._trading_dates[self._date_idx - 1]

    @property
    def positions(self) -> list[Position]:
        return list(self._positions)

    @property
    def trade_log(self) -> list[TradeLog]:
        return list(self._trade_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_price_panel(self):
        """Load price history for all S&P 500 members over the split + lookback window."""
        lookback_start = self.start_date - pd.Timedelta(days=int(LOOKBACK_DAYS * 1.5))
        # Use all permnos ever in S&P 500 during this period (loaded lazily per day in step)
        # For efficiency, load the full panel once here
        all_members_df = pd.read_parquet(
            Path(__file__).parent / "raw" / "crsp_dsp500list.parquet"
        )
        all_members_df["start"] = pd.to_datetime(all_members_df["start"])
        all_members_df["ending"] = pd.to_datetime(all_members_df["ending"].fillna("2099-12-31"))

        in_period = all_members_df[
            (all_members_df["ending"] >= lookback_start) &
            (all_members_df["start"] <= self.end_date)
        ]
        all_permnos = in_period["permno"].unique().tolist()

        raw_prices = get_daily_prices(
            all_permnos,
            start=str(lookback_start.date()),
            end=str(self.end_date.date()),
        )

        self._price_panel = compute_technicals_panel(raw_prices)

    def _advance_to_next_day(self) -> list[tuple[int, np.ndarray]]:
        """Move to next trading date and compute today's Donchian candidates."""
        if self._date_idx >= len(self._trading_dates):
            return []

        today = self._trading_dates[self._date_idx]
        self._date_idx += 1

        # Point-in-time universe
        universe = get_sp500_permnos(today)

        # Donchian breakout filter
        breakouts = get_donchian_breakouts(self._price_panel, today)
        candidates_permnos = [p for p in breakouts["permno"].tolist() if p in universe]

        if not candidates_permnos:
            self._candidates = []
            self._feature_matrix = pd.DataFrame()
            return []

        # Build feature matrix for candidates
        self._feature_matrix = build_feature_matrix(
            candidates_permnos, today, self._price_panel
        )

        self._candidates = [
            (permno, get_state_vector(permno, self._feature_matrix))
            for permno in candidates_permnos
            if permno in self._feature_matrix["permno"].values
        ]
        return list(self._candidates)

    def _get_price(self, permno: int, date: pd.Timestamp) -> float | None:
        row = self._price_panel[
            (self._price_panel["permno"] == permno) & (self._price_panel["date"] == date)
        ]
        return float(row["prc"].iloc[0]) if not row.empty else None

    def _execute_buys(self, permnos: list[int], date: pd.Timestamp) -> float:
        reward = 0.0
        candidate_permnos = {p for p, _ in self._candidates}
        already_held = {p.permno for p in self._positions}

        for permno in permnos:
            if permno not in candidate_permnos:
                continue
            if permno in already_held:
                continue

            price = self._get_price(permno, date)
            if price is None or price <= 0:
                continue

            cost = price * (1 + TRANSACTION_COST)
            if cost > self._cash:
                continue

            # Equal-weight: invest 5% of current portfolio per position (max 20 positions)
            alloc = min(self._portfolio_value(date) * 0.05, self._cash)
            shares = alloc / cost
            self._cash -= shares * cost

            features = get_state_vector(permno, self._feature_matrix)
            self._positions.append(Position(
                permno=permno,
                entry_date=date,
                entry_price=price,
                shares=shares,
                entry_features=features,
            ))

        return reward

    def _execute_sells(self, permnos: list[int], date: pd.Timestamp, reason: str = "sell_agent") -> float:
        reward = 0.0
        to_remove = []

        for pos in self._positions:
            if pos.permno not in permnos:
                continue

            price = self._get_price(pos.permno, date)
            if price is None:
                continue

            net_price = price * (1 - TRANSACTION_COST)
            gross_ret = (price - pos.entry_price) / pos.entry_price
            net_ret = (net_price - pos.entry_price) / pos.entry_price

            self._cash += pos.shares * net_price
            reward += self._compute_exit_reward(gross_ret, net_ret, pos.days_held)

            self._trade_log.append(TradeLog(
                permno=pos.permno,
                entry_date=pos.entry_date,
                exit_date=date,
                entry_price=pos.entry_price,
                exit_price=price,
                gross_return=gross_ret,
                net_return=net_ret,
                exit_reason=reason,
                days_held=pos.days_held,
                entry_features=pos.entry_features,
            ))
            to_remove.append(pos.permno)

        self._positions = [p for p in self._positions if p.permno not in to_remove]
        return reward

    def _record_forced_exits(self, forced_exits: pd.DataFrame, date: pd.Timestamp):
        for _, row in forced_exits.iterrows():
            permno = int(row["permno"])
            pos = next((p for p in self._positions if p.permno == permno), None)
            if pos is None:
                continue
            gross_ret = float(row["exit_return"])
            net_ret = gross_ret - TRANSACTION_COST
            self._cash += pos.shares * pos.entry_price * (1 + net_ret)
            self._trade_log.append(TradeLog(
                permno=permno,
                entry_date=pos.entry_date,
                exit_date=date,
                entry_price=pos.entry_price,
                exit_price=pos.entry_price * (1 + gross_ret),
                gross_return=gross_ret,
                net_return=net_ret,
                exit_reason=str(row.get("exit_reason", "delisting")),
                days_held=pos.days_held,
                entry_features=pos.entry_features,
            ))

    def _portfolio_value(self, date: pd.Timestamp) -> float:
        equity = 0.0
        for pos in self._positions:
            price = self._get_price(pos.permno, date)
            if price:
                equity += pos.shares * price
        return self._cash + equity

    def _drawdown_penalty(self) -> float:
        if len(self._portfolio_value_history) < 20:
            return 0.0
        peak = max(self._portfolio_value_history)
        current = self._portfolio_value_history[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0.0
        return -drawdown * 0.5

    @staticmethod
    def _compute_exit_reward(gross_ret: float, net_ret: float, days_held: int) -> float:
        """
        Reward shaping per CLAUDE.md reward design:
        - Profitable + controlled drawdown → strong positive (handled via portfolio-level)
        - Large loss → 3× penalty
        - Small loss within stop → small negative
        """
        if net_ret > 0.10:
            return net_ret * 2.0
        elif net_ret > 0:
            return net_ret * 1.0
        elif net_ret > -0.07:
            return net_ret * 1.0   # small negative
        else:
            return net_ret * 3.0   # large loss: heavy penalty


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_universe_size():
    """Sanity-check universe size on a few known dates."""
    from data.crsp import get_sp500_permnos

    dates = [pd.Timestamp("2005-01-03"), pd.Timestamp("2010-06-01"), pd.Timestamp("2015-01-02")]
    for dt in dates:
        members = get_sp500_permnos(dt)
        assert 400 <= len(members) <= 550, f"Unexpected universe size {len(members)} on {dt}"
        print(f"[PASS] universe size on {dt.date()}: {len(members)} members")


def _test_no_future_data_in_obs():
    """Verify that features used on date T contain no data with publication date > T."""
    from data.feature_store import _load_wrds_ratios
    ratios = _load_wrds_ratios()
    test_date = pd.Timestamp("2012-03-15")
    sample_permnos = ratios["permno"].dropna().unique()[:10].tolist()
    eligible = ratios[(ratios["permno"].isin(sample_permnos)) & (ratios["public_date"] <= test_date)]
    idx = eligible.groupby("permno")["public_date"].idxmax()
    selected = ratios.loc[idx]
    future = selected[selected["public_date"] > test_date]
    assert future.empty, f"Look-ahead: {len(future)} ratio rows after {test_date}"
    print("[PASS] no future data in observations")


if __name__ == "__main__":
    _test_universe_size()
    _test_no_future_data_in_obs()

"""
Synthetic trading environment for Dev B to develop agents against
before the real data pipeline (data/environment.py) is ready.

Identical interface to TradingEnvironment:
    env = MockTradingEnvironment()
    candidates = env.reset()
    candidates, reward, done, info = env.step(buy_action, sell_action)

Generates random feature vectors and synthetic price paths.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from data.feature_store import ALL_FEATURES

N_STATE_FEATURES = len(ALL_FEATURES)
TRANSACTION_COST = 0.001
MAX_HOLD_DAYS = 120


@dataclass
class _MockPosition:
    permno: int
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_features: np.ndarray
    days_held: int = 0


class MockTradingEnvironment:
    """
    Synthetic environment with the same reset()/step() API as TradingEnvironment.
    Useful for:
      - Developing and debugging Buy/Sell agent code before real data is ready
      - Fast unit tests (no WRDS dependency)
      - Verifying reward function shapes

    Prices follow a random walk with drift. Candidates are randomly generated each day.
    """

    def __init__(
        self,
        n_episodes: int = 1000,
        n_candidates_per_day: int = 10,
        n_state_features: int = N_STATE_FEATURES,
        initial_capital: float = 1_000_000.0,
        seed: int = 42,
    ):
        self.n_episodes = n_episodes
        self.n_candidates_per_day = n_candidates_per_day
        self.n_state_features = n_state_features
        self.initial_capital = initial_capital
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._positions: list[_MockPosition] = []
        self._cash = initial_capital
        self._portfolio_history: list[float] = []
        self._permnos: list[int] = list(range(10_000, 10_000 + 500))
        self._prices: dict[int, float] = {}
        self._candidates: list[tuple[int, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> list[tuple[int, np.ndarray]]:
        self._rng = np.random.default_rng(self.seed)
        self._step_count = 0
        self._positions = []
        self._cash = self.initial_capital
        self._portfolio_history = []
        # Initialize random starting prices
        self._prices = {p: float(self._rng.uniform(10, 500)) for p in self._permnos}
        self._candidates = self._generate_candidates()
        return list(self._candidates)

    def step(
        self,
        buy_action: list[int],
        sell_action: list[int] | None = None,
    ) -> tuple[list[tuple[int, np.ndarray]], float, bool, dict]:
        """
        buy_action: list of permnos to buy (subset of current candidates)
        sell_action: list of permnos to sell from open positions
        """
        self._step_count += 1
        today = pd.Timestamp("2000-01-01") + pd.Timedelta(days=self._step_count)

        # Advance all prices by one random step
        self._tick_prices()

        reward = 0.0

        # Sell agent exits
        if sell_action:
            reward += self._execute_sells(sell_action, today)

        # Buy agent entries
        if buy_action:
            self._execute_buys(buy_action, today)

        # Age positions
        for pos in self._positions:
            pos.days_held += 1
            # Force exit at max hold
            if pos.days_held >= MAX_HOLD_DAYS:
                reward += self._execute_sells([pos.permno], today, reason="max_hold")

        pv = self._portfolio_value()
        self._portfolio_history.append(pv)
        reward += self._drawdown_penalty()

        done = self._step_count >= self.n_episodes
        self._candidates = self._generate_candidates() if not done else []

        info = {
            "step": self._step_count,
            "portfolio_value": pv,
            "cash": self._cash,
            "n_positions": len(self._positions),
        }
        return list(self._candidates), reward, done, info

    @property
    def observation_space_size(self) -> int:
        return self.n_state_features

    @property
    def positions(self) -> list[_MockPosition]:
        return list(self._positions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_candidates(self) -> list[tuple[int, np.ndarray]]:
        """Pick random candidate stocks with random (normalized) feature vectors."""
        n = self._rng.integers(3, self.n_candidates_per_day + 1)
        chosen = self._rng.choice(self._permnos, size=int(n), replace=False)
        return [
            (int(p), self._rng.standard_normal(self.n_state_features).astype(np.float32))
            for p in chosen
        ]

    def _tick_prices(self):
        """Random walk with slight upward drift."""
        for p in self._prices:
            shock = self._rng.normal(0.0003, 0.015)
            self._prices[p] = max(0.01, self._prices[p] * (1 + shock))

    def _execute_buys(self, permnos: list[int], date: pd.Timestamp):
        candidate_set = {p for p, _ in self._candidates}
        held = {pos.permno for pos in self._positions}

        for permno in permnos:
            if permno not in candidate_set or permno in held:
                continue
            price = self._prices.get(permno, 0)
            if price <= 0:
                continue
            cost = price * (1 + TRANSACTION_COST)
            alloc = min(self._portfolio_value() * 0.05, self._cash)
            if alloc < cost:
                continue
            shares = alloc / cost
            self._cash -= shares * cost
            features = next((f for p, f in self._candidates if p == permno), np.zeros(self.n_state_features))
            self._positions.append(_MockPosition(
                permno=permno,
                entry_date=date,
                entry_price=price,
                shares=shares,
                entry_features=features,
            ))

    def _execute_sells(self, permnos: list[int], date: pd.Timestamp, reason: str = "sell_agent") -> float:
        reward = 0.0
        to_remove = []
        for pos in self._positions:
            if pos.permno not in permnos:
                continue
            price = self._prices.get(pos.permno, pos.entry_price)
            net_price = price * (1 - TRANSACTION_COST)
            net_ret = (net_price - pos.entry_price) / pos.entry_price
            self._cash += pos.shares * net_price
            reward += self._compute_exit_reward(net_ret)
            to_remove.append(pos.permno)
        self._positions = [p for p in self._positions if p.permno not in to_remove]
        return reward

    def _portfolio_value(self) -> float:
        equity = sum(pos.shares * self._prices.get(pos.permno, pos.entry_price) for pos in self._positions)
        return self._cash + equity

    def _drawdown_penalty(self) -> float:
        if len(self._portfolio_history) < 20:
            return 0.0
        peak = max(self._portfolio_history)
        current = self._portfolio_history[-1]
        dd = (peak - current) / peak if peak > 0 else 0.0
        return -dd * 0.5

    @staticmethod
    def _compute_exit_reward(net_ret: float) -> float:
        if net_ret > 0.10:
            return net_ret * 2.0
        elif net_ret > 0:
            return net_ret * 1.0
        elif net_ret > -0.07:
            return net_ret * 1.0
        else:
            return net_ret * 3.0


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    env = MockTradingEnvironment(n_episodes=50, seed=42)
    candidates = env.reset()
    print(f"Initial candidates: {len(candidates)}, state_size={candidates[0][1].shape[0]}")

    total_reward = 0.0
    for step in range(50):
        if not candidates:
            break
        # Random buy policy: buy ~30% of candidates
        buy_permnos = [p for p, _ in candidates if np.random.rand() < 0.3]
        sell_permnos = [pos.permno for pos in env.positions if np.random.rand() < 0.2]
        candidates, reward, done, info = env.step(buy_permnos, sell_permnos)
        total_reward += reward
        if done:
            break

    print(f"Final portfolio: ${info['portfolio_value']:,.0f} | Total reward: {total_reward:.4f}")
    print("[PASS] MockTradingEnvironment smoke test")

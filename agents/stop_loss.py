"""
Stop loss module — fires before the Sell agent on every step.
Three layers:
  1. ATR-based stop: 2x ATR adjusted by beta (dynamic risk sizing)
  2. Hard floor: 7% loss from entry, no exceptions
  3. Trailing stop: activates after 5% gain, trails by 2x ATR from peak

All rules are deterministic — no ML involved.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class StopLossState:
    """Per-position state tracked by the stop loss module."""
    permno: int
    entry_price: float
    atr_at_entry: float         # ATR value on the day of entry
    beta: float                 # market beta at entry (default 1.0)
    peak_price: float = field(init=False)
    trailing_active: bool = field(default=False, init=False)

    def __post_init__(self):
        self.peak_price = self.entry_price

    @property
    def stop_distance(self) -> float:
        """ATR-based stop distance, widened for high-beta stocks."""
        beta_adj = max(1.0, self.beta)          # never tighten below 1x beta
        return self.atr_at_entry * 2.0 * beta_adj

    @property
    def hard_floor(self) -> float:
        return self.entry_price * (1 - 0.07)

    def trailing_stop_price(self) -> float | None:
        """Returns trailing stop price if active, else None."""
        if not self.trailing_active:
            return None
        return self.peak_price - self.stop_distance

    def update_peak(self, current_price: float):
        """Call each day to ratchet the peak price up and activate trailing stop."""
        if current_price > self.peak_price:
            self.peak_price = current_price
        gain = (self.peak_price - self.entry_price) / self.entry_price
        if gain >= 0.05:
            self.trailing_active = True


class StopLossManager:
    """
    Manages stop loss state for all open positions.
    Call check() each day before the Sell agent runs.
    """

    def __init__(self):
        self._states: dict[int, StopLossState] = {}

    def register(self, permno: int, entry_price: float, atr: float, beta: float = 1.0):
        """Register a new position when the Buy agent enters."""
        self._states[permno] = StopLossState(
            permno=permno,
            entry_price=entry_price,
            atr_at_entry=atr,
            beta=beta,
        )

    def remove(self, permno: int):
        """Deregister a position after it's been closed."""
        self._states.pop(permno, None)

    def check(self, current_prices: dict[int, float]) -> list[int]:
        """
        Given today's prices, return list of PERMNOs that must be exited.
        Updates trailing stop peaks before checking triggers.
        """
        exits = []
        for permno, state in self._states.items():
            price = current_prices.get(permno)
            if price is None:
                continue

            state.update_peak(price)

            triggered, reason = self._is_triggered(state, price)
            if triggered:
                exits.append(permno)

        return exits

    def exit_reason(self, permno: int, current_price: float) -> str:
        """Return human-readable reason for the stop trigger."""
        state = self._states.get(permno)
        if state is None:
            return "unknown"
        _, reason = self._is_triggered(state, current_price)
        return reason

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_triggered(state: StopLossState, price: float) -> tuple[bool, str]:
        # 1. Hard floor — checked first, highest priority
        if price <= state.hard_floor:
            return True, "stop_hard_floor"

        # 2. ATR-based initial stop (before trailing activates)
        if not state.trailing_active:
            atr_stop = state.entry_price - state.stop_distance
            if price <= atr_stop:
                return True, "stop_atr"

        # 3. Trailing stop
        trailing = state.trailing_stop_price()
        if trailing is not None and price <= trailing:
            return True, "stop_trailing"

        return False, ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_hard_floor():
    mgr = StopLossManager()
    mgr.register(permno=1, entry_price=100.0, atr=2.0, beta=1.0)
    # Price drops 7.1% — should trigger hard floor
    hits = mgr.check({1: 92.8})
    assert 1 in hits, "Hard floor should trigger at -7.1%"
    print("[PASS] hard floor trigger")


def _test_atr_stop():
    mgr = StopLossManager()
    # entry=100, atr=3, beta=1 → stop_distance=6 → atr_stop=94
    mgr.register(permno=2, entry_price=100.0, atr=3.0, beta=1.0)
    hits = mgr.check({2: 93.5})   # below 94
    assert 2 in hits, "ATR stop should trigger at 93.5"
    hits2 = mgr.check({2: 94.5})  # above 94
    assert 2 not in hits2, "ATR stop should not trigger at 94.5"
    print("[PASS] ATR stop trigger")


def _test_trailing_stop():
    mgr = StopLossManager()
    # entry=100, atr=2, beta=1 → stop_distance=4
    mgr.register(permno=3, entry_price=100.0, atr=2.0, beta=1.0)

    # Price rises to 106 (+6%) — trailing activates, peak=106, trailing_stop=102
    mgr.check({3: 106.0})
    assert mgr._states[3].trailing_active, "Trailing should activate after +5%"
    assert mgr._states[3].peak_price == 106.0

    # Price falls to 101.5 — below trailing stop of 102
    hits = mgr.check({3: 101.5})
    assert 3 in hits, "Trailing stop should trigger at 101.5 (stop=102)"
    print("[PASS] trailing stop trigger")


def _test_beta_adjustment():
    mgr = StopLossManager()
    # beta=2 → stop_distance = atr*2*beta = 2*2*2 = 8 → atr_stop = 92
    # hard_floor = 100 * 0.93 = 93.0  (so test at 93.5 to stay above hard floor)
    mgr.register(permno=4, entry_price=100.0, atr=2.0, beta=2.0)
    # 93.5 > atr_stop(92) and > hard_floor(93) — should not trigger
    hits = mgr.check({4: 93.5})
    assert 4 not in hits, "High-beta widens stop, 93.5 should not trigger"
    # 91.5 < atr_stop(92) — should trigger
    hits2 = mgr.check({4: 91.5})
    assert 4 in hits2, "High-beta ATR stop should trigger at 91.5"
    print("[PASS] beta adjustment widens stop")


if __name__ == "__main__":
    _test_hard_floor()
    _test_atr_stop()
    _test_trailing_stop()
    _test_beta_adjustment()
    print("\nAll stop loss tests passed.")

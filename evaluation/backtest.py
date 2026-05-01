"""
Walk-forward backtest on the test set (2020-2024).
Run: python -m evaluation.backtest

Runs the full Buy → Stop Loss → Sell pipeline with real TradingEnvironment.
Prints a PerformanceReport and saves results to logs/backtest_results.csv.

IMPORTANT: Run on val set first to tune agents. Run test set only once.
"""

import csv
import numpy as np
import torch
from pathlib import Path

from data.environment import TradingEnvironment
from data.mock_environment import MockTradingEnvironment
from agents.buy_agent import BuyAgent
from agents.sell_agent import SellAgent, build_sell_state
from agents.stop_loss import StopLossManager
from evaluation.metrics import compute_metrics, print_report

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_agents(device: str = "cpu") -> tuple[BuyAgent, SellAgent]:
    """Load best checkpoints for both agents."""
    # Probe feature size
    probe = MockTradingEnvironment(n_episodes=5, seed=0)
    candidates = probe.reset()
    n_features = candidates[0][1].shape[0] if candidates else 75

    buy_agent = BuyAgent(n_features=n_features, device=device)
    buy_ckpt = CKPT_DIR / "buy" / "latest.pt"
    if buy_ckpt.exists():
        buy_agent.load(buy_ckpt)
        print(f"Loaded buy agent: {buy_ckpt}")
    else:
        print("WARNING: No buy agent checkpoint found, using random weights")

    sell_agent = SellAgent(n_base_features=n_features, device=device)
    sell_ckpt = CKPT_DIR / "sell" / "latest.pt"
    if sell_ckpt.exists():
        sell_agent.load(sell_ckpt)
        print(f"Loaded sell agent: {sell_ckpt}")
    else:
        print("WARNING: No sell agent checkpoint found, using random weights")

    buy_agent.network.eval()
    sell_agent.network.eval()
    return buy_agent, sell_agent


def run_backtest(split: str = "val", use_mock: bool = False) -> dict:
    """
    Run full backtest on a given split.

    split: 'train' | 'val' | 'test'
    use_mock: use MockTradingEnvironment (for smoke testing without WRDS data)

    Returns dict with portfolio_values, trade_log, and PerformanceReport.
    """
    assert split in ("train", "val", "test"), f"Invalid split: {split}"
    if split == "test":
        print("WARNING: Running on TEST set. Do this only once after val tuning is complete.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    buy_agent, sell_agent = load_agents(device)
    stop_mgr = StopLossManager()

    if use_mock:
        env = MockTradingEnvironment(n_episodes=500, seed=777)
        n_base_features = buy_agent.n_features
    else:
        env = TradingEnvironment(split=split)
        n_base_features = buy_agent.n_features

    candidates = env.reset()
    portfolio_values = []
    all_stop_exits_log = []

    print(f"\nRunning backtest on {split} split {'(mock)' if use_mock else '(real WRDS)'}")
    step = 0

    while True:
        step += 1

        # 1. Stop loss check (fires first)
        if use_mock:
            price_map = {pos.permno: env._prices.get(pos.permno, pos.entry_price) for pos in env.positions}
        else:
            price_map = {}
            for pos in env.positions:
                p = env._get_price(pos.permno, env.current_date)
                if p:
                    price_map[pos.permno] = p

        stop_exits = stop_mgr.check(price_map)
        for p in stop_exits:
            reason = stop_mgr.exit_reason(p, price_map.get(p, 0))
            all_stop_exits_log.append({"permno": p, "reason": reason, "step": step})
            stop_mgr.remove(p)

        # 2. Sell agent on remaining positions
        if use_mock:
            position_states = _build_position_states_mock(env, n_base_features)
        else:
            position_states = _build_position_states_real(env, n_base_features)

        position_states_no_stop = [(p, s) for p, s in position_states if p not in stop_exits]
        sell_permnos = sell_agent.act(position_states_no_stop, deterministic=True)

        all_exits = list(set(stop_exits + sell_permnos))

        # 3. Buy agent on candidates
        buy_permnos = buy_agent.act(candidates, deterministic=True)

        # Register new buys with stop loss
        for p in buy_permnos:
            price = price_map.get(p, 100.0)
            # ATR: use atr_pct feature if available, else 2% proxy
            atr = price * 0.02
            beta = 1.0
            stop_mgr.register(p, price, atr=atr, beta=beta)

        # 4. Step environment
        candidates, reward, done, info = env.step(buy_permnos, all_exits)
        portfolio_values.append(info.get("portfolio_value", 0))

        if step % 50 == 0:
            print(f"  Step {step:4d} | PV=${info.get('portfolio_value', 0):,.0f} | positions={info.get('n_positions', 0)}")

        if done:
            break

    trade_log = env.trade_log if hasattr(env, "trade_log") else []
    report = compute_metrics(portfolio_values, trade_log)

    return {
        "portfolio_values": portfolio_values,
        "trade_log": trade_log,
        "report": report,
        "stop_exits_log": all_stop_exits_log,
    }


def save_results(results: dict, split: str):
    """Save trade log and portfolio values to CSV."""
    pv_path = LOG_DIR / f"backtest_{split}_portfolio.csv"
    with open(pv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "portfolio_value"])
        for i, pv in enumerate(results["portfolio_values"]):
            w.writerow([i + 1, pv])

    trades_path = LOG_DIR / f"backtest_{split}_trades.csv"
    trade_log = results.get("trade_log", [])
    if trade_log:
        with open(trades_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["permno", "entry_date", "exit_date", "entry_price", "exit_price",
                         "gross_return", "net_return", "exit_reason", "days_held"])
            for t in trade_log:
                w.writerow([
                    getattr(t, "permno", ""),
                    getattr(t, "entry_date", ""),
                    getattr(t, "exit_date", ""),
                    getattr(t, "entry_price", ""),
                    getattr(t, "exit_price", ""),
                    getattr(t, "gross_return", ""),
                    getattr(t, "net_return", ""),
                    getattr(t, "exit_reason", ""),
                    getattr(t, "days_held", ""),
                ])

    print(f"\nSaved portfolio values → {pv_path}")
    print(f"Saved trade log       → {trades_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_position_states_mock(env, n_base_features: int) -> list[tuple[int, np.ndarray]]:
    result = []
    for pos in env.positions:
        current_price = env._prices.get(pos.permno, pos.entry_price)
        peak_price = max(current_price, pos.entry_price)
        base = np.zeros(n_base_features, dtype=np.float32)
        aug = build_sell_state(base, pos.entry_price, current_price, pos.days_held, peak_price, False)
        result.append((pos.permno, aug))
    return result


def _build_position_states_real(env, n_base_features: int) -> list[tuple[int, np.ndarray]]:
    from data.feature_store import get_state_vector
    result = []
    for pos in env.positions:
        current_price = env._get_price(pos.permno, env.current_date) or pos.entry_price
        peak_price = max(current_price, pos.entry_price)
        # Use pre-built feature matrix if available, else zero vector
        if not env._feature_matrix.empty and pos.permno in env._feature_matrix["permno"].values:
            base = get_state_vector(pos.permno, env._feature_matrix)
        else:
            base = np.zeros(n_base_features, dtype=np.float32)
        aug = build_sell_state(base, pos.entry_price, current_price, pos.days_held, peak_price, False)
        result.append((pos.permno, aug))
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--mock", action="store_true", help="Use mock environment (no WRDS required)")
    args = parser.parse_args()

    results = run_backtest(split=args.split, use_mock=args.mock)
    print_report(results["report"], label=args.split)
    save_results(results, args.split)

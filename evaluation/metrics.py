"""
Performance metrics for backtest results.
All metrics operate on a list of daily portfolio values and a trade log.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    sharpe_ratio: float
    sortino_ratio: float
    cagr: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_hold_days: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    exit_reason_breakdown: dict[str, int]
    survivorship_bias_estimate: float | None = None


def compute_metrics(
    portfolio_values: list[float],
    trade_log: list,
    benchmark_values: list[float] | None = None,
    trading_days_per_year: int = 252,
) -> PerformanceReport:
    """
    Compute full performance report from portfolio value series and trade log.

    portfolio_values: daily portfolio values, starting from initial capital
    trade_log: list of TradeLog dataclass instances (from environment.py)
    benchmark_values: optional daily S&P 500 values (same length) for survivorship check
    """
    pv = np.array(portfolio_values, dtype=np.float64)
    daily_returns = np.diff(pv) / pv[:-1]

    sharpe = _sharpe(daily_returns, trading_days_per_year)
    sortino = _sortino(daily_returns, trading_days_per_year)
    cagr = _cagr(pv, trading_days_per_year)
    max_dd = _max_drawdown(pv)
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Trade-level stats
    net_returns = [t.net_return for t in trade_log if hasattr(t, "net_return")]
    wins = [r for r in net_returns if r > 0]
    losses = [r for r in net_returns if r <= 0]

    win_rate = len(wins) / len(net_returns) if net_returns else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else np.nan

    hold_days = [t.days_held for t in trade_log if hasattr(t, "days_held")]
    avg_hold = float(np.mean(hold_days)) if hold_days else 0.0

    exit_reasons: dict[str, int] = {}
    for t in trade_log:
        reason = getattr(t, "exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Survivorship bias estimate: universe return vs benchmark
    surv_bias = None
    if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
        bv = np.array(benchmark_values, dtype=np.float64)
        universe_cagr = cagr
        bench_daily = np.diff(bv) / bv[:-1]
        bench_cagr = _cagr(bv, trading_days_per_year)
        surv_bias = universe_cagr - bench_cagr

    return PerformanceReport(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        cagr=cagr,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        avg_hold_days=avg_hold,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        n_trades=len(net_returns),
        exit_reason_breakdown=exit_reasons,
        survivorship_bias_estimate=surv_bias,
    )


def print_report(report: PerformanceReport, label: str = ""):
    header = f"=== Performance Report{': ' + label if label else ''} ==="
    print(header)
    print(f"  Sharpe Ratio:      {report.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:     {report.sortino_ratio:.3f}")
    print(f"  CAGR:              {report.cagr:.2%}")
    print(f"  Max Drawdown:      {report.max_drawdown:.2%}")
    print(f"  Calmar Ratio:      {report.calmar_ratio:.3f}")
    print(f"  Win Rate:          {report.win_rate:.2%}")
    print(f"  Avg Hold Days:     {report.avg_hold_days:.1f}")
    print(f"  Avg Win:           {report.avg_win:.2%}")
    print(f"  Avg Loss:          {report.avg_loss:.2%}")
    print(f"  Profit Factor:     {report.profit_factor:.2f}")
    print(f"  N Trades:          {report.n_trades}")
    print(f"  Exit Reasons:      {report.exit_reason_breakdown}")
    if report.survivorship_bias_estimate is not None:
        print(f"  Survivorship Bias: {report.survivorship_bias_estimate:.2%} (vs benchmark)")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe(daily_returns: np.ndarray, ann: int) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(ann))


def _sortino(daily_returns: np.ndarray, ann: int) -> float:
    if len(daily_returns) < 2:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside) + 1e-10 if len(downside) > 0 else 1e-10
    return float(np.mean(daily_returns) / downside_std * np.sqrt(ann))


def _cagr(pv: np.ndarray, ann: int) -> float:
    if len(pv) < 2 or pv[0] <= 0:
        return 0.0
    n_years = len(pv) / ann
    return float((pv[-1] / pv[0]) ** (1 / n_years) - 1)


def _max_drawdown(pv: np.ndarray) -> float:
    if len(pv) < 2:
        return 0.0
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    return float(drawdown.min())

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning trading system that mimics professional trader decision-making, based on "Pro Trader RL" (Gu et al., 2024, Expert Systems with Applications). Uses institutional-grade WRDS data.

**Four-module pipeline:** Buy Agent (which breakouts to take) → Sell Agent (when to exit) → Stop Loss (hard risk rules) → Environment (simulates market with point-in-time data).

Buy and Sell agents are **separate** PPO-trained neural networks — never merge them into one.

## Common Commands

```bash
# Install dependencies
pip install wrds pandas pandas-ta torch pyarrow scikit-learn matplotlib tqdm

# Download all WRDS data (run once, ~30-60 min)
python -m data.wrds_download

# Run full pipeline
python main.py

# Train only buy agent
python -m training.train_buy

# Run backtest on test set
python -m evaluation.backtest
```

## Architecture

```
data/         → WRDS downloads, preprocessing, feature assembly, RL environment
agents/       → Buy agent, Sell agent, stop loss rules
training/     → PPO training loops
evaluation/   → Backtest, metrics, benchmark comparison
```

**Build order** — test each module independently before integrating:
1. `data/crsp.py` — constituent lookups + delisting handling
2. `data/compustat.py` — CCM link + lag logic
3. `data/preprocess.py` — technical indicators
4. `data/feature_store.py` — unified assembly (**write lag-correctness tests**)
5. `data/environment.py` — point-in-time selection + delisting exits (**write tests**)
6. `agents/stop_loss.py` — rule-based, simplest module
7. `agents/buy_agent.py` + `training/train_buy.py`
8. `agents/sell_agent.py` + `training/train_sell.py`
9. `evaluation/` — backtest + metrics + benchmark comparison
10. Enhancement data layers (one at a time, measure improvement before adding the next)

## Data Sources (WRDS)

All data is keyed on **PERMNO** (CRSP permanent ID). Never use tickers as join keys — they change over time.

| Source | Purpose | Lag Rule |
|--------|---------|----------|
| CRSP `dsf` | Daily prices, returns, volume | Same-day |
| CRSP `dsedelist` | Delisting codes + returns | Event-driven |
| CRSP `dsp500list` | S&P 500 constituent history | Point-in-time lookup |
| Compustat `fundq` via CCM | Quarterly financials | Use `rdq`; fallback `datadate + 90d` if null |
| WRDS Financial Ratios | 70+ pre-computed ratios | Use `public_date` column |
| I/B/E/S `statsum_epsus` | Analyst estimates, earnings surprises | Use `statpers` |
| Fama-French `fivefactors_daily` | Market regime factors | Same-day |
| WRDS Beta Suite | Stock-level factor betas | Same-day |

Enhancement datasets (add after base works): RavenPack, MarketPsych, Short Volume, OptionMetrics.

## Critical Rules — Backtest Validity

**Violating any of these invalidates the entire backtest.**

1. **No look-ahead bias.** Every feature must be available BEFORE the date it's used. Fundamental data uses report date lag. Ratios use `public_date`. If `rdq` is null, assume 90-day lag from `datadate`.

2. **Point-in-time universe.** Only trade stocks that were in the S&P 500 on that specific date per `crsp.dsp500list`. Never use current index membership for historical dates.

3. **Delisting returns are mandatory.** When a held stock delists, force-exit using CRSP's `dlret`. By `dlstcd`: 200-299 = acquisition, 400-499 = liquidation, 500-599 = dropped. Fill missing returns per Shumway (1997): acquisitions → 0%, failures → -40%.

4. **Temporal splits only.** Train: 2000–2017, Val: 2018–2019, Test: 2020–2024. Never shuffle dates.

5. **Transaction costs in every trade.** 0.1% per trade (covers slippage + commission). Bake into both reward and backtest.

## Feature Store (`data/feature_store.py`)

State vector (~75-100 features), normalized cross-sectionally (z-score across stocks on the same date). All features must be price-level invariant.

- **Technical (~25):** Donchian Channel position/breakouts, RSI, MACD, MAs, ATR%, relative volume, multi-horizon returns
- **Fundamental (~20):** PE, P/B, ROE, debt/equity, FCF yield, revenue growth, accruals, Piotroski — all lagged
- **Analyst (~10):** SUE, estimate revisions, dispersion, coverage, recommendation changes, days to earnings
- **Market regime (~15):** trailing 20/60/120d Fama-French factor returns, risk-free rate
- **Risk (~5):** market beta, size beta, value beta, momentum beta

## Donchian Channel (Primary Buy Signal)

The Buy agent only considers stocks where `close > 20-day upper band`. This filters ~500 S&P 500 stocks down to ~5-20 candidates per day. The agent's job is to decide which breakouts are worth taking.

## Reward Design

- Profitable + controlled drawdown → strong positive
- Profitable + sloppy risk → weak positive
- Small loss within stop → small negative
- Large loss → heavy penalty (3× multiplier)
- Correctly skipped bad trade → small positive
- Missed good trade → no penalty (conservative bias is intentional)

## Agent Behavior Expectations

- **Buy agent:** 1-3 buys/day from 10-20 signals is healthy. Buying everything = broken reward.
- **Sell agent:** Should be patient. Closing after 1-2 days consistently = broken reward. Max hold = 120 trading days.
- **Stop loss fires before Sell agent:** ATR-based (2×, adjusted by beta), 7% hard floor, trailing after 5% gain.

## Code Conventions

- Python 3.10+, PyTorch for neural networks
- `pandas` + `pyarrow` for data; parquet format for storage
- `pandas-ta` for technical indicators
- `wrds` library for data access via `db.raw_sql()`
- Set seeds everywhere: `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)`
- Log every trade: permno, dates, prices, return, exit_reason, days_held, features at entry
- Save model checkpoints every 100 episodes, keep best by validation Sharpe

## Debugging Guide

| Symptom | Likely Cause |
|---------|-------------|
| Backtest looks too good | Look-ahead bias in features or survivorship bias in universe |
| Agent buys everything | Reward needs more penalty for bad risk/reward |
| Agent never buys | Epsilon too low or reward too punishing |
| Sharpe collapses train→val | Overfitting — reduce features or add regularization |

Compare universe returns vs `^GSPC` — the difference estimates survivorship bias.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning trading system that mimics professional trader decision-making, based on "Pro Trader RL" (Gu et al., 2024, Expert Systems with Applications). Uses institutional-grade WRDS data.

**Four-module pipeline:** Buy Agent (which breakouts to take) → Sell Agent (when to exit) → Stop Loss (hard risk rules) → Environment (simulates market with point-in-time data).

Buy and Sell agents are **separate** PPO-trained neural networks — never merge them into one.

## Common Commands

```bash
# Install dependencies (use python3.11 — pandas-ta requires pandas<2.2)
python3.11 -m pip install wrds "pandas<2.2" torch pyarrow scikit-learn matplotlib tqdm
python3.11 -m pip install git+https://github.com/twopirllc/pandas-ta.git@development

# Download all WRDS data (run once, ~30-60 min; prompts for WRDS credentials)
python3.11 -m data.wrds_download

# Run full pipeline
python3.11 main.py

# Train buy agent (uses MockTradingEnvironment by default)
python3.11 -m training.train_buy

# Train sell agent (loads buy checkpoint, freezes it, trains sell)
python3.11 -m training.train_sell

# Smoke test backtest with no WRDS required
python3.11 -m evaluation.backtest --mock

# Backtest on val set (run this before test set)
python3.11 -m evaluation.backtest --split val

# Backtest on test set — run only once after val tuning is done
python3.11 -m evaluation.backtest --split test

# Run stop loss unit tests
python3.11 -m agents.stop_loss

# Run CRSP lag/delisting tests
python3.11 -m data.crsp

# Run Compustat lag tests
python3.11 -m data.compustat

# Check WRDS schema (run if download queries fail)
python3.11 -m data.wrds_schema_check
```

## Architecture

```
data/         → WRDS downloads, preprocessing, feature assembly, RL environment
agents/       → Buy agent, Sell agent, stop loss rules
training/     → PPO training loops
evaluation/   → Backtest, metrics, benchmark comparison
```

### Implementation Status

**Dev A — Data Pipeline (complete):**
- `data/wrds_download.py` — downloads all 8 WRDS sources to `data/raw/*.parquet`
- `data/crsp.py` — point-in-time S&P 500 universe, delisting exits with Shumway fills
- `data/compustat.py` — CCM link, `rdq`/90-day lag, derived accounting ratios
- `data/preprocess.py` — Donchian, RSI, MACD, ATR%, MAs, multi-horizon returns (all price-invariant)
- `data/feature_store.py` — joins all sources into ~75-100 feature cross-sectional z-scored matrix
- `data/environment.py` — full RL env: `reset()` / `step(buy_action, sell_action)` API, point-in-time universe, breakout filter, delisting exits, 0.1% txn cost
- `data/mock_environment.py` — identical interface, synthetic data — Dev B starts here

**Dev B — Agents/Training/Evaluation (complete):**
- `agents/stop_loss.py` — rule-based: ATR-based (2×, beta-adjusted), 7% hard floor, trailing after +5% gain
- `agents/buy_agent.py` — PPO actor-critic MLP with LayerNorm; binary buy/skip per candidate
- `agents/sell_agent.py` — separate PPO network; state augmented with position-specific features (unrealized P&L, days held, trailing flag, drawdown from peak)
- `training/train_buy.py` — PPO with GAE, selectivity reward shaping, val Sharpe checkpointing every 100 episodes
- `training/train_sell.py` — buy agent frozen during sell training; patience penalty for exits within 5 days
- `evaluation/backtest.py` — walk-forward backtest with `--split` and `--mock` flags
- `evaluation/metrics.py` — Sharpe, Sortino, CAGR, Calmar, max drawdown, win rate, exit reason breakdown

**WRDS data download: complete.** All 9 parquet files saved to `data/raw/`:
- `crsp_dsf.parquet` — 49.8M rows (v1 + v2 union)
- `crsp_dsedelist.parquet` — 23.6K rows
- `crsp_dsp500list.parquet` — 2,084 rows (v1 + v2 union)
- `compustat_fundq.parquet` — 802K rows
- `wrds_ratios.parquet` — 1.38M rows
- `ibes_statsum.parquet` — 5.32M rows (SUE computed as `(actual-meanest)/stdev`)
- `ibes_ticker_permno.parquet` — 37.7K rows
- `ff_factors.parquet` — 6,830 rows
- `beta_suite.parquet` — empty stub (not in subscription; beta features zero-filled in feature store)

**Known schema notes** (already handled in `wrds_download.py`):
- `dsf_v2` uses `dlycaldt/dlyprc/dlyret/dlyvol/dlycumfacpr/dlycumfacshr` instead of v1 names
- `dsp500list_v2` uses `mbrstartdt/mbrenddt` instead of `start/ending`
- `dsedelist_v2` and `ccmxpf_lnkhist_v2` do not exist — v1 only
- `firm_ratio` does not have `dltt_eq` or `short_ratio` — removed
- `ibes.statsum_epsus` does not have `suescore` — computed manually

**Next step:** run lag-correctness tests before first real training run:
```bash
python3.11 -m data.crsp
python3.11 -m data.compustat
```
Then swap `MockTradingEnvironment` → `TradingEnvironment` in `training/train_buy.py`.

**Environment API** (both real and mock expose the same interface):
```python
env = TradingEnvironment(split="train")   # or MockTradingEnvironment()
candidates = env.reset()                  # list of (permno, state_vector: np.ndarray)
candidates, reward, done, info = env.step(buy_action, sell_action)
# buy_action: list[int] of permnos to buy (subset of candidates)
# sell_action: list[int] of permnos to exit from open positions
```

**Sell agent state vector** = base market features + 5 position features:
- `unrealized_pnl_pct`, `days_held_norm` (÷120), `days_held_sq`, `trailing_stop_active`, `drawdown_from_peak`

**Checkpoints** saved under `checkpoints/buy/` and `checkpoints/sell/`. Best checkpoint selected by validation Sharpe.

**Enhancement data layers** (add after base pipeline validated): RavenPack, MarketPsych, Short Volume, OptionMetrics — one at a time, measure Sharpe improvement before adding the next.

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

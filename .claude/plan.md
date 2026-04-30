# Dev Split Plan

Two parallel tracks that converge at integration. Dev A owns the data pipeline; Dev B owns the agent stack. Dev B can develop against mock/synthetic data while Dev A builds the real pipeline.

---

## Dev A — Data Pipeline & Environment

**Owns:** `data/`, `data/environment.py`

**Goal:** Deliver a clean, tested, point-in-time feature matrix and RL environment that Dev B's agents can plug into.

### Tasks (in order)

1. **`data/wrds_download.py`**
   - Download CRSP `dsf`, `dsedelist`, `dsp500list`
   - Download Compustat `fundq` via CCM link
   - Download WRDS Financial Ratios, I/B/E/S `statsum_epsus`, Fama-French `fivefactors_daily`, WRDS Beta Suite
   - Save all raw tables as parquet under `data/raw/`

2. **`data/crsp.py`**
   - Point-in-time S&P 500 constituent lookup via `dsp500list`
   - Delisting return handling (`dlret`, `dlstcd` categories, Shumway fills)
   - Unit tests: constituent lookup on known historical dates, delisting fill logic

3. **`data/compustat.py`**
   - CCM link merge; use `rdq` as report date, fallback `datadate + 90d` if null
   - Unit tests: lag correctness (no feature uses data from after the as-of date)

4. **`data/preprocess.py`**
   - Technical indicators via `pandas-ta`: Donchian Channel (20-day), RSI, MACD, MAs, ATR%, relative volume, multi-horizon returns
   - All indicators must be price-level invariant

5. **`data/feature_store.py`**
   - Assemble full ~75-100 feature state vector per (date, permno)
   - Cross-sectional z-score normalization (across stocks on same date)
   - Join all sources: CRSP + Compustat + Ratios + I/B/E/S + FF factors + Betas
   - **Write lag-correctness tests** — no feature value from date T may use data with a publication date > T

6. **`data/environment.py`**
   - RL environment wrapping the feature store
   - Point-in-time universe filter: only stocks in S&P 500 on that date
   - Donchian breakout filter: pass only stocks where `close > 20-day upper band` (~5-20/day)
   - Force-exit on delisting (use `dlret`)
   - Transaction cost: deduct 0.1% per trade in step reward
   - Temporal split constants: Train 2000–2017, Val 2018–2019, Test 2020–2024
   - **Write tests:** universe size sanity check per date, delisting exit trigger, no future data in obs

### Deliverable for integration
- `data/environment.py` with a stable `reset()` / `step(action)` / `observe()` API
- A synthetic/mock version of the environment (`data/mock_environment.py`) that Dev B can use immediately — random features, same interface

---

## Dev B — Agents, Training & Evaluation

**Owns:** `agents/`, `training/`, `evaluation/`

**Goal:** Build and train the two PPO agents and produce a backtested performance report.

### Tasks (in order)

1. **`data/mock_environment.py`** *(coordinate with Dev A on interface spec first)*
   - Synthetic environment with the same `reset()` / `step()` / `observe()` API
   - Random feature vectors, configurable action space
   - Used for all agent development until Dev A's real environment is ready

2. **`agents/stop_loss.py`**
   - Rule-based, no ML
   - ATR-based stop: 2× ATR, adjusted by beta
   - Hard floor: 7% loss
   - Trailing stop: activates after 5% gain
   - Fires before Sell agent on every step
   - Unit tests: each trigger condition in isolation

3. **`agents/buy_agent.py`**
   - PPO neural network; input = state vector from environment
   - Output = binary (take breakout / skip)
   - Architecture: MLP with LayerNorm; tune depth/width after baseline works
   - Seeds: `torch.manual_seed(42)`

4. **`training/train_buy.py`**
   - PPO loop for buy agent
   - Reward: profitable + controlled drawdown → strong positive; large loss → 3× penalty; skipped bad trade → small positive; missed good trade → no penalty
   - Target behavior: 1-3 buys/day from 10-20 signals; buying everything = broken reward
   - Checkpoint every 100 episodes, keep best by validation Sharpe
   - Log every trade: permno, dates, prices, return, exit_reason, days_held, entry features

5. **`agents/sell_agent.py`**
   - Separate PPO network from buy agent — different input features (position age, unrealized P&L, etc.)
   - Output = binary (hold / exit)
   - Max hold = 120 trading days

6. **`training/train_sell.py`**
   - PPO loop for sell agent
   - Reward: patient holding of winners → positive; closing after 1-2 days consistently = broken reward
   - Stop loss fires before sell agent check on each step

7. **`evaluation/backtest.py`**
   - Walk-forward backtest on test set (2020–2024)
   - Apply both agents + stop loss in sequence per day
   - Include transaction costs

8. **`evaluation/metrics.py`**
   - Sharpe ratio, max drawdown, CAGR, win rate, avg hold days, exit reason breakdown
   - Compare universe returns vs `^GSPC` to estimate survivorship bias

### Deliverable for integration
- Agents that accept obs vectors matching Dev A's environment interface
- Passing training loop on mock environment before switching to real environment

---

## Integration Checklist

- [ ] Dev A's `data/environment.py` API matches Dev B's mock interface spec
- [ ] Feature vector dimensions are agreed and documented in `data/feature_store.py`
- [ ] End-to-end smoke test: `python main.py` runs one episode on real data without errors
- [ ] Lag-correctness tests pass on real feature store
- [ ] Backtest Sharpe on val set (2018-2019) reviewed before running test set — **run test set only once**

---

## Shared Conventions (both devs)

- Python 3.10+, PyTorch, `pandas` + `pyarrow`, parquet storage, `pandas-ta`
- WRDS access via `db.raw_sql()`
- Set seeds: `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)`
- Join key is always **PERMNO**, never ticker

# polymarket_alpha

A production-grade prediction-market research and execution system for Polymarket.

> ⚠️ **IMPORTANT WARNINGS**
> - This system provides **no guarantees of profitability**. All trading involves risk of loss.
> - Past backtest performance **does not predict** live trading results.
> - Never trade with money you cannot afford to lose.
> - Run exclusively in **paper trading mode** until you fully understand the system and have validated it over an extended period.
> - This is research software. It has not been independently audited.

---

## Overview

`polymarket_alpha` scans live Polymarket markets every minute, builds structured features, queries Claude as one probabilistic signal among several, calibrates ensemble estimates, and makes risk-managed trade decisions. Every decision is fully logged for auditability and reproducibility.

### What this system does

- Fetches live market data from Polymarket's public APIs every 60 seconds
- Builds ~30 structured features (price, momentum, volatility, liquidity, time-to-resolution, category)
- Queries Claude for a calibrated probability estimate with confidence score
- Combines LLM, microstructure, and heuristic signals via a confidence-weighted ensemble
- Post-hoc calibrates the ensemble output using isotonic regression trained on historical data
- Applies fractional Kelly sizing with hard caps on position size, category exposure, and daily loss
- Logs every decision with raw inputs, features, signal probabilities, and final action
- Provides a Streamlit dashboard showing live markets, trade log, bankroll curve, calibration metrics, and signal breakdown

### What this system does NOT do

- Guarantee any return or edge
- Fabricate historical performance
- Hide losing trades
- Use future information in features or signals
- Use full Kelly sizing
- Place live orders by default (paper trading only)

---

## Repository Structure

```
polymarket_alpha/
├── polymarket_alpha/
│   ├── schemas.py              # Typed Pydantic data models
│   ├── config.py               # Configuration management
│   ├── ingestion/
│   │   ├── collector.py        # MarketCollector: fetches live data
│   │   └── store.py            # MarketStore: PostgreSQL persistence
│   ├── features/
│   │   └── builder.py          # FeatureBuilder: causal feature engineering
│   ├── models/
│   │   ├── llm_model.py        # LLMProbabilityModel: Claude signal
│   │   ├── signal_models.py    # MicrostructureModel, HeuristicModel
│   │   └── ensemble.py         # EnsembleModel: confidence-weighted combination
│   ├── calibration/
│   │   └── calibrator.py       # Calibrator: isotonic regression, Brier, ECE
│   ├── risk/
│   │   └── manager.py          # RiskManager: Kelly sizing, position limits
│   ├── execution/
│   │   ├── decision_engine.py  # TradeDecisionEngine: full pipeline
│   │   ├── paper_trader.py     # PaperTrader: simulated live loop
│   │   └── live_trader.py      # LiveTrader: real order placement (disabled)
│   ├── backtest/
│   │   └── engine.py           # BacktestEngine: no-lookahead simulation
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard
│   └── utils/
│       ├── logging_utils.py    # Structured JSON logging
│       └── metrics_tracker.py  # MetricsTracker: rolling performance
├── tests/
│   └── test_all.py             # Full test suite
├── scripts/
│   ├── run_collector.py        # Collect live market data
│   ├── run_paper_trader.py     # Run paper trading loop
│   ├── run_backtest.py         # Run historical backtest
│   └── retrain_calibrators.py  # Retrain calibration models
├── config/
│   └── default.yaml            # Default configuration
├── .env.example                # Environment variable template
├── docker-compose.yml          # Reproducible deployment
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (or use Docker Compose)
- An Anthropic API key (for the LLM signal; the system degrades gracefully without it)

### 1. Clone and install

```bash
git clone https://github.com/your-org/polymarket_alpha.git
cd polymarket_alpha
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your values
```

Required:
```
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/polymarket_alpha
```

Optional (only for live trading, leave blank for paper trading):
```
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_API_PASSPHRASE=
```

### 3. Start the database

```bash
# Using Docker Compose:
docker compose up db -d

# Or create a local PostgreSQL database:
createdb polymarket_alpha
```

The schema is created automatically on first run.

---

## Running the System

### Run the data collector

Fetches live Polymarket markets every 60 seconds and stores snapshots:

```bash
python scripts/run_collector.py
```

### Run paper trading

Fetches live data and runs the full decision pipeline without placing real orders:

```bash
python scripts/run_paper_trader.py
```

Paper trading is the **default and recommended mode**. The `PAPER_TRADING=true` setting in `.env` must remain `true` until you are ready for live trading.

### Run the dashboard

```bash
streamlit run polymarket_alpha/dashboard/app.py
# Visit http://localhost:8501
```

The dashboard shows:
- Live markets ranked by estimated edge
- Trade and skip log with reasons
- Bankroll curve and drawdown
- Calibration metrics (Brier score, ECE, drift alerts)
- Signal breakdown per market (LLM, microstructure, heuristic, ensemble)
- Category-level performance

### Run a backtest

```bash
python scripts/run_backtest.py --start 2023-01-01 --end 2024-01-01 --bankroll 10000
```

**Note:** The backtest requires historical snapshots already stored by the collector, and a separate resolutions file mapping market IDs to outcomes. See `scripts/run_backtest.py` for details.

Backtest guarantees:
- Data is processed strictly in chronological order
- Only information available at each timestamp is used
- Resolved outcomes are never fed back into signal generation
- Fills are simulated with configured slippage and fees

### Retrain calibrators

After accumulating resolved market outcomes:

```bash
python scripts/retrain_calibrators.py --resolutions resolutions.json
```

`resolutions.json` format: `{"market_id_1": 1.0, "market_id_2": 0.0, ...}`

---

## Switching to Live Trading

> ⛔ Read this section carefully before proceeding.

Live trading places **real money** on Polymarket. There are no guarantees that this system will be profitable. You may lose your entire trading capital.

Before enabling live trading:
1. Run paper trading for at least several weeks
2. Validate calibration on out-of-sample data
3. Review all trade decisions and skip reasons
4. Understand the risk parameters and adjust them for your risk tolerance
5. Start with a small bankroll

To enable live trading:
1. Set `PAPER_TRADING=false` in `.env`
2. Set `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE`
3. Implement HMAC authentication in `execution/live_trader.py` (see Polymarket docs)
4. Run with `python scripts/run_paper_trader.py` (LiveTrader is automatically used when `paper_trading=false`)

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Test coverage includes:
- Claude JSON parsing (valid, malformed, embedded, code-fenced)
- Calibration fit, inference, drift detection
- Kelly sizing under various conditions
- Stale data detection
- Backtest no-lookahead enforcement
- Trade skip logic (low liquidity, high spread, negative EV)
- Schema validation and immutability

---

## Docker Compose (Full Stack)

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY

docker compose up --build
```

Services started:
- `db` — PostgreSQL
- `redis` — Redis (optional cache)
- `collector` — Market data collector
- `paper_trader` — Paper trading loop
- `dashboard` — Streamlit on port 8501

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_threshold` | 0.04 | Minimum expected edge after fees/slippage to trade |
| `kelly_fraction` | 0.25 | Fractional Kelly multiplier (never use 1.0) |
| `max_position_pct` | 0.05 | Max 5% of bankroll per position |
| `max_category_exposure_pct` | 0.20 | Max 20% in one category |
| `max_daily_loss_pct` | 0.05 | Halt if daily loss exceeds 5% of bankroll |
| `max_drawdown_pct` | 0.15 | Halt if drawdown from peak exceeds 15% |
| `min_liquidity` | 500 | Minimum USD liquidity to trade |
| `max_spread` | 0.08 | Maximum acceptable spread |
| `slippage_estimate` | 0.005 | Estimated slippage per trade |
| `fee_estimate` | 0.02 | Estimated Polymarket fee (~2%) |
| `min_confidence` | 0.50 | Minimum ensemble confidence to trade |
| `stale_data_threshold_seconds` | 300 | Skip markets with data older than 5 minutes |
| `calibration_drift_threshold` | 0.05 | Alert if Brier score degrades by this amount |

All parameters can be overridden via environment variables (uppercased).

---

## Design Principles

### No lookahead bias
- Features are computed only from data available at the snapshot timestamp
- Historical lookups strictly filter `timestamp < current_snapshot.timestamp`
- Resolved outcomes are stored separately and never passed to the signal pipeline
- Backtest iterates in strict chronological order

### Calibration over raw probabilities
- All signal outputs are post-hoc calibrated using isotonic regression
- Calibrators are trained on time-split data only (no data leakage across the train/test boundary)
- Calibration drift is monitored; position sizing is reduced automatically when calibration degrades

### Conservative risk management
- Fractional Kelly (0.25× by default) — never full Kelly
- Hard position caps regardless of Kelly output
- Per-category exposure limits
- Daily loss and drawdown halts
- Liquidity-aware sizing (never exceed 5% of market liquidity)

### Full audit trail
- Every decision is logged with: raw inputs, computed features, per-signal probabilities, calibrated estimate, risk state, and final action
- Every LLM prompt and response is stored for evaluation
- Every skipped trade includes a machine-readable reason

---

## Limitations and Risks

- **Model risk**: The ensemble model and calibrators may be miscalibrated, especially for new market categories or during unusual market conditions.
- **Data risk**: The Polymarket API may return stale, incomplete, or incorrect data. Stale data is detected and rejected, but silent API errors may go undetected.
- **Execution risk**: Slippage estimates are approximations. In thin markets, actual slippage may be much larger.
- **Liquidity risk**: Position sizes are capped by available liquidity, but liquidity can disappear rapidly near resolution.
- **Overfitting risk**: Any model trained on historical data may fail to generalise. Walk-forward evaluation reduces but does not eliminate this risk.
- **API risk**: Polymarket API changes may break data collection. Monitor logs for errors.
- **LLM risk**: Claude may produce systematically biased or overconfident probability estimates. It is used as one signal among several, with confidence-based down-weighting.

---

## License

MIT. See LICENSE.

This project is not affiliated with Polymarket or Anthropic.
# tradingbot

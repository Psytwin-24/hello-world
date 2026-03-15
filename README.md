# NSE/BSE Options Trading Bot

A production-grade, self-improving algorithmic options trading system for Indian markets (NSE/BSE).

> **DISCLAIMER**: This software is for educational and research purposes. Options trading involves substantial financial risk. Always start with paper trading and thorough backtesting. Past performance does not guarantee future results. Consult a SEBI-registered advisor before trading with real capital.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     OPTIONS TRADING BOT                          │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│  Data Layer  │  Analytics   │  Strategies  │  Auto-Research      │
│  NSE/BSE API │  BS Pricing  │  Iron Condor │  ML Retrain (4h)    │
│  yfinance    │  Full Greeks │  Short Strdl │  XGBoost/LightGBM   │
│  WebSockets  │  IV Rank/IVP │  Bull/Bear   │  Optuna HPO         │
│              │  Vol Surface │  Momentum    │  Walk-Forward CV    │
├──────────────┴──────────────┴──────────────┴─────────────────────┤
│                       Risk Manager                               │
│   Kelly Sizing · Drawdown Limits · Greeks Budget · Circuit Breaker│
├──────────────────────────────────────────────────────────────────┤
│                    Execution Engine                              │
│        Zerodha Kite · Fyers · Angel One · Paper Broker          │
├──────────────────────────────────────────────────────────────────┤
│               Notifications + Dashboard                         │
│         Telegram Alerts · Plotly Dash (port 8050)               │
└──────────────────────────────────────────────────────────────────┘
```

## Strategies

| Strategy | Type | Best Market | IV Condition |
|---|---|---|---|
| Iron Condor | Neutral / Credit | Sideways | IVR > 30 |
| Short Straddle | Neutral / Credit | Tight range | IVR > 50 |
| Bull Put Spread | Bullish / Credit | Uptrend | IVR > 30 |
| Bear Call Spread | Bearish / Credit | Downtrend | IVR > 30 |
| Momentum Directional | Directional / Debit | Strong trend | IVR < 50 |

## Auto-Research Loop (Self-Improvement)

Every 4 hours the bot automatically:
1. Fetches 2 years of market data
2. Engineers 40+ features (TA + IV + VIX + calendar effects)
3. Trains XGBoost/LightGBM ensemble with TimeSeriesSplit CV
4. Runs Optuna hyperparameter optimisation (100 trials)
5. Promotes new model only if AUC > 0.55 and win rate > 55%
6. Adjusts strategy weights based on recent Sharpe ratios
7. Detects regime change and adapts the active strategy mix

## Quick Start

### 1. VM Setup (Ubuntu 22.04)
```bash
git clone <repo> && cd options-trading-bot
bash scripts/setup_vm.sh
```

### 2. Configure credentials
```bash
cp config/.env.example .env
# Edit .env with your broker API keys
# PAPER_TRADING=true (default) — safe to start
```

### 3. Paper trading (always start here)
```bash
source venv/bin/activate
python src/bot.py --paper
```

### 4. Backtest a strategy
```bash
python scripts/run_backtest.py --strategy iron_condor --symbol NIFTY --days 365
```

### 5. Live dashboard
```bash
python src/bot.py --dashboard
# http://your-vm-ip:8050
```

### 6. Run as always-on systemd service
```bash
sudo systemctl start options-bot options-dashboard
sudo journalctl -fu options-bot
```

## Risk Controls

- Daily circuit breaker at 3% portfolio loss (halts all trading)
- Weekly limit: 7% drawdown
- Monthly limit: 15% drawdown
- Quarter-Kelly position sizing with fixed-fractional override
- Max 10 concurrent open positions
- Liquidity gate: min OI=500, volume=100, max bid-ask spread=2%
- Portfolio Greeks budget: max delta=50, vega=5000

## File Structure

```
src/
  bot.py                       # Main orchestrator (async event loop)
  data/market_data.py          # NSE/BSE API, option chain parser, PCR, max pain
  analytics/
    options_engine.py          # Black-Scholes, full Greeks, IV solver, vol surface
    technical.py               # 40+ TA indicators + ML feature engineering
  strategies/
    base.py                    # BaseStrategy interface
    premium_selling.py         # IC, Short Straddle, Bull Put, Bear Call, Momentum
  backtest/engine.py           # Event-driven backtester + walk-forward analyser
  research/auto_researcher.py  # Self-improving ML loop
  risk/manager.py              # Risk management + Kelly sizing + circuit breakers
  execution/broker.py          # Broker abstraction (Zerodha/Fyers/Angel/Paper)
  dashboard/app.py             # Plotly Dash real-time UI
config/settings.py             # All configuration (env-var driven)
scripts/
  setup_vm.sh                  # One-command VM setup
  run_backtest.py              # Standalone backtest runner
tests/                         # pytest test suite
```

## Brokers

| Broker | Status |
|---|---|
| Zerodha (Kite Connect) | Full |
| Paper trading | Full |
| Fyers | Planned |
| Angel One (SmartAPI) | Planned |

## Tests

```bash
pytest tests/ -v
```

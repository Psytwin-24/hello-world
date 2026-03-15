"""
Standalone backtest runner.

Usage:
    python scripts/run_backtest.py --strategy iron_condor --symbol NIFTY --days 365

For a full backtest with real historical option chain data, you need
to populate the snapshots DB first. This script demonstrates the API.
"""

import argparse
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import config
from src.backtest.engine import OptionsBacktestEngine, WalkForwardAnalyzer, PerformanceAnalytics
from src.analytics.options_engine import BlackScholes, OptionSpec
from src.strategies.premium_selling import IronCondorStrategy, ShortStraddleStrategy
from src.data.market_data import HistoricalDataFetcher


def simulate_option_chain_snapshots(
    symbol: str = "NIFTY",
    days: int = 252,
) -> list:
    """
    Generate synthetic option chain snapshots from historical OHLCV.
    In production: load from a database of real historical option chains.
    This approximation uses BS pricing from HV as proxy for IV.
    """
    logger.info(f"Generating synthetic snapshots for {symbol} over {days} days...")

    fetcher = HistoricalDataFetcher()
    ohlcv = fetcher.fetch_ohlcv(symbol, period=f"{days + 60}d")

    if ohlcv.empty:
        logger.error("No OHLCV data available")
        return []

    # Use last `days` rows
    ohlcv = ohlcv.tail(days).copy()

    # Estimate daily IV ≈ 20-day HV × 1.2 (typical IV premium over HV)
    log_ret = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    ohlcv["hv20"] = log_ret.rolling(20).std() * np.sqrt(252)
    ohlcv["iv"] = (ohlcv["hv20"] * 1.2).fillna(0.18).clip(0.10, 0.80)

    snapshots = []
    risk_free = 0.07

    step = 50 if symbol in ("NIFTY", "FINNIFTY") else 100

    for date, row in ohlcv.iterrows():
        spot = float(row["close"])
        iv = float(row["iv"])

        # Build synthetic chain around spot: ±10 strikes
        chain_rows = []
        expiry_dt = pd.Timestamp(date) + pd.Timedelta(days=21)
        expiry_str = expiry_dt.strftime("%d-%b-%Y").upper()
        tte = 21 / 365

        for i in range(-10, 11):
            strike = round(spot / step) * step + i * step
            if strike <= 0:
                continue

            for opt_type in ("CE", "PE"):
                spec = OptionSpec(
                    underlying=spot, strike=strike, tte=tte,
                    iv=iv, risk_free=risk_free, opt_type=opt_type,
                )
                from src.analytics.options_engine import BlackScholes
                g = BlackScholes.greeks(spec)

                moneyness = abs(strike - spot) / spot
                oi_base = int(500000 * np.exp(-10 * moneyness))  # OI peaks at ATM
                volume = oi_base // 20

                chain_rows.append({
                    "strike": strike,
                    "expiry": expiry_str,
                    "type": opt_type,
                    "ltp": round(g.price, 2),
                    "bid": round(g.price * 0.99, 2),
                    "ask": round(g.price * 1.01, 2),
                    "iv": iv * 100,
                    "delta": round(g.delta, 4),
                    "gamma": round(g.gamma, 6),
                    "theta": round(g.theta, 4),
                    "vega": round(g.vega, 4),
                    "oi": max(oi_base, 100),
                    "volume": max(volume, 10),
                    "oi_change": 0,
                    "change": 0,
                    "pct_change": 0,
                })

        chain_df = pd.DataFrame(chain_rows)

        # IV rank (rolling 252-day)
        iv_window = ohlcv["iv"].loc[:date].tail(252)
        iv_rank = float((iv - iv_window.min()) / (iv_window.max() - iv_window.min() + 1e-9) * 100)

        snapshots.append({
            "symbol": symbol,
            "spot": spot,
            "expiry": expiry_str,
            "vix": iv * 15 / 0.18,  # rough VIX proxy
            "iv_rank": iv_rank,
            "atm_iv": iv,
            "regime": 0,
            "chain": chain_df,
            "pcr": {"pcr": 1.0},
            "timestamp": str(date),
        })

    logger.info(f"Generated {len(snapshots)} snapshots")
    return snapshots


def run_backtest(strategy_name: str = "iron_condor", symbol: str = "NIFTY", days: int = 252):
    snapshots = simulate_option_chain_snapshots(symbol, days)
    if not snapshots:
        return

    # Select strategy
    strategy_map = {
        "iron_condor": IronCondorStrategy(config),
        "short_straddle": ShortStraddleStrategy(config),
    }

    strat = strategy_map.get(strategy_name)
    if not strat:
        logger.error(f"Unknown strategy: {strategy_name}")
        return

    engine = OptionsBacktestEngine(
        strategy_fn=strat.generate_signal,
        manage_fn=strat.manage_position,
        config=config,
        initial_capital=1_000_000,
        lot_size=config.market.lot_sizes.get(symbol, 50),
    )

    results = engine.run(snapshots, lots=1)
    print(results.summary())

    # Save results
    import json
    out = {
        "strategy": strategy_name,
        "symbol": symbol,
        "days": days,
        "metrics": results.metrics,
        "n_trades": len(results.trades),
    }
    print("\nMetrics JSON:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="iron_condor", choices=["iron_condor", "short_straddle"])
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--days", type=int, default=252)
    args = parser.parse_args()

    run_backtest(args.strategy, args.symbol, args.days)

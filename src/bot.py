"""
Main trading bot orchestrator.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                     OPTIONS TRADING BOT                      │
  ├──────────────┬──────────────┬──────────────┬────────────────┤
  │  Data Layer  │  Analytics   │  Strategies  │  Auto-Research │
  │  (NSE/BSE)   │  (Greeks,    │  (IC, SS,    │  (ML retrain   │
  │              │   IV, TA)    │   spreads)   │   every 4h)    │
  ├──────────────┴──────────────┴──────────────┴────────────────┤
  │                    Risk Manager                              │
  │         (Greeks limits, drawdown, position sizing)          │
  ├──────────────────────────────────────────────────────────────┤
  │                  Execution Engine                            │
  │       (Zerodha/Fyers/Angel/Paper broker abstraction)        │
  ├──────────────────────────────────────────────────────────────┤
  │               Notifications + Dashboard                      │
  └──────────────────────────────────────────────────────────────┘

Run:
  python src/bot.py --paper       # paper trading
  python src/bot.py --live        # live trading (needs broker credentials)
  python src/bot.py --backtest    # run backtest and exit
  python src/bot.py --research    # run single research cycle and exit
"""

import asyncio
import signal
import sys
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

import pytz
from loguru import logger

from config.settings import config
from src.data.market_data import MarketDataAggregator
from src.analytics.options_engine import IVMetrics, VolatilitySurface
from src.analytics.technical import TechnicalIndicators
from src.strategies.premium_selling import (
    IronCondorStrategy,
    ShortStraddleStrategy,
    BullPutSpreadStrategy,
    BearCallSpreadStrategy,
    MomentumDirectionalStrategy,
)
from src.risk.manager import RiskManager
from src.execution.broker import get_broker, format_nse_options_symbol
from src.research.auto_researcher import AutoResearcher
from src.strategies.base import SignalType

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Telegram notifier
# ---------------------------------------------------------------------------

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    async def send(self, message: str):
        if not self.token:
            return
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            async with aiohttp.ClientSession() as s:
                await s.post(url, json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                })
        except Exception as e:
            logger.warning(f"Telegram failed: {e}")


# ---------------------------------------------------------------------------
# Main bot class
# ---------------------------------------------------------------------------

class OptionsBot:
    def __init__(self):
        self.cfg = config
        self.market = MarketDataAggregator()
        self.broker = get_broker(config)
        self.risk = RiskManager(config, portfolio_value=self._get_portfolio_value())
        self.researcher = AutoResearcher(config, self.market)
        self.notifier = TelegramNotifier(
            config.notifications.telegram_token,
            config.notifications.telegram_chat_id,
        )

        # Instantiate all active strategies
        self.strategies = {
            "iron_condor": IronCondorStrategy(config),
            "short_straddle": ShortStraddleStrategy(config),
            "bull_put_spread": BullPutSpreadStrategy(config),
            "bear_call_spread": BearCallSpreadStrategy(config),
            "momentum_directional": MomentumDirectionalStrategy(config),
        }

        self._running = False
        self._last_research = None
        self._snapshots_cache: Dict[str, Dict] = {}
        self._iv_history: Dict[str, List[float]] = {}

    def _get_portfolio_value(self) -> float:
        try:
            funds = self.broker.get_funds()
            return funds.get("total", 1_000_000)
        except Exception:
            return 1_000_000

    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:  # Saturday, Sunday
            return False
        market_open = dtime(9, 15)
        market_close = dtime(15, 30)
        return market_open <= now.time() <= market_close

    def is_entry_window(self) -> bool:
        """Only enter new positions 9:30–14:00 IST."""
        now = datetime.now(IST).time()
        return dtime(9, 30) <= now <= dtime(14, 0)

    async def _build_snapshot(self, symbol: str) -> Optional[Dict]:
        """Aggregate all data needed by strategies into one snapshot dict."""
        try:
            snap = await self.market.get_full_market_snapshot(symbol)
            chain_df = snap.get("chain")
            spot = snap.get("spot", 0)

            if chain_df is None or chain_df.empty or spot == 0:
                return None

            # Compute IV rank
            ohlcv = self.market.get_historical_ohlcv(symbol, period="1y")
            hv = IVMetrics.historical_volatility(ohlcv["close"]) if not ohlcv.empty else 0.20

            vol_surface = VolatilitySurface(chain_df, spot)
            atm_iv = vol_surface.get_atm_iv()
            skew = vol_surface.get_skew()

            iv_hist = self._iv_history.setdefault(symbol, [])
            if atm_iv > 0:
                iv_hist.append(atm_iv)
            if len(iv_hist) > 252:
                iv_hist = iv_hist[-252:]
                self._iv_history[symbol] = iv_hist

            import pandas as pd
            iv_series = pd.Series(iv_hist)
            iv_rank = IVMetrics.iv_rank(atm_iv, iv_series) if len(iv_hist) > 10 else 50.0

            # Technical features
            regime = 0
            features = {}
            if not ohlcv.empty:
                feat_df = TechnicalIndicators.feature_matrix(ohlcv)
                if not feat_df.empty:
                    regime = int(feat_df["regime"].iloc[-1])
                    features = feat_df.iloc[-1].to_dict()

            # ML entry signal
            entry_prob = self.researcher.predict_entry(symbol, feat_df if not ohlcv.empty and not feat_df.empty else pd.DataFrame())

            snap.update({
                "iv_rank": iv_rank,
                "atm_iv": atm_iv,
                "hv": hv,
                "skew": skew,
                "regime": regime,
                "features": features,
                "entry_probability": entry_prob,
            })

            return snap

        except Exception as e:
            logger.error(f"Snapshot build failed for {symbol}: {e}")
            return None

    async def run_trading_cycle(self):
        """One iteration of the main trading loop."""
        for symbol in self.cfg.market.indices[:2]:  # Start with NIFTY + BANKNIFTY
            snap = await self._build_snapshot(symbol)
            if snap is None:
                continue

            chain_df = snap.get("chain")

            # 1. Manage existing positions
            for pos in list(self.risk.open_positions):
                for strat_name, strategy in self.strategies.items():
                    if pos.strategy_name == strat_name:
                        exit_signal = strategy.manage_position(pos, snap)
                        if exit_signal:
                            await self._execute_signal(exit_signal, snap, chain_df)

            # 2. Generate new entries (only during entry window)
            if not self.is_entry_window():
                continue

            strategy_weights = self.researcher.get_strategy_weights()

            for name, strategy in self.strategies.items():
                if name not in self.cfg.strategy.active_strategies:
                    continue

                # Only run strategies with non-trivial weight
                weight = strategy_weights.get(name, 0.1)
                if weight < 0.05:
                    continue

                # ML gate — only enter if model predicts good conditions
                entry_prob = snap.get("entry_probability", 0.5)
                if entry_prob < 0.45:
                    logger.debug(f"ML gate blocked {name} for {symbol}: prob={entry_prob:.2f}")
                    continue

                signal = strategy.generate_signal(snap)
                if signal:
                    await self._execute_signal(signal, snap, chain_df)

    async def _execute_signal(self, signal, snap, chain_df):
        """Risk-check and execute a strategy signal."""
        # Get historical stats for this strategy
        stats = {"win_rate": 0.60, "avg_win_pct": 0.50, "avg_loss_pct": 1.0}

        approved, reason, lots = self.risk.approve_signal(signal, chain_df, stats)
        if not approved:
            logger.debug(f"Signal rejected: {reason}")
            return

        logger.info(
            f"{'🟢 ENTRY' if signal.signal in (SignalType.ENTER_SHORT, SignalType.ENTER_LONG) else '🔴 EXIT'} "
            f"{signal.strategy_name} | {signal.symbol} | {signal.notes}"
        )

        lot_size = self.cfg.market.lot_sizes.get(signal.symbol, 50)
        orders = self.broker.place_signal(
            signal, lots, lot_size,
            lambda sym, k, exp, t: format_nse_options_symbol(sym, k, exp, t),
        )

        if self.cfg.notifications.alert_on_trade:
            mode = "PAPER" if self.cfg.paper_trading else "LIVE"
            msg = (
                f"[{mode}] *{signal.strategy_name.upper()}*\n"
                f"Signal: {signal.signal.value}\n"
                f"Symbol: {signal.symbol}\n"
                f"Lots: {lots} × {lot_size}\n"
                f"Credit: ₹{signal.expected_credit:.0f}\n"
                f"Notes: {signal.notes}"
            )
            await self.notifier.send(msg)

    async def run_research_if_due(self):
        """Trigger auto-research cycle every N hours."""
        now = datetime.now()
        interval = self.cfg.research.research_interval_hours

        if (self._last_research is None or
                (now - self._last_research).total_seconds() > interval * 3600):
            logger.info("Starting auto-research cycle...")
            for symbol in self.cfg.market.indices[:2]:
                report = await self.researcher.run_research_cycle(symbol)
                logger.info(
                    f"Research done for {symbol}: "
                    f"status={report.get('status')} "
                    f"deployed={report.get('deployed', False)}"
                )
            self._last_research = now

    async def send_daily_summary(self):
        """Send EOD performance summary via Telegram."""
        status = self.risk.get_status()
        msg = (
            f"📊 *Daily Summary — {datetime.now().strftime('%d %b %Y')}*\n"
            f"Portfolio: ₹{status['portfolio']:,.0f}\n"
            f"Daily P&L: {status['daily_pnl_pct']:+.2f}%\n"
            f"Open Positions: {status['open_positions']}\n"
            f"Circuit Breaker: {'🔴 ACTIVE' if status['circuit_breaker'] else '🟢 OK'}"
        )
        await self.notifier.send(msg)

    async def run(self):
        """Main async event loop."""
        self._running = True
        logger.info(
            f"🚀 NSE/BSE Options Bot starting — "
            f"mode={'PAPER' if self.cfg.paper_trading else '⚠️ LIVE'}"
        )

        # Initial research cycle
        await self.run_research_if_due()

        tick = 0
        while self._running:
            try:
                if self.is_market_open():
                    await self.run_trading_cycle()
                else:
                    # Run research during off-hours
                    await self.run_research_if_due()

                # Daily summary at 15:35 IST
                now_ist = datetime.now(IST)
                if now_ist.hour == 15 and now_ist.minute == 35 and tick % 12 == 0:
                    await self.send_daily_summary()

                tick += 1
                await asyncio.sleep(60)  # 1-minute loop

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Loop error: {e}")
                await asyncio.sleep(30)

        await self.shutdown()

    async def shutdown(self):
        logger.info("Shutting down...")
        await self.market.close()
        await self.notifier.send("🛑 Bot shut down")

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="NSE/BSE Options Trading Bot")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--backtest", action="store_true", help="Run backtest and exit")
    parser.add_argument("--research", action="store_true", help="Run one research cycle and exit")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard only")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol for backtest/research")
    args = parser.parse_args()

    if args.paper:
        config.paper_trading = True
    if args.live:
        config.paper_trading = False

    if args.dashboard:
        from src.dashboard.app import launch_dashboard
        launch_dashboard()
        return

    if args.backtest:
        await run_backtest(args.symbol)
        return

    if args.research:
        bot = OptionsBot()
        report = await bot.researcher.run_research_cycle(args.symbol)
        print(json_dumps(report))
        await bot.market.close()
        return

    # Full bot
    bot = OptionsBot()

    def handle_signal(sig, frame):
        logger.info(f"Got signal {sig} — stopping")
        bot.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    await bot.run()


async def run_backtest(symbol: str):
    """Quick backtest runner using last 1 year of data."""
    from src.backtest.engine import OptionsBacktestEngine, PerformanceAnalytics
    from src.strategies.premium_selling import IronCondorStrategy
    from src.data.market_data import HistoricalDataFetcher

    logger.info(f"Running backtest on {symbol}...")
    # In production: load historical option chain snapshots from DB
    # Here we demonstrate with a placeholder
    logger.info("Note: Full backtest requires historical option chain snapshots in DB.")
    logger.info("See src/backtest/engine.py for the full engine API.")


def json_dumps(obj) -> str:
    import json
    def default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, indent=2, default=default)


if __name__ == "__main__":
    asyncio.run(main())

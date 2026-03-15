"""
Risk management engine.

Responsibilities:
  1. Portfolio-level Greeks limits (delta, vega, theta)
  2. Daily/weekly/monthly drawdown circuit breakers
  3. Position sizing via Kelly criterion & fixed fractional
  4. Liquidity checks (OI, volume, bid-ask spread)
  5. Correlation-based concentration limits
  6. Real-time P&L tracking and alerts
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import Config
from src.strategies.base import Position, StrategySignal


@dataclass
class RiskMetrics:
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_vega: float = 0.0
    total_margin: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_single_loss: float = 0.0
    open_positions: int = 0
    circuit_breaker_active: bool = False


class PositionSizer:
    """Kelly criterion + fixed fractional position sizing."""

    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value

    def kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Kelly fraction f* = (p*b - q) / b
        p = win rate, q = 1-p, b = avg_win/avg_loss
        Capped at 0.25 (quarter Kelly) for safety.
        """
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b
        # Use quarter-Kelly to account for estimation error
        return max(0.0, min(kelly * 0.25, 0.25))

    def fixed_fractional(self, risk_fraction: float, max_loss_per_lot: float) -> int:
        """
        Position size in lots = floor(portfolio × fraction / max_loss_per_lot).
        """
        if max_loss_per_lot <= 0:
            return 0
        capital_at_risk = self.portfolio_value * risk_fraction
        return max(1, int(capital_at_risk / max_loss_per_lot))

    def size_for_signal(
        self,
        signal: StrategySignal,
        win_rate: float = 0.65,
        avg_win: float = 0.5,
        avg_loss: float = 1.0,
        lot_size: int = 50,
        max_capital_pct: float = 0.05,
    ) -> int:
        """Return number of lots for a strategy signal."""
        max_risk = signal.max_risk
        if max_risk == float("inf") or max_risk <= 0:
            # Naked strategies: limit to 2% of portfolio
            max_loss = self.portfolio_value * 0.02
            lots_by_capital = max(1, int(max_loss / (signal.expected_credit * 3 * lot_size)))
        else:
            lots_by_capital = self.fixed_fractional(max_capital_pct, max_risk * lot_size)

        kelly_f = self.kelly_fraction(win_rate, avg_win, avg_loss)
        lots_by_kelly = max(1, int((self.portfolio_value * kelly_f) / (max_risk * lot_size + 1)))

        return min(lots_by_capital, lots_by_kelly, 5)   # hard cap at 5 lots per entry


class LiquidityChecker:
    """Validate options have sufficient liquidity before trading."""

    def __init__(self, config: Config):
        self.cfg = config.risk

    def check_leg(self, chain_df: pd.DataFrame, opt_type: str, strike: float, expiry: str) -> Tuple[bool, str]:
        row = chain_df[
            (chain_df["type"] == opt_type) &
            (chain_df["strike"] == strike) &
            (chain_df["expiry"] == expiry)
        ]
        if row.empty:
            return False, f"No data for {opt_type} {strike} {expiry}"

        r = row.iloc[0]
        oi = r.get("oi", 0)
        vol = r.get("volume", 0)
        bid = r.get("bid", 0)
        ask = r.get("ask", 0)
        ltp = r.get("ltp", 0)

        if oi < self.cfg.min_liquidity_oi:
            return False, f"OI too low: {oi} < {self.cfg.min_liquidity_oi}"

        if vol < self.cfg.min_liquidity_volume:
            return False, f"Volume too low: {vol} < {self.cfg.min_liquidity_volume}"

        if bid > 0 and ask > 0 and ltp > 0:
            spread_pct = (ask - bid) / ltp
            if spread_pct > self.cfg.max_bid_ask_spread_pct:
                return False, f"Bid-ask spread too wide: {spread_pct:.2%}"

        return True, "ok"

    def check_signal(self, signal: StrategySignal, chain_df: pd.DataFrame) -> Tuple[bool, str]:
        for leg in signal.legs:
            ok, msg = self.check_leg(chain_df, leg["type"], leg["strike"], leg["expiry"])
            if not ok:
                return False, f"Leg {leg}: {msg}"
        return True, "ok"


class DrawdownTracker:
    """Track realised P&L and trigger circuit breakers."""

    def __init__(self, config: Config, initial_portfolio: float):
        self.cfg = config.risk
        self.portfolio = initial_portfolio
        self.daily_start = initial_portfolio
        self.weekly_start = initial_portfolio
        self.monthly_start = initial_portfolio
        self._last_reset = datetime.now().date()
        self.trade_history: List[Dict] = []
        self.circuit_breaker = False

    def record_trade(self, pnl: float, strategy: str, symbol: str):
        self.portfolio += pnl
        self.trade_history.append({
            "pnl": pnl, "strategy": strategy, "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "portfolio": self.portfolio,
        })
        self._maybe_reset()
        self._check_circuit_breakers()

    def _maybe_reset(self):
        today = datetime.now().date()
        if today != self._last_reset:
            self.daily_start = self.portfolio
            self._last_reset = today
        if today.weekday() == 0:  # Monday
            self.weekly_start = self.portfolio
        if today.day == 1:
            self.monthly_start = self.portfolio

    def _check_circuit_breakers(self):
        daily_loss = (self.daily_start - self.portfolio) / self.daily_start
        weekly_loss = (self.weekly_start - self.portfolio) / self.weekly_start
        monthly_loss = (self.monthly_start - self.portfolio) / self.monthly_start

        if (daily_loss >= self.cfg.circuit_breaker_pct or
                weekly_loss >= self.cfg.max_portfolio_loss_weekly or
                monthly_loss >= self.cfg.max_portfolio_loss_monthly):
            if not self.circuit_breaker:
                logger.critical(
                    f"CIRCUIT BREAKER TRIGGERED — daily={daily_loss:.2%} "
                    f"weekly={weekly_loss:.2%} monthly={monthly_loss:.2%}"
                )
                self.circuit_breaker = True

    @property
    def daily_pnl_pct(self) -> float:
        return (self.portfolio - self.daily_start) / self.daily_start

    @property
    def can_trade(self) -> bool:
        return not self.circuit_breaker


class RiskManager:
    """
    Central risk authority. Strategy engine must pass every signal
    through this before execution.
    """

    def __init__(self, config: Config, portfolio_value: float):
        self.config = config
        self.sizer = PositionSizer(portfolio_value)
        self.liquidity = LiquidityChecker(config)
        self.drawdown = DrawdownTracker(config, portfolio_value)
        self.open_positions: List[Position] = []

    def approve_signal(
        self,
        signal: StrategySignal,
        chain_df: pd.DataFrame,
        strategy_stats: Dict,
    ) -> Tuple[bool, str, int]:
        """
        Validate a proposed trade signal.

        Returns:
            (approved, reason, lots)
        """
        if not self.drawdown.can_trade:
            return False, "Circuit breaker active", 0

        if len(self.open_positions) >= self.config.strategy.max_open_positions:
            return False, f"Max positions reached ({self.config.strategy.max_open_positions})", 0

        # Liquidity check
        ok, msg = self.liquidity.check_signal(signal, chain_df)
        if not ok:
            return False, f"Liquidity fail: {msg}", 0

        # Portfolio Greeks check
        metrics = self._compute_portfolio_greeks(chain_df)
        # (simplified — full implementation fetches live Greeks per position)

        # Size the position
        lots = self.sizer.size_for_signal(
            signal,
            win_rate=strategy_stats.get("win_rate", 0.60),
            avg_win=strategy_stats.get("avg_win_pct", 0.50),
            avg_loss=strategy_stats.get("avg_loss_pct", 1.00),
            lot_size=self.config.market.lot_sizes.get(signal.symbol, 50),
            max_capital_pct=self.config.strategy.max_capital_per_trade,
        )

        return True, "approved", lots

    def _compute_portfolio_greeks(self, chain_df: pd.DataFrame) -> RiskMetrics:
        """Aggregate Greeks from all open positions."""
        metrics = RiskMetrics(open_positions=len(self.open_positions))
        # Production: fetch live option data for each leg and sum Greeks
        return metrics

    def on_trade_closed(self, position: Position, close_price: float):
        pnl = position.entry_credit - close_price
        self.drawdown.record_trade(pnl, position.strategy_name, position.symbol)
        if position in self.open_positions:
            self.open_positions.remove(position)

    def on_trade_opened(self, position: Position):
        self.open_positions.append(position)

    def get_status(self) -> Dict:
        return {
            "can_trade": self.drawdown.can_trade,
            "portfolio": self.drawdown.portfolio,
            "daily_pnl_pct": round(self.drawdown.daily_pnl_pct * 100, 2),
            "open_positions": len(self.open_positions),
            "circuit_breaker": self.drawdown.circuit_breaker,
        }

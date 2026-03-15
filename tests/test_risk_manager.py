"""Tests for risk management engine."""

import pytest
import pandas as pd

from config.settings import config
from src.risk.manager import DrawdownTracker, LiquidityChecker, PositionSizer
from src.strategies.base import StrategySignal, SignalType


class TestPositionSizer:
    def test_kelly_fraction_positive(self):
        sizer = PositionSizer(1_000_000)
        f = sizer.kelly_fraction(win_rate=0.65, avg_win=0.5, avg_loss=1.0)
        assert 0 < f <= 0.25

    def test_kelly_fraction_cap(self):
        # Even with perfect stats, should be capped at 0.25
        sizer = PositionSizer(1_000_000)
        f = sizer.kelly_fraction(win_rate=0.99, avg_win=10.0, avg_loss=1.0)
        assert f <= 0.25

    def test_fixed_fractional_returns_at_least_one(self):
        sizer = PositionSizer(1_000_000)
        lots = sizer.fixed_fractional(risk_fraction=0.01, max_loss_per_lot=5000)
        assert lots >= 1


class TestDrawdownTracker:
    def test_circuit_breaker_triggers_on_large_loss(self):
        tracker = DrawdownTracker(config, initial_portfolio=100_000)
        # Simulate 6% loss (above 5% circuit breaker)
        tracker.record_trade(-6_000, "iron_condor", "NIFTY")
        assert tracker.circuit_breaker is True

    def test_no_circuit_breaker_on_small_loss(self):
        tracker = DrawdownTracker(config, initial_portfolio=100_000)
        tracker.record_trade(-1_000, "iron_condor", "NIFTY")
        assert tracker.circuit_breaker is False
        assert tracker.can_trade is True

    def test_daily_pnl_pct(self):
        tracker = DrawdownTracker(config, initial_portfolio=100_000)
        tracker.record_trade(2_000, "iron_condor", "NIFTY")
        assert abs(tracker.daily_pnl_pct - 0.02) < 1e-6


class TestLiquidityChecker:
    def make_chain(self, oi=1000, volume=200, bid=100, ask=102, ltp=101):
        return pd.DataFrame([{
            "type": "CE", "strike": 20000, "expiry": "26-DEC-2024",
            "oi": oi, "volume": volume, "bid": bid, "ask": ask, "ltp": ltp,
        }])

    def test_passes_liquid_option(self):
        checker = LiquidityChecker(config)
        chain = self.make_chain()
        ok, _ = checker.check_leg(chain, "CE", 20000, "26-DEC-2024")
        assert ok is True

    def test_fails_low_oi(self):
        checker = LiquidityChecker(config)
        chain = self.make_chain(oi=10)
        ok, msg = checker.check_leg(chain, "CE", 20000, "26-DEC-2024")
        assert ok is False
        assert "OI" in msg

    def test_fails_wide_spread(self):
        checker = LiquidityChecker(config)
        chain = self.make_chain(bid=90, ask=115, ltp=100)  # 25% spread
        ok, msg = checker.check_leg(chain, "CE", 20000, "26-DEC-2024")
        assert ok is False
        assert "spread" in msg.lower()

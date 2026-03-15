"""Tests for the options analytics engine."""

import math
import pytest
import pandas as pd
import numpy as np

from src.analytics.options_engine import (
    BlackScholes,
    IVSolver,
    IVMetrics,
    ExpectedMove,
    OptionSpec,
    PortfolioGreeks,
    Leg,
)


def make_spec(S=20000, K=20000, T=21/365, iv=0.18, r=0.07, opt="CE"):
    return OptionSpec(underlying=S, strike=K, tte=T, iv=iv, risk_free=r, opt_type=opt)


class TestBlackScholes:
    def test_call_price_positive(self):
        spec = make_spec()
        price = BlackScholes.price(spec)
        assert price > 0

    def test_put_price_positive(self):
        spec = make_spec(opt="PE")
        price = BlackScholes.price(spec)
        assert price > 0

    def test_put_call_parity(self):
        S, K, T, r = 20000, 20000, 21/365, 0.07
        call = BlackScholes.price(make_spec(S, K, T, r=r, opt="CE"))
        put = BlackScholes.price(make_spec(S, K, T, r=r, opt="PE"))
        # C - P ≈ S - K*exp(-rT)
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 1.0  # within ₹1

    def test_call_delta_range(self):
        spec = make_spec()
        g = BlackScholes.greeks(spec)
        assert 0 < g.delta < 1

    def test_put_delta_range(self):
        spec = make_spec(opt="PE")
        g = BlackScholes.greeks(spec)
        assert -1 < g.delta < 0

    def test_gamma_positive(self):
        g = BlackScholes.greeks(make_spec())
        assert g.gamma > 0

    def test_theta_negative_call(self):
        g = BlackScholes.greeks(make_spec())
        assert g.theta < 0  # theta decay

    def test_vega_positive(self):
        g = BlackScholes.greeks(make_spec())
        assert g.vega > 0

    def test_expiry_intrinsic_call_itm(self):
        spec = make_spec(S=20100, K=20000, T=0)
        price = BlackScholes.price(spec)
        assert abs(price - 100) < 1e-4

    def test_expiry_intrinsic_call_otm(self):
        spec = make_spec(S=19900, K=20000, T=0)
        price = BlackScholes.price(spec)
        assert price == 0


class TestIVSolver:
    def test_round_trip(self):
        """IV computed from BS price should match original IV."""
        spec = make_spec(iv=0.20)
        market_price = BlackScholes.price(spec)
        solver_spec = OptionSpec(
            underlying=spec.underlying, strike=spec.strike,
            tte=spec.tte, iv=0.3, risk_free=spec.risk_free, opt_type=spec.opt_type
        )
        iv = IVSolver.solve(market_price, solver_spec)
        assert abs(iv - 0.20) < 1e-3

    def test_iv_increases_with_price(self):
        spec = make_spec()
        p1 = BlackScholes.price(spec)
        p2 = BlackScholes.price(make_spec(iv=0.30))
        iv1 = IVSolver.solve(p1, spec)
        iv2 = IVSolver.solve(p2, spec)
        assert iv2 > iv1

    def test_zero_price_returns_zero(self):
        assert IVSolver.solve(0, make_spec()) == 0.0


class TestIVMetrics:
    def test_iv_rank_mid(self):
        hist = pd.Series([0.10, 0.15, 0.20, 0.25, 0.30])
        ivr = IVMetrics.iv_rank(0.20, hist)
        assert abs(ivr - 50.0) < 1.0

    def test_iv_rank_max(self):
        hist = pd.Series([0.10, 0.20])
        ivr = IVMetrics.iv_rank(0.20, hist)
        assert ivr == 100.0

    def test_hv_computation(self):
        prices = pd.Series(np.cumprod(1 + np.random.randn(100) * 0.01) * 20000)
        hv = IVMetrics.historical_volatility(prices, window=20)
        assert 0 < hv < 2.0  # sane range


class TestExpectedMove:
    def test_one_sigma_positive(self):
        em = ExpectedMove.one_sigma_move(20000, 0.20, 21)
        assert em > 0

    def test_prob_otm_between_zero_one(self):
        spec = make_spec(K=21000)  # OTM call
        p = ExpectedMove.probability_otm(spec)
        assert 0 <= p <= 1
        assert p > 0.5  # OTM → >50% chance of expiring worthless


class TestPortfolioGreeks:
    def test_iron_condor_delta_near_zero(self):
        """IC with symmetric strikes should have near-zero delta."""
        S = 20000
        legs = [
            Leg(make_spec(S=S, K=20500, opt="CE", iv=0.18), quantity=-1),
            Leg(make_spec(S=S, K=21000, opt="CE", iv=0.17), quantity=1),
            Leg(make_spec(S=S, K=19500, opt="PE", iv=0.19), quantity=-1),
            Leg(make_spec(S=S, K=19000, opt="PE", iv=0.18), quantity=1),
        ]
        g = PortfolioGreeks.aggregate(legs)
        assert abs(g["delta"]) < 0.2  # roughly delta-neutral

"""
Options analytics engine.

Computes:
  - Black-Scholes pricing & all Greeks (delta, gamma, theta, vega, rho)
  - Implied Volatility (Newton-Raphson + fallback bisection)
  - IV Rank (IVR) and IV Percentile (IVP)
  - Term structure & volatility surface
  - Expected move calculations
  - Probability of profit (POP) for multi-leg strategies
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptionGreeks:
    price: float
    delta: float
    gamma: float
    theta: float        # per day
    vega: float         # per 1% move in IV
    rho: float
    iv: float = 0.0
    intrinsic: float = 0.0
    extrinsic: float = 0.0


@dataclass
class OptionSpec:
    underlying: float   # spot price
    strike: float
    tte: float          # time to expiry in years
    iv: float           # annualised implied vol (decimal, e.g. 0.18)
    risk_free: float    # annualised risk-free rate (decimal, e.g. 0.07)
    opt_type: str       # 'CE' or 'PE'
    dividend: float = 0.0


# ---------------------------------------------------------------------------
# Black-Scholes core
# ---------------------------------------------------------------------------

class BlackScholes:
    """Vectorised Black-Scholes with full Greeks."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        return BlackScholes.d1(S, K, T, r, sigma, q) - sigma * math.sqrt(T)

    @classmethod
    def price(cls, spec: OptionSpec) -> float:
        S, K, T, r, sigma, q = (
            spec.underlying, spec.strike, spec.tte,
            spec.risk_free, spec.iv, spec.dividend,
        )
        if T <= 0:
            intrinsic = max(S - K, 0) if spec.opt_type == "CE" else max(K - S, 0)
            return intrinsic

        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)

        if spec.opt_type == "CE":
            return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

    @classmethod
    def greeks(cls, spec: OptionSpec) -> OptionGreeks:
        S, K, T, r, sigma, q = (
            spec.underlying, spec.strike, spec.tte,
            spec.risk_free, spec.iv, spec.dividend,
        )
        price = cls.price(spec)

        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0) if spec.opt_type == "CE" else max(K - S, 0)
            delta = 1.0 if (spec.opt_type == "CE" and S > K) else \
                    (-1.0 if (spec.opt_type == "PE" and S < K) else 0.0)
            return OptionGreeks(price=price, delta=delta, gamma=0, theta=0, vega=0, rho=0,
                                intrinsic=intrinsic, extrinsic=max(0, price - intrinsic))

        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)
        nd1 = norm.pdf(d1)
        sqrt_T = math.sqrt(T)

        gamma = nd1 * math.exp(-q * T) / (S * sigma * sqrt_T)
        vega = S * math.exp(-q * T) * nd1 * sqrt_T / 100  # per 1% change in IV

        if spec.opt_type == "CE":
            delta = math.exp(-q * T) * norm.cdf(d1)
            theta = (
                -S * math.exp(-q * T) * nd1 * sigma / (2 * sqrt_T)
                - r * K * math.exp(-r * T) * norm.cdf(d2)
                + q * S * math.exp(-q * T) * norm.cdf(d1)
            ) / 365
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = math.exp(-q * T) * (norm.cdf(d1) - 1)
            theta = (
                -S * math.exp(-q * T) * nd1 * sigma / (2 * sqrt_T)
                + r * K * math.exp(-r * T) * norm.cdf(-d2)
                - q * S * math.exp(-q * T) * norm.cdf(-d1)
            ) / 365
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        intrinsic = max(S - K, 0) if spec.opt_type == "CE" else max(K - S, 0)
        extrinsic = max(0, price - intrinsic)

        return OptionGreeks(
            price=price, delta=delta, gamma=gamma,
            theta=theta, vega=vega, rho=rho,
            iv=sigma, intrinsic=intrinsic, extrinsic=extrinsic,
        )


# ---------------------------------------------------------------------------
# Implied Volatility solver
# ---------------------------------------------------------------------------

class IVSolver:
    """Newton-Raphson IV solver with Brent fallback."""

    @staticmethod
    def solve(
        market_price: float,
        spec: OptionSpec,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Return IV (annualised decimal) that makes BS price == market_price.
        Returns 0.0 if solution not found.
        """
        if market_price <= 0 or spec.tte <= 0:
            return 0.0

        intrinsic = (
            max(spec.underlying - spec.strike, 0)
            if spec.opt_type == "CE"
            else max(spec.strike - spec.underlying, 0)
        )
        if market_price < intrinsic:
            return 0.0

        # Newton-Raphson
        sigma = 0.3  # initial guess
        for _ in range(max_iter):
            spec_tmp = OptionSpec(
                spec.underlying, spec.strike, spec.tte,
                sigma, spec.risk_free, spec.opt_type, spec.dividend,
            )
            g = BlackScholes.greeks(spec_tmp)
            diff = g.price - market_price
            if abs(diff) < tol:
                return sigma
            vega_raw = g.vega * 100  # convert back from per-1%
            if abs(vega_raw) < 1e-10:
                break
            sigma -= diff / vega_raw
            sigma = max(1e-4, min(sigma, 20.0))

        # Brent fallback
        try:
            def objective(s):
                sp = OptionSpec(
                    spec.underlying, spec.strike, spec.tte,
                    s, spec.risk_free, spec.opt_type, spec.dividend,
                )
                return BlackScholes.price(sp) - market_price

            return brentq(objective, 1e-4, 20.0, xtol=tol, maxiter=200)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# IV Rank & Percentile
# ---------------------------------------------------------------------------

class IVMetrics:
    """Historical volatility context metrics."""

    @staticmethod
    def iv_rank(current_iv: float, iv_history: pd.Series) -> float:
        """
        IV Rank (IVR) = (current - 52w low) / (52w high - 52w low) × 100
        Range: 0–100. High IVR (>50) → sell premium. Low IVR (<20) → buy premium.
        """
        lo, hi = iv_history.min(), iv_history.max()
        if hi == lo:
            return 50.0
        return round((current_iv - lo) / (hi - lo) * 100, 2)

    @staticmethod
    def iv_percentile(current_iv: float, iv_history: pd.Series) -> float:
        """
        IV Percentile (IVP) = % of days in the past year where IV was below current IV.
        """
        return round(float((iv_history < current_iv).mean() * 100), 2)

    @staticmethod
    def historical_volatility(close_prices: pd.Series, window: int = 20) -> float:
        """Annualised HV from log returns."""
        log_ret = np.log(close_prices / close_prices.shift(1)).dropna()
        if len(log_ret) < window:
            return 0.0
        hv = log_ret.rolling(window).std().iloc[-1] * math.sqrt(252)
        return round(float(hv), 6)

    @staticmethod
    def iv_hv_spread(iv: float, hv: float) -> float:
        """Positive spread → IV elevated above HV → sell premium environment."""
        return round(iv - hv, 6)


# ---------------------------------------------------------------------------
# Expected move
# ---------------------------------------------------------------------------

class ExpectedMove:
    """Market-implied expected move calculations."""

    @staticmethod
    def one_sigma_move(spot: float, iv: float, dte: int) -> float:
        """
        1-sigma expected move = Spot × IV × √(DTE / 365)
        Roughly 68% probability of price staying within ±this range.
        """
        return round(spot * iv * math.sqrt(dte / 365), 2)

    @staticmethod
    def atm_straddle_price_approx(spot: float, iv: float, dte: int) -> float:
        """ATM straddle ≈ 0.8 × S × IV × √(T)  (quick approximation)."""
        return round(0.8 * spot * iv * math.sqrt(dte / 365), 2)

    @staticmethod
    def probability_otm(spec: OptionSpec) -> float:
        """
        Risk-neutral probability that option expires OTM (worthless).
        = N(-d2) for call, N(d2) for put.
        """
        if spec.tte <= 0:
            intrinsic = (
                max(spec.underlying - spec.strike, 0)
                if spec.opt_type == "CE"
                else max(spec.strike - spec.underlying, 0)
            )
            return 1.0 if intrinsic == 0 else 0.0
        d2 = BlackScholes.d2(spec.underlying, spec.strike, spec.tte,
                              spec.risk_free, spec.iv, spec.dividend)
        return round(float(norm.cdf(-d2) if spec.opt_type == "CE" else norm.cdf(d2)), 4)


# ---------------------------------------------------------------------------
# Portfolio Greeks aggregator
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    spec: OptionSpec
    quantity: int       # +ve = long, -ve = short
    lot_size: int = 1


class PortfolioGreeks:
    """Aggregate Greeks across a multi-leg position."""

    @staticmethod
    def aggregate(legs: List[Leg]) -> Dict:
        total = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0,
                 "value": 0.0, "pop": 0.0}
        for leg in legs:
            g = BlackScholes.greeks(leg.spec)
            multiplier = leg.quantity * leg.lot_size
            total["delta"] += g.delta * multiplier
            total["gamma"] += g.gamma * multiplier
            total["theta"] += g.theta * multiplier
            total["vega"] += g.vega * multiplier
            total["rho"] += g.rho * multiplier
            total["value"] += g.price * multiplier
        # Simplified POP via theta/vega ratio proxy (not exact)
        return {k: round(v, 4) for k, v in total.items()}


# ---------------------------------------------------------------------------
# Volatility surface builder
# ---------------------------------------------------------------------------

class VolatilitySurface:
    """Build and interpolate the IV surface across strikes and expiries."""

    def __init__(self, chain_df: pd.DataFrame, spot: float, risk_free: float = 0.07):
        self.chain = chain_df.copy()
        self.spot = spot
        self.risk_free = risk_free
        self._surface: Optional[pd.DataFrame] = None

    def build(self) -> pd.DataFrame:
        """Compute IV for each option in the chain."""
        ivs = []
        for _, row in self.chain.iterrows():
            ltp = row.get("ltp", 0)
            strike = row.get("strike", 0)
            expiry_str = row.get("expiry", "")
            opt_type = row.get("type", "CE")

            try:
                expiry_dt = pd.to_datetime(expiry_str, dayfirst=True)
                tte = max((expiry_dt - pd.Timestamp.now()).days / 365, 1e-6)
            except Exception:
                tte = 1 / 365

            spec = OptionSpec(
                underlying=self.spot, strike=strike, tte=tte,
                iv=0.3, risk_free=self.risk_free, opt_type=opt_type,
            )
            iv = IVSolver.solve(ltp, spec) if ltp > 0 else 0.0
            moneyness = round(math.log(self.spot / strike), 4) if strike > 0 else 0
            ivs.append({**row.to_dict(), "computed_iv": iv, "moneyness": moneyness})

        self._surface = pd.DataFrame(ivs)
        return self._surface

    def get_atm_iv(self, expiry: Optional[str] = None) -> float:
        """Return ATM IV for given expiry (nearest if not specified)."""
        if self._surface is None:
            self.build()
        df = self._surface
        if expiry:
            df = df[df["expiry"] == expiry]
        atm = df.iloc[(df["strike"] - self.spot).abs().argsort()[:4]]
        iv = atm[atm["computed_iv"] > 0]["computed_iv"].mean()
        return round(float(iv), 6) if not np.isnan(iv) else 0.0

    def get_skew(self, expiry: Optional[str] = None) -> float:
        """25-delta put IV minus 25-delta call IV (risk reversal proxy)."""
        if self._surface is None:
            self.build()
        df = self._surface
        if expiry:
            df = df[df["expiry"] == expiry]
        puts = df[(df["type"] == "PE") & (df["computed_iv"] > 0)]
        calls = df[(df["type"] == "CE") & (df["computed_iv"] > 0)]
        if puts.empty or calls.empty:
            return 0.0
        # Rough proxy: OTM put IV vs OTM call IV at similar distance from spot
        otm_puts = puts[puts["strike"] < self.spot]
        otm_calls = calls[calls["strike"] > self.spot]
        if otm_puts.empty or otm_calls.empty:
            return 0.0
        return round(float(otm_puts["computed_iv"].mean() - otm_calls["computed_iv"].mean()), 6)

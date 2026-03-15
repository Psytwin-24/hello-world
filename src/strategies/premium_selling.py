"""
Premium-selling (short volatility) strategies — the most consistent
edge in Indian markets.

Strategies:
  - Short Straddle   : Short ATM CE + PE same strike
  - Short Strangle   : Short OTM CE + OTM PE
  - Iron Condor      : Bull Put Spread + Bear Call Spread
  - Bull Put Spread  : Short higher strike put + Long lower strike put
  - Bear Call Spread : Short lower strike call + Long higher strike call
"""

from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from .base import BaseStrategy, Position, SignalType, StrategySignal


class ShortStraddleStrategy(BaseStrategy):
    """
    Short ATM straddle — highest premium, highest risk.
    Best in ranging, high-IV environments (IVR > 50).
    Adjustments: roll untested side when breached.
    """

    name = "short_straddle"

    def generate_signal(self, snap: Dict) -> Optional[StrategySignal]:
        if not self.is_entry_allowed(snap):
            return None

        iv_rank = snap.get("iv_rank", 0)
        vix = snap.get("vix", 15)
        regime = snap.get("regime", 0)
        chain_df: pd.DataFrame = snap.get("chain")
        spot = snap.get("spot", 0)
        symbol = snap.get("symbol", "NIFTY")

        if chain_df is None or chain_df.empty:
            return None

        # Enter only when IV is elevated and market is ranging
        if iv_rank < self.config.strategy.target_iv_rank_min:
            return None
        if abs(regime) == 1 and vix > 25:  # trending + high vix → avoid
            return None

        target_dte = self.config.strategy.preferred_dte
        expiry = self._select_expiry(chain_df, target_dte)
        if not expiry:
            return None

        # ATM strike — round to nearest 50 for NIFTY
        atm = round(spot / 50) * 50
        ce_row = chain_df[(chain_df["expiry"] == expiry) &
                          (chain_df["type"] == "CE") &
                          (chain_df["strike"] == atm)]
        pe_row = chain_df[(chain_df["expiry"] == expiry) &
                          (chain_df["type"] == "PE") &
                          (chain_df["strike"] == atm)]

        if ce_row.empty or pe_row.empty:
            return None

        ce_price = float(ce_row.iloc[0]["ltp"])
        pe_price = float(pe_row.iloc[0]["ltp"])
        total_premium = ce_price + pe_price

        if total_premium < 50:  # minimum premium filter
            return None

        breakeven_up = atm + total_premium
        breakeven_dn = atm - total_premium

        return StrategySignal(
            strategy_name=self.name,
            signal=SignalType.ENTER_SHORT,
            symbol=symbol,
            legs=[
                {"type": "CE", "strike": atm, "expiry": expiry, "action": "sell", "qty": 1},
                {"type": "PE", "strike": atm, "expiry": expiry, "action": "sell", "qty": 1},
            ],
            confidence=min(iv_rank / 100, 0.9),
            expected_credit=total_premium,
            max_risk=float("inf"),   # naked straddle — unlimited on one side
            breakevens=[breakeven_dn, breakeven_up],
            notes=f"ATM straddle @ {atm}, IVR={iv_rank:.1f}",
        )

    def manage_position(self, position: Position, snap: Dict) -> Optional[StrategySignal]:
        chain_df = snap.get("chain")
        spot = snap.get("spot", 0)
        if chain_df is None or chain_df.empty:
            return None

        credit = position.entry_credit
        current_value = self._estimate_position_value(position, chain_df)
        pnl_pct = (credit - current_value) / credit if credit > 0 else 0

        # Profit target: 50% of credit
        if pnl_pct >= self.config.strategy.profit_target_pct:
            return self._exit_signal(position, snap, "profit_target")

        # Stop loss: 2x credit received
        if current_value > credit * self.config.strategy.stop_loss_pct:
            return self._exit_signal(position, snap, "stop_loss")

        # DTE-based exit
        expiry = position.legs[0].get("expiry", "")
        if expiry:
            dte = (pd.to_datetime(expiry, dayfirst=True) - pd.Timestamp.now()).days
            if dte <= self.config.strategy.dte_exit:
                return self._exit_signal(position, snap, "dte_exit")

        return None

    def _estimate_position_value(self, position: Position, chain_df: pd.DataFrame) -> float:
        total = 0.0
        for leg in position.legs:
            row = chain_df[
                (chain_df["type"] == leg["type"]) &
                (chain_df["strike"] == leg["strike"]) &
                (chain_df["expiry"] == leg["expiry"])
            ]
            if not row.empty:
                total += float(row.iloc[0]["ltp"])
        return total

    def _exit_signal(self, position: Position, snap: Dict, reason: str) -> StrategySignal:
        exit_legs = [{**leg, "action": "buy"} for leg in position.legs]
        return StrategySignal(
            strategy_name=self.name,
            signal=SignalType.EXIT_SHORT,
            symbol=position.symbol,
            legs=exit_legs,
            confidence=1.0,
            notes=f"Exit: {reason}",
        )


class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor — defined-risk premium selling.
    Best in sideways markets with elevated IV.

    Structure: Bull Put Spread + Bear Call Spread
    Typically placed ~0.16–0.30 delta OTM strikes.
    """

    name = "iron_condor"

    def generate_signal(self, snap: Dict) -> Optional[StrategySignal]:
        if not self.is_entry_allowed(snap):
            return None

        iv_rank = snap.get("iv_rank", 0)
        regime = snap.get("regime", 0)
        chain_df: pd.DataFrame = snap.get("chain")
        spot = snap.get("spot", 0)
        symbol = snap.get("symbol", "NIFTY")
        sr = snap.get("support_resistance", {})

        if chain_df is None or chain_df.empty or spot == 0:
            return None

        if iv_rank < self.config.strategy.target_iv_rank_min:
            return None

        # Prefer ranging regime, but allow mild trend
        if abs(regime) == 1 and iv_rank < 60:
            return None

        target_dte = self.config.strategy.preferred_dte
        expiry = self._select_expiry(chain_df, target_dte)
        if not expiry:
            return None

        lo, hi = self.config.strategy.short_delta_range
        lg_lo, lg_hi = self.config.strategy.long_delta_range

        # Short strikes (sell these)
        short_call = self._find_strike_by_delta(chain_df, expiry, "CE", (lo + hi) / 2)
        short_put = self._find_strike_by_delta(chain_df, expiry, "PE", (lo + hi) / 2)

        if short_call == 0 or short_put == 0:
            return None

        # Long strikes (buy these for protection), one strike width away
        step = self._detect_strike_step(chain_df, symbol)
        long_call = short_call + step * 2
        long_put = short_put - step * 2

        # Gather prices
        def ltp(t, k):
            r = chain_df[(chain_df["type"] == t) & (chain_df["strike"] == k) &
                         (chain_df["expiry"] == expiry)]
            return float(r.iloc[0]["ltp"]) if not r.empty else 0.0

        sc = ltp("CE", short_call)
        lc = ltp("CE", long_call)
        sp = ltp("PE", short_put)
        lp = ltp("PE", long_put)

        net_credit = (sc - lc) + (sp - lp)
        if net_credit < 30:  # filter for minimum premium
            return None

        max_risk = step * 2 - net_credit
        pop = 1 - (net_credit / (step * 2)) if step > 0 else 0.5

        return StrategySignal(
            strategy_name=self.name,
            signal=SignalType.ENTER_SHORT,
            symbol=symbol,
            legs=[
                {"type": "CE", "strike": long_call, "expiry": expiry, "action": "buy", "qty": 1},
                {"type": "CE", "strike": short_call, "expiry": expiry, "action": "sell", "qty": 1},
                {"type": "PE", "strike": short_put, "expiry": expiry, "action": "sell", "qty": 1},
                {"type": "PE", "strike": long_put, "expiry": expiry, "action": "buy", "qty": 1},
            ],
            confidence=min(0.5 + iv_rank / 200, 0.9),
            expected_credit=net_credit,
            max_risk=max_risk,
            breakevens=[short_put - net_credit, short_call + net_credit],
            notes=f"IC {long_put}/{short_put}/{short_call}/{long_call} credit={net_credit:.0f} IVR={iv_rank:.1f}",
            metadata={"pop_estimate": round(pop, 3)},
        )

    def manage_position(self, position: Position, snap: Dict) -> Optional[StrategySignal]:
        chain_df = snap.get("chain")
        if chain_df is None:
            return None

        credit = position.entry_credit
        current_value = sum(
            float(chain_df[
                (chain_df["type"] == l["type"]) &
                (chain_df["strike"] == l["strike"]) &
                (chain_df["expiry"] == l["expiry"])
            ].iloc[0]["ltp"]) if not chain_df[
                (chain_df["type"] == l["type"]) &
                (chain_df["strike"] == l["strike"])
            ].empty else 0
            for l in position.legs
        )

        pnl_pct = (credit - current_value) / credit if credit > 0 else 0

        if pnl_pct >= self.config.strategy.profit_target_pct:
            return self._close(position, "profit_target")

        if current_value > credit * self.config.strategy.stop_loss_pct:
            return self._close(position, "stop_loss")

        expiry = position.legs[0].get("expiry", "")
        if expiry:
            dte = (pd.to_datetime(expiry, dayfirst=True) - pd.Timestamp.now()).days
            if dte <= self.config.strategy.dte_exit:
                return self._close(position, "dte_exit")

        return None

    def _close(self, position: Position, reason: str) -> StrategySignal:
        flip = {"buy": "sell", "sell": "buy"}
        exit_legs = [{**l, "action": flip[l["action"]]} for l in position.legs]
        return StrategySignal(
            strategy_name=self.name,
            signal=SignalType.EXIT_SHORT,
            symbol=position.symbol,
            legs=exit_legs,
            confidence=1.0,
            notes=f"Exit: {reason}",
        )

    @staticmethod
    def _detect_strike_step(chain_df: pd.DataFrame, symbol: str) -> float:
        steps = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "SENSEX": 100}
        if symbol in steps:
            return steps[symbol]
        strikes = sorted(chain_df["strike"].unique())
        if len(strikes) >= 2:
            return float(strikes[1] - strikes[0])
        return 50.0


class BullPutSpreadStrategy(BaseStrategy):
    """
    Bull Put Spread (Credit Put Spread).
    Sell higher strike put, buy lower strike put.
    Directional bullish bias with limited risk.
    """

    name = "bull_put_spread"

    def generate_signal(self, snap: Dict) -> Optional[StrategySignal]:
        if not self.is_entry_allowed(snap):
            return None

        regime = snap.get("regime", 0)
        iv_rank = snap.get("iv_rank", 0)
        pcr = snap.get("pcr", {}).get("pcr", 1.0)
        chain_df = snap.get("chain")
        spot = snap.get("spot", 0)
        symbol = snap.get("symbol", "NIFTY")

        if chain_df is None or chain_df.empty:
            return None

        # Need bullish bias AND elevated IV
        if regime != -1:   # -1 = bullish in our regime detector
            return None
        if iv_rank < 30:
            return None
        if pcr > 1.5:      # extreme fear — wait
            return None

        expiry = self._select_expiry(chain_df, self.config.strategy.preferred_dte)
        if not expiry:
            return None

        step = IronCondorStrategy._detect_strike_step(chain_df, symbol)
        short_put = self._find_strike_by_delta(chain_df, expiry, "PE", 0.30)
        long_put = short_put - step * 2

        def ltp(k):
            r = chain_df[(chain_df["type"] == "PE") &
                         (chain_df["strike"] == k) &
                         (chain_df["expiry"] == expiry)]
            return float(r.iloc[0]["ltp"]) if not r.empty else 0.0

        credit = ltp(short_put) - ltp(long_put)
        if credit < 20:
            return None

        return StrategySignal(
            strategy_name=self.name,
            signal=SignalType.ENTER_SHORT,
            symbol=symbol,
            legs=[
                {"type": "PE", "strike": short_put, "expiry": expiry, "action": "sell", "qty": 1},
                {"type": "PE", "strike": long_put, "expiry": expiry, "action": "buy", "qty": 1},
            ],
            confidence=0.6 + min(iv_rank / 200, 0.2),
            expected_credit=credit,
            max_risk=step * 2 - credit,
            breakevens=[short_put - credit],
            notes=f"Bull put {long_put}/{short_put} credit={credit:.0f}",
        )

    def manage_position(self, position: Position, snap: Dict) -> Optional[StrategySignal]:
        return None   # simple P&L exit handled by risk manager


class BearCallSpreadStrategy(BaseStrategy):
    """
    Bear Call Spread (Credit Call Spread).
    Sell lower strike call, buy higher strike call.
    Directional bearish bias with limited risk.
    """

    name = "bear_call_spread"

    def generate_signal(self, snap: Dict) -> Optional[StrategySignal]:
        if not self.is_entry_allowed(snap):
            return None

        regime = snap.get("regime", 0)
        iv_rank = snap.get("iv_rank", 0)
        chain_df = snap.get("chain")
        spot = snap.get("spot", 0)
        symbol = snap.get("symbol", "NIFTY")

        if chain_df is None or chain_df.empty:
            return None

        if regime != 1:    # 1 = bearish in our regime detector
            return None
        if iv_rank < 30:
            return None

        expiry = self._select_expiry(chain_df, self.config.strategy.preferred_dte)
        if not expiry:
            return None

        step = IronCondorStrategy._detect_strike_step(chain_df, symbol)
        short_call = self._find_strike_by_delta(chain_df, expiry, "CE", 0.30)
        long_call = short_call + step * 2

        def ltp(k):
            r = chain_df[(chain_df["type"] == "CE") &
                         (chain_df["strike"] == k) &
                         (chain_df["expiry"] == expiry)]
            return float(r.iloc[0]["ltp"]) if not r.empty else 0.0

        credit = ltp(short_call) - ltp(long_call)
        if credit < 20:
            return None

        return StrategySignal(
            strategy_name=self.name,
            signal=SignalType.ENTER_SHORT,
            symbol=symbol,
            legs=[
                {"type": "CE", "strike": short_call, "expiry": expiry, "action": "sell", "qty": 1},
                {"type": "CE", "strike": long_call, "expiry": expiry, "action": "buy", "qty": 1},
            ],
            confidence=0.6,
            expected_credit=credit,
            max_risk=step * 2 - credit,
            breakevens=[short_call + credit],
            notes=f"Bear call {short_call}/{long_call} credit={credit:.0f}",
        )

    def manage_position(self, position: Position, snap: Dict) -> Optional[StrategySignal]:
        return None


class MomentumDirectionalStrategy(BaseStrategy):
    """
    Momentum-driven long options (debit spreads).
    Triggered when strong trend detected + low IV (cheap options).
    Buys ATM or slightly OTM options in direction of trend.
    """

    name = "momentum_directional"

    def generate_signal(self, snap: Dict) -> Optional[StrategySignal]:
        if not self.is_entry_allowed(snap):
            return None

        regime = snap.get("regime", 0)
        iv_rank = snap.get("iv_rank", 100)
        vix = snap.get("vix", 15)
        chain_df = snap.get("chain")
        spot = snap.get("spot", 0)
        symbol = snap.get("symbol", "NIFTY")
        features = snap.get("features", {})

        if chain_df is None or chain_df.empty:
            return None

        # Only trade when trend is strong AND IV is LOW (cheap premiums)
        if regime == 0:
            return None
        if iv_rank > 50:  # too expensive for debit
            return None

        # Use short DTE for momentum (3-7 DTE)
        expiry = self._select_expiry(chain_df, 7)
        if not expiry:
            return None

        step = IronCondorStrategy._detect_strike_step(chain_df, symbol)
        is_bullish = regime == -1   # regime -1 = trend up

        opt_type = "CE" if is_bullish else "PE"
        # Buy ATM option
        atm = round(spot / step) * step

        def ltp(k):
            r = chain_df[(chain_df["type"] == opt_type) &
                         (chain_df["strike"] == k) &
                         (chain_df["expiry"] == expiry)]
            return float(r.iloc[0]["ltp"]) if not r.empty else 0.0

        entry_cost = ltp(atm)
        if entry_cost < 10:
            return None

        return StrategySignal(
            strategy_name=self.name,
            signal=SignalType.ENTER_LONG,
            symbol=symbol,
            legs=[
                {"type": opt_type, "strike": atm, "expiry": expiry, "action": "buy", "qty": 1},
            ],
            confidence=0.55,
            expected_credit=-entry_cost,   # negative = debit
            max_risk=entry_cost,
            breakevens=[atm + entry_cost if is_bullish else atm - entry_cost],
            notes=f"Momentum {'bull' if is_bullish else 'bear'} {opt_type}@{atm}",
        )

    def manage_position(self, position: Position, snap: Dict) -> Optional[StrategySignal]:
        return None

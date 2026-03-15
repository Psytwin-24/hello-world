"""
Base strategy interface and signal types.
All concrete strategies inherit from BaseStrategy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd


class SignalType(Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    NO_ACTION = "no_action"
    ADJUST = "adjust"


class StrategyState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"   # risk limit hit
    CLOSED = "closed"


@dataclass
class StrategySignal:
    strategy_name: str
    signal: SignalType
    symbol: str
    legs: List[Dict]            # each leg: {type, strike, expiry, action, qty}
    confidence: float = 0.5     # 0–1
    expected_credit: float = 0.0
    max_risk: float = 0.0
    breakevens: List[float] = field(default_factory=list)
    notes: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    strategy_name: str
    symbol: str
    legs: List[Dict]
    entry_credit: float         # premium received (positive = credit)
    entry_time: str = ""
    status: str = "open"        # open | closed | partial
    pnl: float = 0.0
    max_pnl: float = 0.0        # high-water mark for trailing exits
    adjustments: int = 0
    metadata: Dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    All options strategies inherit this.
    Implement generate_signal() and manage_position().
    """

    name: str = "base"

    def __init__(self, config):
        self.config = config
        self.state = StrategyState.IDLE
        self.positions: List[Position] = []

    @abstractmethod
    def generate_signal(self, market_snapshot: Dict) -> Optional[StrategySignal]:
        """
        Analyse market_snapshot and return a trading signal or None.
        market_snapshot contains: spot, chain_df, vix, pcr, iv_rank, regime, ...
        """

    @abstractmethod
    def manage_position(self, position: Position, market_snapshot: Dict) -> Optional[StrategySignal]:
        """
        Check if open position needs to be exited or adjusted.
        Return an exit/adjust signal or None.
        """

    def is_entry_allowed(self, market_snapshot: Dict) -> bool:
        """Global pre-checks before generating entry signals."""
        vix = market_snapshot.get("vix", 0)
        # Don't trade when VIX > 40 (extreme fear) unless specifically a vol strategy
        if vix > 40 and self.name not in ("vega_scalping", "long_straddle"):
            return False
        return True

    def _select_expiry(self, chain_df: pd.DataFrame, target_dte: int) -> str:
        """Pick the expiry closest to target DTE."""
        expiries = chain_df["expiry"].unique()
        best = None
        best_diff = float("inf")
        for exp in expiries:
            try:
                dt = pd.to_datetime(exp, dayfirst=True)
                dte = (dt - pd.Timestamp.now()).days
                if dte < 1:
                    continue
                if abs(dte - target_dte) < best_diff:
                    best_diff = abs(dte - target_dte)
                    best = exp
            except Exception:
                pass
        return best or (expiries[0] if len(expiries) > 0 else "")

    def _find_strike_by_delta(
        self, chain_df: pd.DataFrame, expiry: str, opt_type: str, target_delta: float
    ) -> float:
        """Find strike closest to a target absolute delta."""
        filtered = chain_df[
            (chain_df["expiry"] == expiry) &
            (chain_df["type"] == opt_type) &
            (chain_df["delta"].notna()) &
            (chain_df["delta"] != 0)
        ].copy()
        if filtered.empty:
            return 0.0
        filtered["delta_diff"] = (filtered["delta"].abs() - abs(target_delta)).abs()
        return float(filtered.sort_values("delta_diff").iloc[0]["strike"])

"""
Options backtesting engine.

Designed specifically for Indian options markets:
  - Event-driven simulation on historical option chain snapshots
  - Walk-forward testing to prevent look-ahead bias
  - Full Greeks P&L attribution
  - Transaction costs: brokerage + STT + stamp duty + exchange charges
  - Slippage modelling
  - Performance analytics: Sharpe, Sortino, Calmar, win rate, etc.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analytics.options_engine import BlackScholes, IVSolver, OptionSpec
from src.strategies.base import Position, StrategySignal, SignalType


# ---------------------------------------------------------------------------
# Transaction cost model for Indian markets
# ---------------------------------------------------------------------------

@dataclass
class TransactionCosts:
    """
    NSE/BSE option transaction costs (approximate, 2024 rates).
    All as fraction of turnover unless noted.
    """
    brokerage_per_order: float = 20.0       # Zerodha flat ₹20/order
    stt_sell_pct: float = 0.0625 / 100     # STT on sell side
    exchange_charge_pct: float = 0.053 / 100
    sebi_charge_pct: float = 0.0001 / 100
    stamp_duty_pct: float = 0.003 / 100    # on buy side only
    gst_pct: float = 0.18                  # GST on brokerage + exchange

    def total_cost(
        self, premium: float, lot_size: int, lots: int, action: str
    ) -> float:
        """Calculate total round-trip cost for one leg."""
        turnover = premium * lot_size * lots
        brokerage = self.brokerage_per_order
        exchange = turnover * self.exchange_charge_pct
        sebi = turnover * self.sebi_charge_pct
        gst = (brokerage + exchange) * self.gst_pct

        if action == "sell":
            stt = turnover * self.stt_sell_pct
        else:
            stt = 0.0
            stamp = turnover * self.stamp_duty_pct
            return brokerage + exchange + sebi + gst + stamp + stt

        return brokerage + exchange + sebi + gst + stt


@dataclass
class BacktestTrade:
    strategy: str
    symbol: str
    entry_date: str
    exit_date: str = ""
    legs: List[Dict] = field(default_factory=list)
    entry_premium: float = 0.0      # credit received (+ = credit, - = debit)
    exit_premium: float = 0.0       # cost to close
    gross_pnl: float = 0.0
    transaction_costs: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""
    dte_at_entry: int = 0
    iv_rank_at_entry: float = 0.0
    vix_at_entry: float = 0.0
    max_adverse_pnl: float = 0.0    # Maximum adverse excursion


@dataclass
class BacktestResults:
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    metrics: Dict

    def summary(self) -> str:
        m = self.metrics
        return (
            f"\n{'='*60}\n"
            f"  BACKTEST SUMMARY\n"
            f"{'='*60}\n"
            f"  Total Trades    : {m.get('total_trades', 0)}\n"
            f"  Win Rate        : {m.get('win_rate', 0):.1%}\n"
            f"  Net P&L         : ₹{m.get('net_pnl', 0):,.0f}\n"
            f"  Sharpe Ratio    : {m.get('sharpe', 0):.2f}\n"
            f"  Sortino Ratio   : {m.get('sortino', 0):.2f}\n"
            f"  Calmar Ratio    : {m.get('calmar', 0):.2f}\n"
            f"  Max Drawdown    : {m.get('max_drawdown', 0):.1%}\n"
            f"  Avg Win         : ₹{m.get('avg_win', 0):,.0f}\n"
            f"  Avg Loss        : ₹{m.get('avg_loss', 0):,.0f}\n"
            f"  Profit Factor   : {m.get('profit_factor', 0):.2f}\n"
            f"  Expectancy      : ₹{m.get('expectancy', 0):,.0f}\n"
            f"{'='*60}"
        )


class PerformanceAnalytics:
    """Compute strategy performance metrics from trade list."""

    @staticmethod
    def compute(
        trades: List[BacktestTrade],
        initial_capital: float = 1_000_000,
        risk_free_rate: float = 0.07,
    ) -> Dict:
        if not trades:
            return {}

        pnls = np.array([t.net_pnl for t in trades])
        equity = initial_capital + np.cumsum(pnls)
        returns = np.diff(equity) / equity[:-1]

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(drawdown.max())

        # Sharpe
        ann_factor = math.sqrt(252)
        daily_ret = returns.mean()
        daily_std = returns.std() if returns.std() > 0 else 1e-10
        sharpe = (daily_ret - risk_free_rate / 252) / daily_std * ann_factor

        # Sortino
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 and downside.std() > 0 else 1e-10
        sortino = (daily_ret - risk_free_rate / 252) / downside_std * ann_factor

        # Calmar
        ann_return = (equity[-1] / initial_capital) ** (252 / len(equity)) - 1
        calmar = ann_return / max_dd if max_dd > 0 else 0.0

        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else float("inf")
        expectancy = float(pnls.mean())

        return {
            "total_trades": len(trades),
            "win_rate": float(len(wins) / len(pnls)),
            "net_pnl": float(pnls.sum()),
            "gross_pnl": float(sum(t.gross_pnl for t in trades)),
            "total_costs": float(sum(t.transaction_costs for t in trades)),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "calmar": round(calmar, 3),
            "max_drawdown": round(max_dd, 4),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
            "profit_factor": round(profit_factor, 3),
            "expectancy": round(expectancy, 2),
            "ann_return": round(ann_return, 4),
            "final_capital": round(float(equity[-1]), 2),
        }


class OptionsBacktestEngine:
    """
    Event-driven options backtesting engine.

    Usage:
        engine = OptionsBacktestEngine(strategy_fn, config)
        results = engine.run(historical_snapshots)
    """

    def __init__(
        self,
        strategy_fn: Callable,           # function(snapshot) -> StrategySignal | None
        manage_fn: Callable,             # function(position, snapshot) -> StrategySignal | None
        config,
        initial_capital: float = 1_000_000,
        lot_size: int = 50,
        slippage_pct: float = 0.002,     # 0.2% slippage on fills
    ):
        self.strategy_fn = strategy_fn
        self.manage_fn = manage_fn
        self.config = config
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.slippage = slippage_pct
        self.costs = TransactionCosts()

    def run(
        self,
        snapshots: List[Dict],
        lots: int = 1,
    ) -> BacktestResults:
        """
        Run backtest on a time-ordered list of market snapshots.
        Each snapshot = output of MarketDataAggregator.get_full_market_snapshot()
        """
        capital = self.initial_capital
        equity_curve = []
        closed_trades: List[BacktestTrade] = []
        open_positions: Dict[str, BacktestTrade] = {}

        for snap in snapshots:
            ts = snap.get("timestamp", "")

            # Manage open positions first
            for pos_id, bt_trade in list(open_positions.items()):
                exit_signal = self._check_exit(bt_trade, snap)
                if exit_signal:
                    closed = self._close_trade(bt_trade, snap, exit_signal, lots)
                    capital += closed.net_pnl
                    closed_trades.append(closed)
                    del open_positions[pos_id]

            # Generate new entry signal
            signal = self.strategy_fn(snap)
            if signal and signal.signal in (SignalType.ENTER_SHORT, SignalType.ENTER_LONG):
                if len(open_positions) < self.config.strategy.max_open_positions:
                    bt_trade = self._open_trade(signal, snap, lots)
                    if bt_trade:
                        open_positions[f"{signal.strategy_name}_{ts}"] = bt_trade

            equity_curve.append(capital)

        # Close any remaining positions at last snapshot
        last_snap = snapshots[-1] if snapshots else {}
        for bt_trade in open_positions.values():
            closed = self._force_close(bt_trade, last_snap, lots)
            capital += closed.net_pnl
            closed_trades.append(closed)

        eq_series = pd.Series(equity_curve, name="equity")
        metrics = PerformanceAnalytics.compute(closed_trades, self.initial_capital)

        return BacktestResults(trades=closed_trades, equity_curve=eq_series, metrics=metrics)

    def _open_trade(self, signal: StrategySignal, snap: Dict, lots: int) -> Optional[BacktestTrade]:
        chain = snap.get("chain")
        if chain is None:
            return None

        total_premium = 0.0
        total_cost = 0.0

        for leg in signal.legs:
            row = chain[
                (chain["type"] == leg["type"]) &
                (chain["strike"] == leg["strike"]) &
                (chain["expiry"] == leg["expiry"])
            ]
            if row.empty:
                return None
            ltp = float(row.iloc[0]["ltp"])
            slipped = ltp * (1 + self.slippage if leg["action"] == "buy" else 1 - self.slippage)
            multiplier = -1 if leg["action"] == "buy" else 1
            total_premium += multiplier * slipped
            total_cost += self.costs.total_cost(slipped, self.lot_size, lots, leg["action"])

        expiry = signal.legs[0]["expiry"] if signal.legs else ""
        dte = 0
        if expiry:
            try:
                dte = (pd.to_datetime(expiry, dayfirst=True) - pd.Timestamp.now()).days
            except Exception:
                dte = 0

        return BacktestTrade(
            strategy=signal.strategy_name,
            symbol=signal.symbol,
            entry_date=snap.get("timestamp", ""),
            legs=signal.legs,
            entry_premium=total_premium,
            transaction_costs=total_cost,
            dte_at_entry=dte,
            iv_rank_at_entry=snap.get("iv_rank", 0),
            vix_at_entry=snap.get("vix", 0),
        )

    def _check_exit(self, trade: BacktestTrade, snap: Dict) -> Optional[str]:
        """Return exit reason or None."""
        chain = snap.get("chain")
        if chain is None:
            return None

        current_value = self._get_current_value(trade, chain)
        credit = trade.entry_premium

        if credit > 0:  # credit strategy
            pnl = credit - current_value
            pnl_pct = pnl / credit if credit > 0 else 0

            if pnl_pct >= self.config.strategy.profit_target_pct:
                return "profit_target"
            if current_value > credit * self.config.strategy.stop_loss_pct:
                return "stop_loss"
        else:  # debit strategy
            pnl = current_value + credit  # credit is negative
            if pnl > abs(credit) * 0.50:
                return "profit_target"
            if pnl < -abs(credit) * 0.5:
                return "stop_loss"

        # DTE exit
        expiry = trade.legs[0].get("expiry", "") if trade.legs else ""
        if expiry:
            try:
                dte = (pd.to_datetime(expiry, dayfirst=True) - pd.Timestamp.now()).days
                if dte <= self.config.strategy.dte_exit:
                    return "dte_exit"
            except Exception:
                pass

        return None

    def _get_current_value(self, trade: BacktestTrade, chain: pd.DataFrame) -> float:
        total = 0.0
        for leg in trade.legs:
            row = chain[
                (chain["type"] == leg["type"]) &
                (chain["strike"] == leg["strike"]) &
                (chain["expiry"] == leg["expiry"])
            ]
            if not row.empty:
                total += float(row.iloc[0]["ltp"])
        return total

    def _close_trade(
        self, trade: BacktestTrade, snap: Dict, reason: str, lots: int
    ) -> BacktestTrade:
        chain = snap.get("chain")
        exit_val = self._get_current_value(trade, chain) if chain is not None else 0.0

        # Slippage on exit
        exit_val *= (1 + self.slippage)

        gross = trade.entry_premium - exit_val
        exit_costs = sum(
            self.costs.total_cost(
                exit_val / max(len(trade.legs), 1), self.lot_size, lots, "buy"
            )
            for _ in trade.legs
        )
        total_costs = trade.transaction_costs + exit_costs
        net = gross - total_costs

        trade.exit_date = snap.get("timestamp", "")
        trade.exit_premium = exit_val
        trade.gross_pnl = gross * lots * self.lot_size
        trade.transaction_costs = total_costs
        trade.net_pnl = net * lots * self.lot_size
        trade.exit_reason = reason

        return trade

    def _force_close(self, trade: BacktestTrade, snap: Dict, lots: int) -> BacktestTrade:
        return self._close_trade(trade, snap, "end_of_data", lots)


class WalkForwardAnalyzer:
    """
    Walk-forward optimisation to prevent overfitting.

    Splits data into in-sample (IS) and out-of-sample (OOS) windows,
    optimises parameters on IS, validates on OOS, repeats.
    """

    def __init__(
        self,
        engine_factory: Callable,      # fn(params) -> OptionsBacktestEngine
        snapshots: List[Dict],
        config,
        n_periods: int = 12,
        train_ratio: float = 0.7,
    ):
        self.factory = engine_factory
        self.snapshots = snapshots
        self.config = config
        self.n_periods = n_periods
        self.train_ratio = train_ratio

    def run(self, param_grid: List[Dict]) -> Dict:
        """
        Run walk-forward over n_periods.
        Returns combined OOS metrics.
        """
        period_size = len(self.snapshots) // self.n_periods
        all_oos_trades = []

        for i in range(self.n_periods):
            start = i * period_size
            end = start + period_size
            chunk = self.snapshots[start:end]

            split = int(len(chunk) * self.train_ratio)
            train = chunk[:split]
            oos = chunk[split:]

            if not train or not oos:
                continue

            # Find best params on train
            best_params = self._optimise(train, param_grid)

            # Validate on OOS
            engine = self.factory(best_params)
            results = engine.run(oos)
            all_oos_trades.extend(results.trades)

            logger.info(
                f"WF Period {i+1}/{self.n_periods}: "
                f"OOS Sharpe={results.metrics.get('sharpe', 0):.2f} "
                f"Trades={len(results.trades)}"
            )

        combined_metrics = PerformanceAnalytics.compute(all_oos_trades, 1_000_000)
        return {"oos_metrics": combined_metrics, "oos_trades": len(all_oos_trades)}

    def _optimise(self, train_data: List[Dict], param_grid: List[Dict]) -> Dict:
        """Grid search for best Sharpe on training data."""
        best_sharpe = -float("inf")
        best_params = param_grid[0] if param_grid else {}

        for params in param_grid:
            try:
                engine = self.factory(params)
                results = engine.run(train_data)
                sharpe = results.metrics.get("sharpe", -999)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
            except Exception as e:
                logger.warning(f"Param {params} failed: {e}")

        return best_params

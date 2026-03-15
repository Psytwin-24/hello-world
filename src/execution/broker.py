"""
Broker abstraction layer.

Supports:
  - Zerodha (Kite Connect)
  - Fyers
  - Angel One (SmartAPI)
  - Paper trading (simulated fills)

All brokers expose the same interface: place_order(), cancel_order(), get_positions()
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Config
from src.strategies.base import StrategySignal


@dataclass
class Order:
    broker_order_id: str = ""
    symbol: str = ""
    exchange: str = "NFO"           # NFO for F&O
    tradingsymbol: str = ""         # e.g. NIFTY24DEC20000CE
    transaction_type: str = "BUY"   # BUY | SELL
    quantity: int = 0
    order_type: str = "LIMIT"       # MARKET | LIMIT | SL | SL-M
    price: float = 0.0
    trigger_price: float = 0.0
    product: str = "NRML"          # NRML for overnight options, MIS for intraday
    validity: str = "DAY"
    status: str = "PENDING"        # PENDING | OPEN | COMPLETE | REJECTED | CANCELLED
    filled_price: float = 0.0
    filled_qty: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_msg: str = ""


@dataclass
class Position:
    tradingsymbol: str
    exchange: str
    quantity: int           # +ve = long, -ve = short
    average_price: float
    ltp: float
    pnl: float
    product: str


class BaseBroker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Returns broker order ID."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Returns True if cancelled."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        pass

    @abstractmethod
    def get_funds(self) -> Dict:
        """Returns dict with 'available', 'used', 'total'."""

    @abstractmethod
    def get_ltp(self, tradingsymbol: str, exchange: str = "NFO") -> float:
        pass

    def place_signal(
        self,
        signal: StrategySignal,
        lots: int,
        lot_size: int,
        symbol_mapper,
    ) -> List[Order]:
        """
        Convert a StrategySignal into broker orders.
        Returns list of placed orders (one per leg).
        """
        orders = []
        for leg in signal.legs:
            tradingsymbol = symbol_mapper(
                signal.symbol, leg["strike"], leg["expiry"], leg["type"]
            )
            ltp = self.get_ltp(tradingsymbol)
            # Limit price: ±0.5% from LTP for better fills
            if leg["action"] == "buy":
                limit_price = round(ltp * 1.005, 1)
                txn = "BUY"
            else:
                limit_price = round(ltp * 0.995, 1)
                txn = "SELL"

            order = Order(
                symbol=signal.symbol,
                tradingsymbol=tradingsymbol,
                transaction_type=txn,
                quantity=lots * lot_size,
                order_type="LIMIT",
                price=limit_price,
                product="NRML",
            )
            order_id = self.place_order(order)
            order.broker_order_id = order_id
            orders.append(order)

        return orders


# ---------------------------------------------------------------------------
# Paper Trading Broker (for simulation)
# ---------------------------------------------------------------------------

class PaperBroker(BaseBroker):
    """
    Simulated broker. Fills all limit orders immediately at LTP.
    Perfect for testing strategies before going live.
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.capital = initial_capital
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self._order_counter = 0
        self._ltp_cache: Dict[str, float] = {}

    def set_ltp(self, tradingsymbol: str, price: float):
        """Feed price data for simulation."""
        self._ltp_cache[tradingsymbol] = price

    def get_ltp(self, tradingsymbol: str, exchange: str = "NFO") -> float:
        return self._ltp_cache.get(tradingsymbol, 0.0)

    def place_order(self, order: Order) -> str:
        self._order_counter += 1
        order_id = f"PAPER_{self._order_counter:06d}"
        order.broker_order_id = order_id
        order.filled_price = self.get_ltp(order.tradingsymbol) or order.price
        order.filled_qty = order.quantity
        order.status = "COMPLETE"

        # Update capital
        cost = order.filled_price * order.quantity
        if order.transaction_type == "BUY":
            self.capital -= cost
        else:
            self.capital += cost

        self.orders[order_id] = order
        logger.info(
            f"[PAPER] {order.transaction_type} {order.quantity} {order.tradingsymbol} "
            f"@ ₹{order.filled_price:.2f}"
        )
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = "CANCELLED"
            return True
        return False

    def get_order_status(self, order_id: str) -> Order:
        return self.orders.get(order_id, Order())

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_funds(self) -> Dict:
        return {
            "available": self.capital,
            "used": 0,
            "total": self.capital,
        }


# ---------------------------------------------------------------------------
# Zerodha (Kite Connect)
# ---------------------------------------------------------------------------

class ZerodhaBroker(BaseBroker):
    """Live broker via Zerodha Kite Connect SDK."""

    def __init__(self, config: Config):
        self.cfg = config.zerodha
        self._kite = None
        self._connect()

    def _connect(self):
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self.cfg.api_key)
            self._kite.set_access_token(self.cfg.access_token)
            logger.info("Zerodha Kite connected")
        except ImportError:
            logger.warning("kiteconnect not installed")
        except Exception as e:
            logger.error(f"Zerodha connection failed: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_order(self, order: Order) -> str:
        if not self._kite:
            raise RuntimeError("Kite not connected")
        order_id = self._kite.place_order(
            tradingsymbol=order.tradingsymbol,
            exchange=order.exchange,
            transaction_type=order.transaction_type,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            product=order.product,
            variety=self._kite.VARIETY_REGULAR,
        )
        logger.info(f"[ZERODHA] Order placed: {order_id}")
        return str(order_id)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._kite.cancel_order(
                variety=self._kite.VARIETY_REGULAR, order_id=order_id
            )
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_order_status(self, order_id: str) -> Order:
        try:
            orders = self._kite.orders()
            for o in orders:
                if str(o["order_id"]) == str(order_id):
                    return Order(
                        broker_order_id=order_id,
                        tradingsymbol=o["tradingsymbol"],
                        transaction_type=o["transaction_type"],
                        quantity=o["quantity"],
                        status=o["status"],
                        filled_price=o.get("average_price", 0),
                        filled_qty=o.get("filled_quantity", 0),
                    )
        except Exception as e:
            logger.error(f"Order status failed: {e}")
        return Order()

    def get_positions(self) -> List[Position]:
        try:
            pos = self._kite.positions()
            result = []
            for p in pos.get("net", []):
                result.append(Position(
                    tradingsymbol=p["tradingsymbol"],
                    exchange=p["exchange"],
                    quantity=p["quantity"],
                    average_price=p["average_price"],
                    ltp=p.get("last_price", 0),
                    pnl=p.get("pnl", 0),
                    product=p["product"],
                ))
            return result
        except Exception as e:
            logger.error(f"Positions failed: {e}")
            return []

    def get_funds(self) -> Dict:
        try:
            margins = self._kite.margins()
            equity = margins.get("equity", {})
            return {
                "available": equity.get("available", {}).get("live_balance", 0),
                "used": equity.get("utilised", {}).get("exposure", 0),
                "total": equity.get("net", 0),
            }
        except Exception as e:
            logger.error(f"Funds failed: {e}")
            return {"available": 0, "used": 0, "total": 0}

    def get_ltp(self, tradingsymbol: str, exchange: str = "NFO") -> float:
        try:
            instrument = f"{exchange}:{tradingsymbol}"
            quote = self._kite.ltp([instrument])
            return float(quote.get(instrument, {}).get("last_price", 0))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Broker factory
# ---------------------------------------------------------------------------

def get_broker(config: Config) -> BaseBroker:
    """Return appropriate broker based on config."""
    if config.paper_trading:
        logger.info("🔵 Paper trading mode active")
        return PaperBroker()

    broker = config.active_broker.lower()
    if broker == "zerodha":
        return ZerodhaBroker(config)
    elif broker == "fyers":
        logger.warning("Fyers broker not yet implemented — using paper")
        return PaperBroker()
    elif broker == "angel":
        logger.warning("Angel broker not yet implemented — using paper")
        return PaperBroker()
    else:
        raise ValueError(f"Unknown broker: {broker}")


# ---------------------------------------------------------------------------
# NSE symbol formatter
# ---------------------------------------------------------------------------

def format_nse_options_symbol(
    underlying: str,
    strike: float,
    expiry_str: str,
    opt_type: str,
) -> str:
    """
    Convert to NSE F&O tradingsymbol format.
    e.g. NIFTY + 20000 + 26DEC2024 + CE → NIFTY24DEC20000CE
    """
    try:
        import pandas as pd
        expiry = pd.to_datetime(expiry_str, dayfirst=True)
        month_map = {
            1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
            7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
        }
        # Weekly format: NIFTY + YY + MMM + DD + STRIKE + CE/PE
        # e.g. NIFTY24DEC2620000CE
        month = month_map[expiry.month]
        year2 = str(expiry.year)[-2:]
        day = expiry.day
        strike_str = str(int(strike))
        return f"{underlying}{year2}{month}{day}{strike_str}{opt_type}"
    except Exception:
        return f"{underlying}{int(strike)}{opt_type}"

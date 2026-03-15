"""
Market data layer — fetches real-time & historical data from NSE/BSE.

Sources (in priority order):
  1. Broker WebSocket (Zerodha/Fyers) — real-time ticks
  2. nsepython — free NSE scraper
  3. yfinance — historical OHLCV
  4. NSE India REST API — option chains, FII/DII data

All data is normalised into canonical DataFrames and cached in Redis.
"""

import asyncio
import json
import time
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import numpy as np
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import nsepython as nse
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False
    logger.warning("nsepython not installed — NSE scraping disabled")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ---------------------------------------------------------------------------
# NSE headers (required to bypass bot detection)
# ---------------------------------------------------------------------------
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

NSE_BASE = "https://www.nseindia.com/api"


class NSEDataClient:
    """Thin async wrapper around NSE India REST API."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cookies: dict = {}
        self._last_cookie_refresh: float = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=NSE_HEADERS)
        # Refresh cookies every 10 min
        if time.time() - self._last_cookie_refresh > 600:
            await self._refresh_cookies()
        return self._session

    async def _refresh_cookies(self):
        """Hit NSE homepage to grab session cookies."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get("https://www.nseindia.com/", headers=NSE_HEADERS) as r:
                    self._cookies = {k: v.value for k, v in r.cookies.items()}
                    self._last_cookie_refresh = time.time()
        except Exception as e:
            logger.warning(f"Cookie refresh failed: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_option_chain(self, symbol: str) -> Dict:
        """Fetch full option chain from NSE."""
        session = await self._get_session()
        url = f"{NSE_BASE}/option-chain-indices?symbol={symbol}" \
              if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY") \
              else f"{NSE_BASE}/option-chain-equities?symbol={symbol}"
        async with session.get(url, cookies=self._cookies, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            return await resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_quote(self, symbol: str) -> Dict:
        """Get live quote for index or equity."""
        session = await self._get_session()
        url = f"{NSE_BASE}/quote-equity?symbol={symbol}"
        async with session.get(url, cookies=self._cookies, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_fii_dii_data(self) -> Dict:
        """Fetch FII/DII cash + derivatives data."""
        session = await self._get_session()
        url = f"{NSE_BASE}/fiidiiTradeReact"
        async with session.get(url, cookies=self._cookies, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class OptionChainParser:
    """Parse NSE option chain JSON into structured DataFrames."""

    @staticmethod
    def parse(raw: Dict) -> Tuple[pd.DataFrame, float, str]:
        """
        Returns:
            df: Option chain DataFrame (one row per strike × expiry × type)
            spot: Current spot price
            expiry: Current/nearest expiry date string
        """
        records = raw.get("records", {})
        data = records.get("data", [])
        spot = records.get("underlyingValue", 0.0)
        expiry_dates = records.get("expiryDates", [])
        nearest_expiry = expiry_dates[0] if expiry_dates else ""

        rows = []
        for item in data:
            strike = item.get("strikePrice", 0)
            expiry = item.get("expiryDate", "")
            for opt_type in ("CE", "PE"):
                opt = item.get(opt_type, {})
                if not opt:
                    continue
                rows.append({
                    "strike": strike,
                    "expiry": expiry,
                    "type": opt_type,
                    "ltp": opt.get("lastPrice", 0),
                    "change": opt.get("change", 0),
                    "pct_change": opt.get("pChange", 0),
                    "volume": opt.get("totalTradedVolume", 0),
                    "oi": opt.get("openInterest", 0),
                    "oi_change": opt.get("changeinOpenInterest", 0),
                    "bid": opt.get("bidprice", 0),
                    "ask": opt.get("askPrice", 0),
                    "iv": opt.get("impliedVolatility", 0),
                    "delta": opt.get("delta", None),
                    "gamma": opt.get("gamma", None),
                    "theta": opt.get("theta", None),
                    "vega": opt.get("vega", None),
                })

        df = pd.DataFrame(rows)
        return df, spot, nearest_expiry


class HistoricalDataFetcher:
    """Fetch OHLCV history via yfinance (NSE suffix) and broker APIs."""

    # Map of our symbols to Yahoo Finance tickers
    YF_SYMBOL_MAP = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "SENSEX": "^BSESN",
    }

    @staticmethod
    def get_yf_ticker(symbol: str) -> str:
        return HistoricalDataFetcher.YF_SYMBOL_MAP.get(symbol, f"{symbol}.NS")

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
        """
        if not YF_AVAILABLE:
            raise RuntimeError("yfinance not installed")
        ticker = self.get_yf_ticker(symbol)
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            logger.warning(f"No data returned for {symbol} ({ticker})")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df.rename(columns={"adj close": "adj_close"})
        df["symbol"] = symbol
        return df

    def fetch_multi(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        return {s: self.fetch_ohlcv(s, **kwargs) for s in symbols}


class VIXFetcher:
    """Fetch India VIX — the NSE fear gauge."""

    @staticmethod
    def get_india_vix() -> float:
        if not YF_AVAILABLE:
            return 0.0
        try:
            ticker = yf.Ticker("^INDIAVIX")
            hist = ticker.history(period="1d")
            return float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
            return 0.0

    @staticmethod
    async def get_india_vix_nse() -> float:
        """Scrape India VIX directly from NSE."""
        url = f"{NSE_BASE}/allIndices"
        try:
            async with aiohttp.ClientSession(headers=NSE_HEADERS) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    for item in data.get("data", []):
                        if item.get("index") == "INDIA VIX":
                            return float(item.get("last", 0))
        except Exception as e:
            logger.warning(f"NSE VIX fetch failed: {e}")
        return 0.0


class PCRAnalyzer:
    """Put-Call Ratio analysis for market sentiment."""

    @staticmethod
    def compute_pcr(option_chain_df: pd.DataFrame, by: str = "oi") -> Dict:
        """
        Compute PCR by OI or volume across strikes.
        by: 'oi' | 'volume'
        """
        col = "oi" if by == "oi" else "volume"
        puts = option_chain_df[option_chain_df["type"] == "PE"][col].sum()
        calls = option_chain_df[option_chain_df["type"] == "CE"][col].sum()

        pcr = puts / calls if calls > 0 else 0.0
        sentiment = "bullish" if pcr < 0.8 else ("bearish" if pcr > 1.2 else "neutral")

        return {
            "pcr": round(pcr, 4),
            "put_oi": int(puts),
            "call_oi": int(calls),
            "sentiment": sentiment,
        }

    @staticmethod
    def find_max_pain(option_chain_df: pd.DataFrame, spot: float) -> float:
        """
        Max pain = strike where total option buyers lose the most.
        This is where most open interest sits; price tends to gravitate here near expiry.
        """
        strikes = option_chain_df["strike"].unique()
        pain = {}
        for s in strikes:
            call_pain = option_chain_df[
                (option_chain_df["type"] == "CE") & (option_chain_df["strike"] <= s)
            ]["oi"].sum() * max(0, s - spot)
            put_pain = option_chain_df[
                (option_chain_df["type"] == "PE") & (option_chain_df["strike"] >= s)
            ]["oi"].sum() * max(0, spot - s)
            pain[s] = call_pain + put_pain
        return min(pain, key=pain.get) if pain else spot

    @staticmethod
    def find_support_resistance_via_oi(
        option_chain_df: pd.DataFrame, top_n: int = 3
    ) -> Dict[str, List[float]]:
        """
        Highest OI strikes = key support/resistance levels.
        Calls = resistance, Puts = support.
        """
        call_oi = (
            option_chain_df[option_chain_df["type"] == "CE"]
            .groupby("strike")["oi"]
            .sum()
            .nlargest(top_n)
        )
        put_oi = (
            option_chain_df[option_chain_df["type"] == "PE"]
            .groupby("strike")["oi"]
            .sum()
            .nlargest(top_n)
        )
        return {
            "resistance": sorted(call_oi.index.tolist()),
            "support": sorted(put_oi.index.tolist()),
        }


class MarketDataAggregator:
    """
    Top-level aggregator — single entry point for all market data.
    Caches results in-memory; plug in Redis for production multi-process use.
    """

    def __init__(self):
        self.nse_client = NSEDataClient()
        self.hist_fetcher = HistoricalDataFetcher()
        self.parser = OptionChainParser()
        self._cache: Dict = {}

    async def get_full_market_snapshot(self, symbol: str) -> Dict:
        """One call → everything needed by the strategy engine."""
        raw_chain = await self.nse_client.get_option_chain(symbol)
        chain_df, spot, expiry = self.parser.parse(raw_chain)

        vix = await VIXFetcher.get_india_vix_nse()
        pcr = PCRAnalyzer.compute_pcr(chain_df)
        max_pain = PCRAnalyzer.find_max_pain(chain_df, spot)
        sr = PCRAnalyzer.find_support_resistance_via_oi(chain_df)

        return {
            "symbol": symbol,
            "spot": spot,
            "expiry": expiry,
            "vix": vix,
            "pcr": pcr,
            "max_pain": max_pain,
            "support_resistance": sr,
            "chain": chain_df,
            "timestamp": datetime.now().isoformat(),
        }

    def get_historical_ohlcv(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        key = f"{symbol}_{period}"
        if key not in self._cache:
            self._cache[key] = self.hist_fetcher.fetch_ohlcv(symbol, period=period)
        return self._cache[key]

    async def close(self):
        await self.nse_client.close()

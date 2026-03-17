"""
Kite API-based option chain builder.
Fallback when NSE website blocks cloud/data-center IPs.
Produces the same DataFrame schema as OptionChainParser.parse().
"""
import os
import time
from datetime import datetime, date
from typing import Tuple, Optional

import pandas as pd
from kiteconnect import KiteConnect
from loguru import logger


class KiteChainBuilder:
      """Build option-chain DataFrames from Zerodha Kite API."""
  
    def __init__(self):
              api_key = os.environ.get("ZERODHA_API_KEY", "")
              access_token = os.environ.get("ZERODHA_ACCESS_TOKEN", "")
              self._kite = KiteConnect(api_key=api_key)
              self._kite.set_access_token(access_token)
              self._instruments_cache = None
              self._instruments_ts = 0
      
    def _load_instruments(self):
              """Cache NFO instruments for the day (refreshed every 6 hours)."""
              now = time.time()
              if self._instruments_cache is not None and (now - self._instruments_ts) < 21600:
                            return self._instruments_cache
                        logger.info("KiteChainBuilder: fetching NFO instruments...")
        raw = self._kite.instruments("NFO")
        self._instruments_cache = raw
        self._instruments_ts = now
        logger.info(f"KiteChainBuilder: loaded {len(raw)} NFO instruments")
        return raw

    def get_option_chain(self, symbol, num_strikes=20):
              instruments = self._load_instruments()
        opts = [i for i in instruments if i["name"] == symbol and i["instrument_type"] in ("CE", "PE") and i["expiry"] >= date.today()]
        if not opts:
                      raise ValueError(f"No options found for {symbol}")
                  expiries = sorted(set(i["expiry"] for i in opts))
        nearest_exp = expiries[0]
        nearest_opts = [i for i in opts if i["expiry"] == nearest_exp]
        idx_map = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK", "FINNIFTY": "NSE:NIFTY FIN SERVICE", "MIDCPNIFTY": "NSE:NIFTY MID SELECT"}
        idx_symbol = idx_map.get(symbol, f"NSE:{symbol}")
        try:
                      qt = self._kite.quote([idx_symbol])
                      spot = qt[idx_symbol]["last_price"]
except Exception as e:
            logger.warning(f"Could not get spot for {idx_symbol}: {e}")
            strikes = sorted(set(i["strike"] for i in nearest_opts))
            spot = strikes[len(strikes) // 2]
        all_strikes = sorted(set(i["strike"] for i in nearest_opts))
        atm_idx = min(range(len(all_strikes)), key=lambda j: abs(all_strikes[j] - spot))
        lo = max(0, atm_idx - num_strikes)
        hi = min(len(all_strikes), atm_idx + num_strikes + 1)
        selected_strikes = set(all_strikes[lo:hi])
        selected = [i for i in nearest_opts if i["strike"] in selected_strikes]
        trading_syms = [f"NFO:{i['tradingsymbol']}" for i in selected]
        quotes = {}
        for j in range(0, len(trading_syms), 200):
                      chunk = trading_syms[j:j + 200]
                      try:
                                        q = self._kite.quote(chunk)
                                        quotes.update(q)
except Exception as e:
                logger.warning(f"Kite quote batch error: {e}")
        rows = []
        for inst in selected:
                      ts = f"NFO:{inst['tradingsymbol']}"
                      q = quotes.get(ts, {})
                      ohlc = q.get("ohlc", {})
                      rows.append({"strike": inst["strike"], "expiry": nearest_exp.strftime("%d-%b-%Y"), "type": inst["instrument_type"], "ltp": q.get("last_price", 0), "change": q.get("net_change", 0), "pct_change": ((q.get("last_price", 0) - ohlc.get("close", 0)) / ohlc.get("close", 1) * 100) if ohlc.get("close") else 0, "volume": q.get("volume_traded", 0) or q.get("volume", 0), "oi": q.get("oi", 0), "oi_change": (q.get("oi_day_high", 0) - q.get("oi_day_low", 0)) if q.get("oi_day_high") else 0, "bid": q.get("depth", {}).get("buy", [{}])[0].get("price", 0) if q.get("depth") else 0, "ask": q.get("depth", {}).get("sell", [{}])[0].get("price", 0) if q.get("depth") else 0, "iv": 0, "delta": None, "gamma": None, "theta": None, "vega": None})
                  df = pd.DataFrame(rows)
        expiry_str = nearest_exp.strftime("%d-%b-%Y")
        logger.info(f"KiteChainBuilder: built chain for {symbol} -- {len(df)} rows, spot={spot}, expiry={expiry_str}")
        return df, spot, expiry_strtest123

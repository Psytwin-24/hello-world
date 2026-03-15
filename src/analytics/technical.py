"""
Technical analysis indicator library.
All functions return float or pd.Series. No side effects.
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Vectorised TA computations on OHLCV DataFrames."""

    # --- Trend ---

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """SuperTrend — trend-following indicator widely used in Indian markets."""
        hl2 = (df["high"] + df["low"]) / 2
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(1, len(df)):
            prev_upper = upper.iloc[i - 1]
            prev_lower = lower.iloc[i - 1]
            prev_close = df["close"].iloc[i - 1]
            cur_close = df["close"].iloc[i]

            upper.iloc[i] = upper.iloc[i] if upper.iloc[i] < prev_upper or prev_close > prev_upper else prev_upper
            lower.iloc[i] = lower.iloc[i] if lower.iloc[i] > prev_lower or prev_close < prev_lower else prev_lower

            if i == 1:
                direction.iloc[i] = 1
            elif supertrend.iloc[i - 1] == prev_upper:
                direction.iloc[i] = -1 if cur_close > upper.iloc[i] else 1
            else:
                direction.iloc[i] = 1 if cur_close < lower.iloc[i] else -1

            supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == -1 else upper.iloc[i]

        return direction  # -1 = bullish, 1 = bearish

    @staticmethod
    def macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})

    # --- Momentum ---

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def stochastic(
        df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        low_min = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        d = k.rolling(d_period).mean()
        return pd.DataFrame({"%K": k, "%D": d})

    # --- Volatility ---

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def bollinger_bands(
        close: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        return pd.DataFrame({
            "upper": mid + std_dev * std,
            "mid": mid,
            "lower": mid - std_dev * std,
            "bandwidth": (2 * std_dev * std) / mid,
            "pct_b": (close - (mid - std_dev * std)) / (2 * std_dev * std),
        })

    @staticmethod
    def keltner_channels(
        df: pd.DataFrame, period: int = 20, multiplier: float = 1.5
    ) -> pd.DataFrame:
        mid = TechnicalIndicators.ema(df["close"], period)
        atr_val = TechnicalIndicators.atr(df, period)
        return pd.DataFrame({
            "upper": mid + multiplier * atr_val,
            "mid": mid,
            "lower": mid - multiplier * atr_val,
        })

    # --- Volume ---

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical * df["volume"]).cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        sign = np.sign(close.diff().fillna(0))
        return (sign * volume).cumsum()

    # --- Composite signals ---

    @staticmethod
    def regime_detector(df: pd.DataFrame) -> pd.Series:
        """
        Classify market regime: 1=trending up, -1=trending down, 0=ranging.
        Uses ADX + EMA crossover.
        """
        adx = TechnicalIndicators._adx(df)
        ema50 = TechnicalIndicators.ema(df["close"], 50)
        ema200 = TechnicalIndicators.ema(df["close"], 200)
        trending = adx > 25
        regime = pd.Series(0, index=df.index)
        regime[trending & (ema50 > ema200)] = 1
        regime[trending & (ema50 < ema200)] = -1
        return regime

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        dm_plus = (df["high"] - df["high"].shift()).clip(lower=0)
        dm_minus = (df["low"].shift() - df["low"]).clip(lower=0)
        dm_plus[dm_plus < dm_minus] = 0
        dm_minus[dm_minus < dm_plus] = 0
        atr_smooth = tr.ewm(span=period, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr_smooth
        di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr_smooth
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
        return dx.ewm(span=period, adjust=False).mean()

    @staticmethod
    def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a rich feature matrix for ML models.
        Adds ~40 features covering trend, momentum, volatility, volume.
        """
        out = df.copy()
        c = df["close"]

        # Trend
        for p in [5, 10, 20, 50, 200]:
            out[f"ema_{p}"] = TechnicalIndicators.ema(c, p)
            out[f"sma_{p}"] = TechnicalIndicators.sma(c, p)
            out[f"ret_{p}"] = c.pct_change(p)

        # Momentum
        out["rsi_14"] = TechnicalIndicators.rsi(c, 14)
        out["rsi_7"] = TechnicalIndicators.rsi(c, 7)
        macd = TechnicalIndicators.macd(c)
        out["macd"] = macd["macd"]
        out["macd_hist"] = macd["histogram"]
        stoch = TechnicalIndicators.stochastic(df)
        out["stoch_k"] = stoch["%K"]
        out["stoch_d"] = stoch["%D"]

        # Volatility
        bb = TechnicalIndicators.bollinger_bands(c)
        out["bb_bw"] = bb["bandwidth"]
        out["bb_pct"] = bb["pct_b"]
        out["atr_14"] = TechnicalIndicators.atr(df, 14)
        out["atr_pct"] = out["atr_14"] / c

        # Volume
        if "volume" in df.columns:
            out["vwap"] = TechnicalIndicators.vwap(df)
            out["obv"] = TechnicalIndicators.obv(c, df["volume"])
            out["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

        # Price patterns
        out["body_size"] = (df["close"] - df["open"]).abs() / df["open"]
        out["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"]
        out["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"]

        # Regime
        out["regime"] = TechnicalIndicators.regime_detector(df)

        return out.dropna()

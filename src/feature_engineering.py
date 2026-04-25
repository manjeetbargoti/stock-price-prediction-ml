"""
src/feature_engineering.py
---------------------------
FeatureEngineer — computes technical indicators, lag features,
rolling statistics, and (optionally) merges daily sentiment scores.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.logger import get_logger

log = get_logger(__name__)


class FeatureEngineer:
    """Transform a clean OHLCV DataFrame into a rich feature matrix."""

    def __init__(
        self,
        sma_windows:      list[int] = config.SMA_WINDOWS,
        ema_windows:      list[int] = config.EMA_WINDOWS,
        lag_periods:      list[int] = config.LAG_PERIODS,
        rolling_windows:  list[int] = config.ROLLING_WINDOWS,
    ):
        self.sma_windows     = sma_windows
        self.ema_windows     = ema_windows
        self.lag_periods     = lag_periods
        self.rolling_windows = rolling_windows

    # ── Main Entry Point ─────────────────────────────────────────────────────

    def build_features(
        self,
        df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Full feature pipeline.
        Parameters
        ----------
        df : cleaned OHLCV DataFrame (from DataPreprocessor.clean)
        sentiment_df : optional DataFrame indexed by date with column
                       'sentiment_mean' (float, −1 to 1)
        Returns
        -------
        DataFrame with all features and no NaN rows.
        """
        feat = df.copy()
        feat = self._add_trend_indicators(feat)
        feat = self._add_momentum_indicators(feat)
        feat = self._add_volatility_indicators(feat)
        feat = self._add_volume_indicators(feat)
        feat = self._add_lag_features(feat)
        feat = self._add_rolling_stats(feat)
        feat = self._add_time_features(feat)

        if sentiment_df is not None:
            feat = self._merge_sentiment(feat, sentiment_df)

        before = len(feat)
        feat.dropna(inplace=True)
        log.info(
            "Features built: %d columns, %d rows (dropped %d NaN rows).",
            len(feat.columns), len(feat), before - len(feat),
        )
        return feat

    def get_feature_names(self, with_sentiment: bool = False) -> list[str]:
        """Return the list of feature column names (excluding target)."""
        base = (
            ["Open", "High", "Low", "Volume", "Daily_Return"]
            + [f"sma_{w}" for w in self.sma_windows]
            + [f"ema_{w}" for w in self.ema_windows]
            + ["macd", "macd_signal", "macd_diff"]
            + ["rsi"]
            + ["stoch_k", "stoch_d"]
            + ["bb_upper", "bb_lower", "bb_width", "bb_pct"]
            + ["atr"]
            + ["obv", "vwap", "volume_ratio"]
            + [f"close_lag{p}" for p in self.lag_periods]
            + [f"return_lag{p}" for p in self.lag_periods]
            + [f"roll_mean_{w}" for w in self.rolling_windows]
            + [f"roll_std_{w}" for w in self.rolling_windows]
            + ["day_of_week", "month", "quarter"]
        )
        if with_sentiment:
            base += ["sentiment_mean", "sentiment_std",
                     "positive_ratio", "negative_ratio"]
        return base

    # ── Trend Indicators ─────────────────────────────────────────────────────

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        # Simple Moving Averages
        for w in self.sma_windows:
            df[f"sma_{w}"] = close.rolling(w).mean()

        # Exponential Moving Averages
        for w in self.ema_windows:
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

        # MACD
        ema_fast   = close.ewm(span=config.MACD_FAST,   adjust=False).mean()
        ema_slow   = close.ewm(span=config.MACD_SLOW,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal     = macd_line.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        df["macd"]        = macd_line
        df["macd_signal"] = signal
        df["macd_diff"]   = macd_line - signal

        return df

    # ── Momentum Indicators ───────────────────────────────────────────────────

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # RSI
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
        loss   = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
        rs     = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator %K and %D
        n    = config.STOCH_PERIOD
        l_n  = low.rolling(n).min()
        h_n  = high.rolling(n).max()
        diff = h_n - l_n
        df["stoch_k"] = 100 * (close - l_n) / diff.replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        return df

    # ── Volatility Indicators ─────────────────────────────────────────────────

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # Bollinger Bands
        w = config.BOLLINGER_WINDOW
        s = config.BOLLINGER_STD
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        df["bb_upper"] = sma + s * std
        df["bb_lower"] = sma - s * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma.replace(0, np.nan)
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"]   = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)

        # Average True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(config.ATR_PERIOD).mean()

        return df

    # ── Volume Indicators ─────────────────────────────────────────────────────

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["Close"]
        volume = df["Volume"]

        # On-Balance Volume (OBV)
        direction = np.sign(close.diff()).fillna(0)
        df["obv"] = (direction * volume).cumsum()

        # Volume Weighted Average Price (approximated over a rolling window)
        tp = (df["High"] + df["Low"] + close) / 3          # typical price
        df["vwap"] = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()

        # Volume ratio vs 20-day average
        vol_ma = volume.rolling(20).mean()
        df["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

        return df

    # ── Lag Features ──────────────────────────────────────────────────────────

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for p in self.lag_periods:
            df[f"close_lag{p}"]  = df["Close"].shift(p)
            df[f"return_lag{p}"] = df["Daily_Return"].shift(p)
        return df

    # ── Rolling Statistics ────────────────────────────────────────────────────

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        for w in self.rolling_windows:
            df[f"roll_mean_{w}"] = close.rolling(w).mean()
            df[f"roll_std_{w}"]  = close.rolling(w).std()
        return df

    # ── Calendar / Time Features ──────────────────────────────────────────────

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        df["day_of_week"] = df.index.dayofweek.astype(float)
        df["month"]       = df.index.month.astype(float)
        df["quarter"]     = df.index.quarter.astype(float)
        return df

    # ── Sentiment Merge ───────────────────────────────────────────────────────

    @staticmethod
    def _merge_sentiment(
        df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Left-join sentiment scores; forward-fill gaps with decay."""
        df = df.join(sentiment_df, how="left")

        sentiment_cols = [c for c in sentiment_df.columns if c in df.columns]
        for col in sentiment_cols:
            # Apply exponential decay while forward-filling
            series = df[col].copy()
            last_val = 0.0
            for i in range(len(series)):
                if pd.isna(series.iloc[i]):
                    last_val *= config.SENTIMENT_DECAY
                    series.iloc[i] = last_val
                else:
                    last_val = series.iloc[i]
            df[col] = series

        log.info("Sentiment merged: %s", sentiment_cols)
        return df


# ── Sequence Builder (for LSTM) ───────────────────────────────────────────────

def create_sequences(
    feature_array: np.ndarray,
    target_array:  np.ndarray,
    look_back:     int = config.LOOK_BACK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of `look_back` over the feature array.

    Parameters
    ----------
    feature_array : shape (n_samples, n_features) — scaled features
    target_array  : shape (n_samples,) — target Close prices
    look_back     : number of past time steps per sample

    Returns
    -------
    X : shape (n_samples - look_back, look_back, n_features)
    y : shape (n_samples - look_back,)
    """
    X, y = [], []
    for i in range(look_back, len(feature_array)):
        X.append(feature_array[i - look_back : i])
        y.append(target_array[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

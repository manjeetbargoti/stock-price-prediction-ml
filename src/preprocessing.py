"""
src/preprocessing.py
---------------------
DataPreprocessor — cleans raw OHLCV data, handles outliers,
scales features, and creates chronological train/val/test splits.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.logger import get_logger

log = get_logger(__name__)


class DataPreprocessor:
    """Clean, scale, and split a raw OHLCV DataFrame."""

    def __init__(
        self,
        train_ratio: float = config.TRAIN_RATIO,
        val_ratio:   float = config.VAL_RATIO,
        outlier_std: float = config.OUTLIER_STD_THRESHOLD,
    ):
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.outlier_std = outlier_std
        self.scaler: MinMaxScaler | None = None
        self._feature_cols: list[str] = []

    # ── Main Pipeline ─────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps, remove outliers, add basic derived columns.
        Returns a clean copy — does NOT scale.
        """
        df = df.copy()
        df = df.sort_index()

        # Forward-fill price; zero-fill volume (non-trading days)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].ffill().bfill()
        df["Volume"] = df["Volume"].fillna(0)

        # Drop duplicate dates
        df = df[~df.index.duplicated(keep="first")]

        # Treat extreme price outliers (data errors, not real events)
        df = self._winsorise(df, "Close")
        df = self._winsorise(df, "Volume")

        # Daily return (used downstream as a feature)
        df["Daily_Return"] = df["Close"].pct_change()

        log.info("Cleaned DataFrame: %d rows, NaN remaining: %d",
                 len(df), df.isna().sum().sum())
        return df

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological train / validation / test split.
        Returns (train_df, val_df, test_df).
        """
        n         = len(df)
        train_end = int(n * self.train_ratio)
        val_end   = int(n * (self.train_ratio + self.val_ratio))

        train = df.iloc[:train_end].copy()
        val   = df.iloc[train_end:val_end].copy()
        test  = df.iloc[val_end:].copy()

        log.info(
            "Split → train=%d [%s→%s]  val=%d [%s→%s]  test=%d [%s→%s]",
            len(train), train.index[0].date(), train.index[-1].date(),
            len(val),   val.index[0].date(),   val.index[-1].date(),
            len(test),  test.index[0].date(),  test.index[-1].date(),
        )
        return train, val, test

    def fit_scale(
        self, train_df: pd.DataFrame, feature_cols: list[str]
    ) -> pd.DataFrame:
        """Fit MinMaxScaler on training data only.  Returns scaled copy."""
        self._feature_cols = feature_cols
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = train_df.copy()
        scaled[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
        log.info("Scaler fitted on %d training rows.", len(train_df))
        return scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted scaler to val/test data."""
        if self.scaler is None:
            raise RuntimeError("Call fit_scale() before transform().")
        out = df.copy()
        out[self._feature_cols] = self.scaler.transform(df[self._feature_cols])
        return out

    def inverse_transform_prices(self, scaled_prices: np.ndarray) -> np.ndarray:
        """
        Inverse-transform a 1-D array of scaled Close prices.
        Assumes 'Close' is in self._feature_cols.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted.")
        close_idx = self._feature_cols.index("Close")
        dummy = np.zeros((len(scaled_prices), len(self._feature_cols)))
        dummy[:, close_idx] = scaled_prices.ravel()
        return self.scaler.inverse_transform(dummy)[:, close_idx]

    def save_scaler(self, path: str) -> None:
        joblib.dump(self.scaler, path)
        log.info("Scaler saved → %s", path)

    def load_scaler(self, path: str) -> None:
        self.scaler = joblib.load(path)
        log.info("Scaler loaded ← %s", path)

    # ── Private ───────────────────────────────────────────────────────────────

    def _winsorise(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Replace extreme values (beyond N std) with rolling median."""
        roll_mean = df[col].rolling(30, min_periods=1).mean()
        roll_std  = df[col].rolling(30, min_periods=1).std().fillna(1)
        upper = roll_mean + self.outlier_std * roll_std
        lower = roll_mean - self.outlier_std * roll_std
        mask  = (df[col] > upper) | (df[col] < lower)
        if mask.any():
            log.warning("Winsorised %d outliers in column '%s'.", mask.sum(), col)
            roll_median = df[col].rolling(30, min_periods=1).median()
            df.loc[mask, col] = roll_median[mask]
        return df

"""
src/data_collection.py
-----------------------
StockDataFetcher — downloads historical OHLCV data from Yahoo Finance,
validates it, caches to disk, and optionally persists to the SQLite DB.
"""

import time
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import pandas as pd
import yfinance as yf

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.logger import get_logger

log = get_logger(__name__)


class StockDataFetcher:
    """Download and cache OHLCV data from Yahoo Finance."""

    REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}

    def __init__(
        self,
        ticker: str,
        start_date: str = config.DEFAULT_START_DATE,
        end_date: str   = config.DEFAULT_END_DATE,
        interval: str   = config.DATA_INTERVAL,
        use_cache: bool = True,
    ):
        self.ticker     = ticker.upper()
        self.yf_ticker  = f"{self.ticker}{config.EXCHANGE_SUFFIX}"
        self.start_date = start_date
        self.end_date   = end_date
        self.interval   = interval
        self.use_cache  = use_cache
        self._cache_path = (
            config.RAW_DIR / f"{self.ticker}_{start_date}_{end_date}.parquet"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
        """Return a validated OHLCV DataFrame.  Reads cache when available."""
        if self.use_cache and self._cache_path.exists():
            log.info("Loading cached data: %s", self._cache_path.name)
            df = pd.read_parquet(self._cache_path)
            self._validate(df)
            return df

        log.info("Fetching %s from Yahoo Finance (%s → %s)",
                 self.yf_ticker, self.start_date, self.end_date)

        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(
                    self.yf_ticker,
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                    auto_adjust=True,
                    progress=False,
                )
                break
            except Exception as exc:
                log.warning("Attempt %d failed: %s", attempt, exc)
                if attempt == retries:
                    raise RuntimeError(
                        f"Failed to fetch {self.ticker} after {retries} attempts."
                    ) from exc
                time.sleep(delay)

        if raw.empty:
            raise ValueError(f"No data returned for {self.yf_ticker}.")

        df = self._clean_columns(raw)
        self._validate(df)

        if self.use_cache:
            df.to_parquet(self._cache_path)
            log.info("Cached %d rows → %s", len(df), self._cache_path.name)

        return df

    def get_company_info(self) -> dict:
        """Return basic company metadata from yfinance."""
        try:
            info = yf.Ticker(self.yf_ticker).info
            keys = ["longName", "sector", "industry", "marketCap",
                    "trailingPE", "dividendYield"]
            return {k: info.get(k) for k in keys}
        except Exception as exc:
            log.warning("Could not fetch company info: %s", exc)
            return {}

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns and keep OHLCV only."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Raise on bad data; warn on minor issues."""
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if len(df) < config.MIN_REQUIRED_ROWS:
            raise ValueError(
                f"Only {len(df)} rows — need ≥ {config.MIN_REQUIRED_ROWS}."
            )

        if (df["Close"] <= 0).any():
            bad = (df["Close"] <= 0).sum()
            raise ValueError(f"{bad} non-positive Close prices found.")

        nan_pct = df.isna().mean().max() * 100
        if nan_pct > 5:
            log.warning("High NaN ratio (%.1f%%) — check data quality.", nan_pct)

        log.info(
            "Validated %s: %d rows, %s → %s",
            self.ticker, len(df),
            df.index[0].date(), df.index[-1].date(),
        )


# ── Convenience wrapper ───────────────────────────────────────────────────────

def fetch_multiple(
    tickers: list[str],
    start_date: str = config.DEFAULT_START_DATE,
    end_date: str   = config.DEFAULT_END_DATE,
) -> dict[str, pd.DataFrame]:
    """Fetch several tickers; return {ticker: df} — skips failures."""
    result = {}
    for t in tickers:
        try:
            result[t] = StockDataFetcher(t, start_date, end_date).fetch()
        except Exception as exc:
            log.error("Skipping %s: %s", t, exc)
    return result

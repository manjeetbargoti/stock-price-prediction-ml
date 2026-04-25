"""
src/database.py
----------------
Lightweight SQLite persistence layer.
Stores raw prices, computed features, and model predictions.
"""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.logger import get_logger

log = get_logger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS stock_prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open_price  REAL NOT NULL,
    high_price  REAL NOT NULL,
    low_price   REAL NOT NULL,
    close_price REAL NOT NULL,
    volume      INTEGER NOT NULL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    date            TEXT NOT NULL,
    actual_price    REAL,
    predicted_price REAL NOT NULL,
    mae             REAL,
    rmse            REAL,
    r2_score        REAL,
    trained_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sp_ticker_date
    ON stock_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_mp_model
    ON model_predictions(ticker, model_name);
"""


# ── Connection Context Manager ────────────────────────────────────────────────

@contextmanager
def _conn(db_path: str | Path = config.DB_PATH):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Public API ────────────────────────────────────────────────────────────────

def initialise(db_path: str | Path = config.DB_PATH) -> None:
    """Create all tables if they do not exist."""
    with _conn(db_path) as conn:
        conn.executescript(_DDL)
    log.info("Database initialised at %s", db_path)


def save_prices(df: pd.DataFrame, ticker: str,
                db_path: str | Path = config.DB_PATH) -> int:
    """
    Upsert an OHLCV DataFrame into stock_prices.
    Returns number of rows inserted.
    """
    rows = [
        (ticker, str(idx.date()), row["Open"], row["High"],
         row["Low"], row["Close"], int(row["Volume"]))
        for idx, row in df.iterrows()
    ]
    sql = """
        INSERT OR REPLACE INTO stock_prices
            (ticker, date, open_price, high_price, low_price, close_price, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with _conn(db_path) as conn:
        conn.executemany(sql, rows)
    log.info("Saved %d price rows for %s.", len(rows), ticker)
    return len(rows)


def load_prices(ticker: str,
                db_path: str | Path = config.DB_PATH) -> pd.DataFrame:
    """Return stored OHLCV data for a ticker as a DataFrame."""
    sql = """
        SELECT date, open_price AS Open, high_price AS High,
               low_price AS Low, close_price AS Close, volume AS Volume
        FROM   stock_prices
        WHERE  ticker = ?
        ORDER BY date
    """
    with _conn(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(ticker,),
                               parse_dates=["date"], index_col="date")
    return df


def save_predictions(
    ticker:     str,
    model_name: str,
    dates:      pd.DatetimeIndex,
    y_true:     "np.ndarray",
    y_pred:     "np.ndarray",
    metrics:    dict,
    db_path:    str | Path = config.DB_PATH,
) -> None:
    """Store prediction results for one model / ticker combination."""
    import numpy as np

    rows = [
        (ticker, model_name, str(d.date()), float(yt), float(yp),
         metrics.get("MAE"), metrics.get("RMSE"), metrics.get("R2"))
        for d, yt, yp in zip(dates, y_true, y_pred)
    ]
    sql = """
        INSERT INTO model_predictions
            (ticker, model_name, date, actual_price, predicted_price,
             mae, rmse, r2_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _conn(db_path) as conn:
        conn.executemany(sql, rows)
    log.info("Saved %d prediction rows for %s/%s.", len(rows), ticker, model_name)


def load_predictions(
    ticker:     str,
    model_name: str,
    db_path:    str | Path = config.DB_PATH,
) -> pd.DataFrame:
    sql = """
        SELECT date, actual_price, predicted_price, mae, rmse, r2_score
        FROM   model_predictions
        WHERE  ticker = ? AND model_name = ?
        ORDER BY date
    """
    with _conn(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(ticker, model_name),
                                 parse_dates=["date"], index_col="date")


# ── Init on import ────────────────────────────────────────────────────────────
initialise()

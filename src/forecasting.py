"""
src/forecasting.py
-------------------
Multi-step forward forecasts beyond the last observed bar.

Each step: append the next business date with a synthetic OHLC bar (O=H=L=Close,
initially the prior close; updated to the model prediction after predict),
rebuild indicators, scale with the fitted preprocessor, and run one forward pass.

Sequence slicing matches create_sequences(...): for target row index i,
X = feature_array[i - look_back : i] (right end exclusive).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.lstm_data import extended_scaled_return_window
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.logger import get_logger

log = get_logger(__name__)


def _align_sentiment(
    sentiment_df: Optional[pd.DataFrame], index: pd.DatetimeIndex
) -> Optional[pd.DataFrame]:
    if sentiment_df is None:
        return None
    aligned = sentiment_df.reindex(index).ffill().bfill().fillna(0)
    return aligned


def _confidence_from_r2(r2: float) -> int:
    """Map test-set R² to a display percentage (educational UI only)."""
    x = float(r2)
    if not np.isfinite(x):
        x = 0.0
    pct = int(round(100 * max(0.0, min(1.0, x))))
    return max(50, min(99, pct)) if pct > 0 else 50


def _signal_from_forecast(last_close: float, first_pred: float) -> str:
    if first_pred > last_close * 1.0001:
        return "Buy"
    if first_pred < last_close * 0.9999:
        return "Sell"
    return "Hold"


def signal_for_forecast_window(
    *,
    fc: dict,
    page: int,
    window: int,
) -> str:
    """Buy/Sell/Hold vs reference close: last historical for page 0, else prior forecast day."""
    prices = fc["prices"]
    start = page * window
    if start >= len(prices):
        return "Hold"
    first_pred = prices[start]
    if page == 0:
        ref = float(fc["last_actual_close"])
    else:
        ref = float(prices[start - 1])
    return _signal_from_forecast(ref, first_pred)


def future_close_forecast(
    *,
    clean_df: pd.DataFrame,
    engineer: FeatureEngineer,
    preprocessor: DataPreprocessor,
    sentiment_df: Optional[pd.DataFrame],
    model: Any,
    is_lstm: bool,
    feature_cols: list[str],
    n_steps: int = config.FORECAST_MAX_HORIZON,
    look_back: int = config.LOOK_BACK,
    test_r2: float = 0.0,
    lstm_univariate: bool = False,
    lstm_target_return: bool = False,
    lstm_ret_scaler: Any | None = None,
) -> dict:
    """
    Produce n_steps of future closing prices after clean_df's last date.

    Returns
    -------
    dict with keys: dates, prices, last_actual_date, last_actual_close,
    confidence_pct, signal, model_horizon_note
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if preprocessor.scaler is None:
        raise RuntimeError("Preprocessor must be fit (fit_scale) before forecasting.")

    extended = clean_df.sort_index().copy()
    last_vol = float(extended["Volume"].iloc[-1])
    forecast_dates: list[pd.Timestamp] = []
    forecast_prices: list[float] = []

    last_close_hist = float(extended["Close"].iloc[-1])
    last_dt_hist = extended.index[-1]
    cols_order = preprocessor._feature_cols

    for _ in range(n_steps):
        last_ix = extended.index[-1]
        next_d = last_ix + pd.tseries.offsets.BDay(1)
        while next_d in extended.index:
            next_d = next_d + pd.tseries.offsets.BDay(1)

        if is_lstm and lstm_target_return:
            if lstm_ret_scaler is None:
                raise ValueError("lstm_target_return requires lstm_ret_scaler")
            close_inr = extended["Close"].values.astype(np.float64)
            if len(close_inr) < look_back + 1:
                raise ValueError(
                    f"Need >={look_back + 1} closes for return LSTM, got {len(close_inr)}"
                )
            X_in = extended_scaled_return_window(
                close_inr, lstm_ret_scaler, look_back
            )
            y_pred_sc = model.predict(X_in)
            ret_next = float(
                lstm_ret_scaler.inverse_transform(
                    np.asarray(y_pred_sc, dtype=np.float64).reshape(-1, 1)
                )[0, 0]
            )
            pred_inr = float(close_inr[-1] * (1.0 + ret_next))
            forecast_dates.append(next_d)
            forecast_prices.append(pred_inr)
            extended.loc[next_d, ["Open", "High", "Low", "Close"]] = pred_inr
            extended.loc[next_d, "Volume"] = last_vol
            extended["Volume"] = extended["Volume"].fillna(0)
            continue

        lc = float(extended["Close"].iloc[-1])
        extended.loc[next_d, ["Open", "High", "Low", "Close"]] = lc
        extended.loc[next_d, "Volume"] = last_vol
        extended["Volume"] = extended["Volume"].fillna(0)

        s_df = _align_sentiment(sentiment_df, extended.index)
        feat = engineer.build_features(extended, s_df)
        missing = set(cols_order) - set(feat.columns)
        if missing:
            raise ValueError(f"Forecast features missing columns: {missing}")
        feat = feat[cols_order].dropna()
        if len(feat) < look_back + 1:
            raise ValueError(
                f"Not enough feature rows for forecast (need > {look_back}, got {len(feat)})."
            )

        feat_scaled = preprocessor.transform(feat)
        X_2d = feat_scaled[feature_cols].values.astype(np.float32)
        close_1d = feat_scaled["Close"].values.astype(np.float32)
        i = len(feat_scaled) - 1

        if is_lstm:
            if lstm_univariate:
                w = close_1d[i - look_back : i]
                if w.shape[0] != look_back:
                    raise ValueError(
                        f"Bad LSTM univariate window {w.shape}, need ({look_back},)."
                    )
                X_in = w.reshape(1, look_back, 1)
            else:
                window = X_2d[i - look_back : i]
                if window.shape[0] != look_back:
                    raise ValueError(
                        f"Bad LSTM window shape {window.shape}, expected ({look_back},)."
                    )
                X_in = window.reshape(1, look_back, window.shape[1])
        else:
            X_in = X_2d[i : i + 1]

        y_pred_sc = model.predict(X_in)
        pred_inr = float(
            preprocessor.inverse_transform_prices(np.array([y_pred_sc.ravel()[0]]))[0]
        )

        forecast_dates.append(next_d)
        forecast_prices.append(pred_inr)

        extended.loc[next_d, ["Open", "High", "Low", "Close"]] = pred_inr

    if is_lstm and lstm_target_return:
        note_h = "return targets + price reconstruction"
    elif is_lstm and lstm_univariate:
        note_h = "univariate scaled Close"
    elif is_lstm:
        note_h = "multivariate features"
    else:
        note_h = "tabular same-row"

    conf = _confidence_from_r2(test_r2)
    if is_lstm and lstm_target_return:
        if np.isfinite(test_r2) and test_r2 >= 0.85:
            conf = max(conf, 95)
        elif np.isfinite(test_r2) and test_r2 >= 0.5:
            conf = max(conf, 90)
    sig = _signal_from_forecast(last_close_hist, forecast_prices[0])
    note = f"Recursive {n_steps}-step horizon; {note_h}."
    log.info(
        "Future forecast (%d steps) last_hist=%s first_pred=%.4f signal=%s",
        n_steps,
        last_dt_hist.date(),
        forecast_prices[0],
        sig,
    )
    return {
        "dates": forecast_dates,
        "prices": forecast_prices,
        "last_actual_date": last_dt_hist,
        "last_actual_close": last_close_hist,
        "confidence_pct": conf,
        "signal": sig,
        "model_horizon_note": note,
    }

"""
LSTM-specific data: daily-return sequences and decode to INR closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from src.feature_engineering import create_sequences


def pct_returns(close: np.ndarray) -> np.ndarray:
    """Simple daily returns; first bar is 0."""
    c = np.asarray(close, dtype=np.float64)
    out = np.zeros(len(c), dtype=np.float64)
    den = np.maximum(c[:-1], 1e-9)
    out[1:] = (c[1:] - c[:-1]) / den
    return out


def prepare_lstm_return_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ret_scaler: StandardScaler,
    look_back: int | None = None,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Fit ret_scaler on train returns; build (X, y) with y = next scaled return.
    Input shape per sample: (look_back, 1).
    """
    lb = look_back if look_back is not None else config.LOOK_BACK
    r_tr = pct_returns(train_df["Close"].values)
    ret_scaler.fit(r_tr.reshape(-1, 1))

    def _scaled(d: pd.DataFrame) -> np.ndarray:
        r = pct_returns(d["Close"].values)
        return ret_scaler.transform(r.reshape(-1, 1)).ravel().astype(np.float32)

    rs_tr = _scaled(train_df)
    rs_va = _scaled(val_df)
    rs_te = _scaled(test_df)

    X_tr, y_tr = create_sequences(rs_tr.reshape(-1, 1), rs_tr, look_back=lb)
    X_va, y_va = create_sequences(rs_va.reshape(-1, 1), rs_va, look_back=lb)
    X_te, y_te = create_sequences(rs_te.reshape(-1, 1), rs_te, look_back=lb)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def decode_return_predictions_to_close(
    ret_pred_scaled: np.ndarray,
    ret_scaler: StandardScaler,
    test_close_inr: np.ndarray,
    look_back: int | None = None,
) -> np.ndarray:
    """One-step-ahead: pred_close[t] = actual_close[t-1] * (1 + pred_return[t])."""
    lb = look_back if look_back is not None else config.LOOK_BACK
    ret_hat = ret_scaler.inverse_transform(
        np.asarray(ret_pred_scaled, dtype=np.float64).reshape(-1, 1)
    ).ravel()
    n = len(ret_hat)
    c = np.asarray(test_close_inr, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        idx = lb + k
        out[k] = c[idx - 1] * (1.0 + ret_hat[k])
    return out


def lstm_validation_ci_halfwidth_inr(
    *,
    y_val_pred_sc: np.ndarray,
    val_df: pd.DataFrame,
    preprocessor,
    lstm_ret_scaler: StandardScaler | None,
    return_mode: bool,
    look_back: int | None = None,
    z: float | None = None,
) -> float:
    """
    Symmetric half-width (INR) for LSTM prediction bands: max(z·σ, |residual| q0.975)
    on validation one-step closes.
    """
    lb = look_back if look_back is not None else config.LOOK_BACK
    zv = z if z is not None else config.LSTM_CI_Z
    if return_mode and lstm_ret_scaler is not None:
        y_hat = decode_return_predictions_to_close(
            y_val_pred_sc, lstm_ret_scaler, val_df["Close"].values, lb
        )
    else:
        y_hat = preprocessor.inverse_transform_prices(y_val_pred_sc)
    y_true = val_df["Close"].values[lb:]
    resid = y_true.astype(float) - y_hat.astype(float)
    sig = float(np.std(resid))
    if not np.isfinite(sig) or sig < 1e-9:
        sig = 1e-6
    q975 = float(np.quantile(np.abs(resid), 0.975)) if len(resid) else sig
    return float(max(zv * sig, q975))


def extended_scaled_return_window(
    close_inr: np.ndarray,
    ret_scaler: StandardScaler,
    look_back: int | None = None,
) -> np.ndarray:
    """Last (look_back,) vector of scaled returns for inference (1, look_back, 1)."""
    lb = look_back if look_back is not None else config.LOOK_BACK
    r = pct_returns(close_inr)
    rs = ret_scaler.transform(r.reshape(-1, 1)).ravel().astype(np.float32)
    w = rs[-lb:]
    if w.shape[0] != lb:
        raise ValueError(f"Need len>={lb} closes for return window, got {len(close_inr)}")
    return w.reshape(1, lb, 1)

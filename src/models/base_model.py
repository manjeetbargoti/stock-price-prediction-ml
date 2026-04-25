"""
src/models/base_model.py
-------------------------
Abstract base class shared by all predictive models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.logger import get_logger

log = get_logger(__name__)


class BaseModel(ABC):
    """
    Unified interface for all predictive models.

    Subclasses implement `train()` and `predict()`.
    `evaluate()` is inherited and works for every model.
    """

    def __init__(self, model_name: str, hyperparams: Optional[dict] = None):
        self.model_name  = model_name
        self.hyperparams = hyperparams or {}
        self.model       = None
        self.trained     = False

    # ── Abstract methods (must be overridden) ─────────────────────────────────

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:          Optional[np.ndarray] = None,
        y_val:          Optional[np.ndarray] = None,
        feature_names:  Optional[list]       = None,
    ) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values for input X."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a trained model from disk."""

    # ── Concrete methods (shared) ─────────────────────────────────────────────

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Compute MAE, RMSE, MAPE, and R² on any split."""
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAPE — guard against zero true values
        nonzero = y_true != 0
        mape = (
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            if nonzero.any() else float("nan")
        )
        r2   = r2_score(y_true, y_pred)

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
        log.info(
            "[%s] MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  R²=%.4f",
            self.model_name, mae, rmse, mape, r2,
        )
        return metrics

    def _check_trained(self) -> None:
        if not self.trained:
            raise RuntimeError(
                f"Model '{self.model_name}' is not trained. Call train() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self.trained else "untrained"
        return f"<{self.model_name} [{status}]>"

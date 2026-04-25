"""
src/models/xgb_model.py
------------------------
XGBoost Gradient Boosting model with early stopping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.models.base_model import BaseModel
from src.logger import get_logger

log = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost Regressor with validation-based early stopping."""

    def __init__(
        self,
        n_estimators:     int   = config.XGB_N_ESTIMATORS,
        learning_rate:    float = config.XGB_LR,
        max_depth:        int   = config.XGB_MAX_DEPTH,
        subsample:        float = config.XGB_SUBSAMPLE,
        colsample_bytree: float = config.XGB_COLSAMPLE,
        reg_alpha:        float = config.XGB_REG_ALPHA,
        reg_lambda:       float = config.XGB_REG_LAMBDA,
        early_stopping:   int   = config.XGB_EARLY_STOPPING,
        random_state:     int   = 42,
    ):
        super().__init__(
            "XGBoost",
            {
                "n_estimators":     n_estimators,
                "learning_rate":    learning_rate,
                "max_depth":        max_depth,
                "subsample":        subsample,
                "colsample_bytree": colsample_bytree,
            },
        )
        self._xgb_kwargs = dict(
            n_estimators     = n_estimators,
            learning_rate    = learning_rate,
            max_depth        = max_depth,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            reg_alpha        = reg_alpha,
            reg_lambda       = reg_lambda,
            objective        = "reg:squarederror",
            n_jobs           = -1,
            random_state     = random_state,
            eval_metric      = "rmse",
        )
        self._early_stopping    = early_stopping
        self.model              = xgb.XGBRegressor(**self._xgb_kwargs)
        self.feature_importance_: pd.Series | None = None
        self._feature_names: list[str] | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        X_tr = self._flatten(X_train)
        self._feature_names = feature_names

        fit_kwargs: dict = dict(verbose=False)
        if X_val is not None and y_val is not None:
            # early_stopping_rounds moved to constructor in XGBoost 2.0+
            self.model = xgb.XGBRegressor(
                **self._xgb_kwargs,
                early_stopping_rounds=self._early_stopping,
            )
            fit_kwargs["eval_set"] = [(self._flatten(X_val), y_val.ravel())]

        self.model.fit(X_tr, y_train.ravel(), **fit_kwargs)
        self.trained = True

        # Feature importance
        if feature_names:
            self.feature_importance_ = (
                pd.Series(self.model.feature_importances_, index=feature_names)
                .sort_values(ascending=False)
            )

        best = getattr(self.model, "best_iteration", "N/A")
        metrics = self.evaluate(y_train, self.predict(X_train))
        log.info(
            "XGBoost trained.  Best iter=%s  Train R²=%.4f",
            best, metrics["R2"],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(self._flatten(X))

    def save(self, path: str) -> None:
        self._check_trained()
        self.model.save_model(path)
        log.info("XGBoost model saved → %s", path)

    def load(self, path: str) -> None:
        self.model   = xgb.XGBRegressor(**self._xgb_kwargs)
        self.model.load_model(path)
        self.trained = True
        log.info("XGBoost model loaded ← %s", path)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1) if X.ndim == 3 else X

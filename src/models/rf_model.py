"""
src/models/rf_model.py
-----------------------
Random Forest Regression model with feature-importance reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.models.base_model import BaseModel
from src.logger import get_logger

log = get_logger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest Regressor.

    After training, `feature_importance_` is available as a
    pandas Series sorted by descending importance.
    """

    def __init__(
        self,
        n_estimators:     int   = config.RF_N_ESTIMATORS,
        min_samples_split: int  = config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int   = config.RF_MIN_SAMPLES_LEAF,
        max_features:     str   = config.RF_MAX_FEATURES,
        n_jobs:           int   = -1,
        random_state:     int   = 42,
    ):
        super().__init__(
            "RandomForest",
            {
                "n_estimators":      n_estimators,
                "min_samples_split": min_samples_split,
                "min_samples_leaf":  min_samples_leaf,
                "max_features":      max_features,
            },
        )
        self.model = RandomForestRegressor(
            n_estimators     = n_estimators,
            min_samples_split= min_samples_split,
            min_samples_leaf = min_samples_leaf,
            max_features     = max_features,
            n_jobs           = n_jobs,
            random_state     = random_state,
            oob_score        = True,
        )
        self.feature_importance_: pd.Series | None = None
        self._feature_names: list[str] | None = None

    # ── Interface ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        X = self._flatten(X_train)
        self._feature_names = feature_names
        self.model.fit(X, y_train.ravel())
        self.trained = True

        self._compute_importance()

        oob = self.model.oob_score_
        train_metrics = self.evaluate(y_train, self.predict(X_train))
        log.info(
            "Random Forest trained.  OOB R²=%.4f  Train R²=%.4f",
            oob, train_metrics["R2"],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(self._flatten(X))

    def save(self, path: str) -> None:
        self._check_trained()
        joblib.dump({"model": self.model, "feature_names": self._feature_names}, path)
        log.info("RandomForest saved → %s", path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self.model          = bundle["model"]
        self._feature_names = bundle.get("feature_names")
        self.trained        = True
        self._compute_importance()
        log.info("RandomForest loaded ← %s", path)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1) if X.ndim == 3 else X

    def _compute_importance(self) -> None:
        names = self._feature_names or list(range(self.model.n_features_in_))
        self.feature_importance_ = (
            pd.Series(self.model.feature_importances_, index=names)
            .sort_values(ascending=False)
        )

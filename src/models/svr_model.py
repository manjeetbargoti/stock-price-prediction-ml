"""
src/models/svr_model.py
------------------------
Support Vector Regression model.
"""

from __future__ import annotations

import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.models.base_model import BaseModel
from src.logger import get_logger

log = get_logger(__name__)


class SVRModel(BaseModel):
    """
    SVR wrapped in a sklearn Pipeline that re-scales inputs via
    StandardScaler (SVR is sensitive to feature scale).
    """

    def __init__(
        self,
        kernel:  str   = config.SVR_KERNEL,
        C:       float = config.SVR_C,
        epsilon: float = config.SVR_EPSILON,
        gamma:   str   = config.SVR_GAMMA,
    ):
        super().__init__(
            "SVR",
            {"kernel": kernel, "C": C, "epsilon": epsilon, "gamma": gamma},
        )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)),
        ])

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list | None = None,
    ) -> None:
        X = self._flatten(X_train)
        log.info("SVR fitting on %d samples…", len(X))
        self.model.fit(X, y_train.ravel())
        self.trained = True
        metrics = self.evaluate(y_train, self.predict(X_train))
        log.info("SVR trained.  Train R²=%.4f", metrics["R2"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(self._flatten(X))

    def save(self, path: str) -> None:
        self._check_trained()
        joblib.dump(self.model, path)
        log.info("SVR saved → %s", path)

    def load(self, path: str) -> None:
        self.model   = joblib.load(path)
        self.trained = True
        log.info("SVR loaded ← %s", path)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1) if X.ndim == 3 else X

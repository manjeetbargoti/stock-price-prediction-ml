"""
src/models/linear_model.py
---------------------------
Linear Regression model — used as a baseline comparator.
"""

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.base_model import BaseModel
from src.logger import get_logger

log = get_logger(__name__)


class LinearRegressionModel(BaseModel):
    """
    Ridge Regression baseline (α = 1.0 provides slight regularisation,
    making it more numerically stable than OLS for correlated features).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__("LinearRegression", {"alpha": alpha})
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None) -> None:
        X = X_train.reshape(len(X_train), -1) if X_train.ndim == 3 else X_train
        self.model.fit(X, y_train.ravel())
        self.trained = True
        train_pred = self.predict(X_train)
        metrics    = self.evaluate(y_train, train_pred)
        log.info("Linear Regression trained.  Train R²=%.4f", metrics["R2"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        X_flat = X.reshape(len(X), -1) if X.ndim == 3 else X
        return self.model.predict(X_flat)

    def save(self, path: str) -> None:
        self._check_trained()
        joblib.dump(self.model, path)
        log.info("LinearRegression saved → %s", path)

    def load(self, path: str) -> None:
        self.model   = joblib.load(path)
        self.trained = True
        log.info("LinearRegression loaded ← %s", path)

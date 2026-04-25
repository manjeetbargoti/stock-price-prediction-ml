"""
src/models/lstm_model.py
-------------------------
Stacked LSTM deep learning model with early stopping,
model checkpointing, and training-history logging.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.models.base_model import BaseModel
from src.logger import get_logger

log = get_logger(__name__)


class LSTMModel(BaseModel):
    """
    Stacked LSTM for sequential stock price regression.

    Input shape : (batch, look_back, n_features)
    Output shape: (batch, 1)  — next-day Close price (scaled)
    """

    def __init__(
        self,
        n_features:     int   = 1,
        look_back:      int   = config.LOOK_BACK,
        units_1:        int   = config.LSTM_UNITS_1,
        units_2:        int   = config.LSTM_UNITS_2,
        dense_units:    int   = config.DENSE_UNITS,
        dropout:        float = config.DROPOUT_RATE,
        learning_rate:  float = config.LSTM_LR,
    ):
        super().__init__("LSTM", {
            "look_back":     look_back,
            "units_1":       units_1,
            "units_2":       units_2,
            "dropout":       dropout,
            "learning_rate": learning_rate,
        })
        self.look_back     = look_back
        self.n_features    = n_features
        self.units_1       = units_1
        self.units_2       = units_2
        self.dense_units   = dense_units
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.history       = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self, n_features: int) -> None:
        """Lazy build so we know n_features at train time."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow is required for LSTMModel. pip install tensorflow")

        tf.random.set_seed(42)

        self.model = Sequential([
            LSTM(
                self.units_1,
                input_shape    = (self.look_back, n_features),
                return_sequences = True,
                dropout        = self.dropout,
                recurrent_dropout = self.dropout,
                name           = "lstm_1",
            ),
            BatchNormalization(),
            LSTM(
                self.units_2,
                return_sequences = False,
                dropout        = self.dropout,
                recurrent_dropout = self.dropout,
                name           = "lstm_2",
            ),
            BatchNormalization(),
            Dense(self.dense_units, activation="relu", name="dense_1"),
            Dropout(self.dropout),
            Dense(1, activation="linear", name="output"),
        ], name="StockLSTM")

        self.model.compile(
            optimizer = Adam(learning_rate=self.learning_rate),
            loss      = "mean_squared_error",
            metrics   = ["mae"],
        )
        log.info("LSTM architecture:\n%s", self.model.summary())

    # ── Interface ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:          np.ndarray | None = None,
        y_val:          np.ndarray | None = None,
        feature_names:  list | None       = None,
        checkpoint_path: str | None       = None,
    ) -> None:
        """
        Train the LSTM model.

        X_train / X_val must be 3-D: (samples, look_back, n_features).
        """
        from tensorflow.keras.callbacks import (
            EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
        )

        if X_train.ndim != 3:
            raise ValueError(
                f"X_train must be 3-D (samples, look_back, n_features), "
                f"got shape {X_train.shape}."
            )

        self.n_features = X_train.shape[2]
        self._build(self.n_features)

        callbacks = [
            EarlyStopping(
                monitor           = "val_loss" if X_val is not None else "loss",
                patience          = config.EARLY_STOP_PATIENCE,
                restore_best_weights = True,
                verbose           = 1,
            ),
            ReduceLROnPlateau(
                monitor  = "val_loss" if X_val is not None else "loss",
                factor   = 0.5,
                patience = 7,
                min_lr   = 1e-6,
                verbose  = 1,
            ),
        ]

        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath   = checkpoint_path,
                    monitor    = "val_loss" if X_val is not None else "loss",
                    save_best_only = True,
                    verbose    = 1,
                )
            )

        fit_kwargs: dict = dict(
            x          = X_train,
            y          = y_train.ravel(),
            epochs     = config.LSTM_EPOCHS,
            batch_size = config.LSTM_BATCH_SIZE,
            callbacks  = callbacks,
            verbose    = 1,
            shuffle    = False,     # preserve temporal order
        )
        if X_val is not None and y_val is not None:
            fit_kwargs["validation_data"] = (X_val, y_val.ravel())

        self.history = self.model.fit(**fit_kwargs)
        self.trained = True

        best_epoch = np.argmin(self.history.history.get("val_loss",
                                                         self.history.history["loss"]))
        log.info(
            "LSTM trained. Best epoch=%d  Final val_loss=%.6f",
            best_epoch + 1,
            self.history.history.get("val_loss", self.history.history["loss"])[-1],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D, got {X.shape}.")
        return self.model.predict(X, verbose=0).ravel()

    def save(self, path: str) -> None:
        self._check_trained()
        self.model.save(path)
        log.info("LSTM saved → %s", path)

    def load(self, path: str) -> None:
        from tensorflow.keras.models import load_model
        self.model   = load_model(path)
        self.trained = True
        log.info("LSTM loaded ← %s", path)

    def get_training_history(self) -> dict | None:
        """Return loss/val_loss history dict for plotting."""
        return self.history.history if self.history else None

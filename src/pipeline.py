"""
src/pipeline.py
----------------
End-to-end orchestration:
  fetch → preprocess → feature-engineer → train → evaluate → persist

Can be used programmatically (by the dashboard) or run as a script.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.data_collection    import StockDataFetcher
from src.preprocessing      import DataPreprocessor
from src.feature_engineering import FeatureEngineer, create_sequences
from src.sentiment          import SentimentAnalyzer
from src.evaluation         import ModelEvaluator
from src import database
from src.logger             import get_logger

log = get_logger(__name__)

# Model registry — import lazily to avoid TF loading overhead
MODEL_REGISTRY = {
    "Linear Regression": "src.models.linear_model.LinearRegressionModel",
    "Random Forest":     "src.models.rf_model.RandomForestModel",
    "XGBoost":           "src.models.xgb_model.XGBoostModel",
    "SVR":               "src.models.svr_model.SVRModel",
    "LSTM":              "src.models.lstm_model.LSTMModel",
}


def _import_model(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class StockPredictionPipeline:
    """
    Full pipeline for one ticker and a selected subset of models.

    Parameters
    ----------
    ticker           : NSE ticker symbol, e.g. "RELIANCE"
    start_date       : data start (YYYY-MM-DD)
    end_date         : data end   (YYYY-MM-DD)
    model_names      : list from MODEL_REGISTRY keys; None → all 5
    use_sentiment    : whether to merge sentiment features
    sentiment_df     : optional pre-computed sentiment DataFrame
    save_models      : persist trained models to MODELS_DIR
    """

    def __init__(
        self,
        ticker:        str,
        start_date:    str  = config.DEFAULT_START_DATE,
        end_date:      str  = config.DEFAULT_END_DATE,
        model_names:   Optional[list[str]] = None,
        use_sentiment: bool = False,
        sentiment_df:  Optional[pd.DataFrame] = None,
        save_models:   bool = True,
    ):
        self.ticker        = ticker
        self.start_date    = start_date
        self.end_date      = end_date
        self.model_names   = model_names or list(MODEL_REGISTRY.keys())
        self.use_sentiment = use_sentiment
        self.sentiment_df  = sentiment_df
        self.save_models   = save_models

        self.preprocessor = DataPreprocessor()
        self.engineer     = FeatureEngineer()
        self.evaluator    = ModelEvaluator()

        # Populated during run()
        self.feature_df: pd.DataFrame | None = None
        self.trained_models: dict = {}
        self.lstm_ret_scaler = None

    # ── Main Entry ────────────────────────────────────────────────────────────

    def run(self) -> ModelEvaluator:
        """Execute full pipeline.  Returns the populated ModelEvaluator."""
        log.info("=== Pipeline START  ticker=%s  models=%s ===",
                 self.ticker, self.model_names)

        # 1 — Fetch
        raw_df = StockDataFetcher(self.ticker, self.start_date, self.end_date).fetch()
        database.save_prices(raw_df, self.ticker)

        # 2 — Clean
        clean_df = self.preprocessor.clean(raw_df)

        # 3 — Features
        s_df = None
        if self.use_sentiment:
            s_df = self.sentiment_df or SentimentAnalyzer.make_zero_scores(clean_df.index)
        self.feature_df = self.engineer.build_features(clean_df, s_df)

        # 4 — Split (on feature_df)
        train_df, val_df, test_df = self.preprocessor.split(self.feature_df)

        # 5 — Scale
        feature_cols = [c for c in self.feature_df.columns if c != "Close"]
        all_cols     = feature_cols + ["Close"]
        train_sc = self.preprocessor.fit_scale(train_df, all_cols)
        val_sc   = self.preprocessor.transform(val_df)
        test_sc  = self.preprocessor.transform(test_df)

        # 6 — Prepare arrays
        X_tr_2d  = train_sc[feature_cols].values
        y_tr     = train_sc["Close"].values
        X_val_2d = val_sc[feature_cols].values
        y_val    = val_sc["Close"].values
        X_te_2d  = test_sc[feature_cols].values
        y_te     = test_sc["Close"].values

        self.lstm_ret_scaler = None
        if "LSTM" in self.model_names and config.LSTM_TARGET_MODE == "return":
            from sklearn.preprocessing import StandardScaler
            from src.lstm_data import prepare_lstm_return_sequences
            self.lstm_ret_scaler = StandardScaler()
            (X_tr_3d, y_tr_3d), (X_val_3d, y_val_3d), (X_te_3d, y_te_3d) = (
                prepare_lstm_return_sequences(
                    train_df, val_df, test_df, self.lstm_ret_scaler, config.LOOK_BACK
                )
            )
            log.info(
                "LSTM: return target (StandardScaler on daily %% returns), input (look_back, 1)"
            )
        else:
            X_tr_3d,  y_tr_3d  = create_sequences(X_tr_2d,  y_tr)
            X_val_3d, y_val_3d = create_sequences(X_val_2d, y_val)
            X_te_3d,  y_te_3d  = create_sequences(X_te_2d,  y_te)
            if "LSTM" in self.model_names and config.LSTM_UNIVARIATE_INPUT:
                X_tr_3d,  y_tr_3d  = create_sequences(y_tr.reshape(-1, 1),  y_tr)
                X_val_3d, y_val_3d = create_sequences(y_val.reshape(-1, 1), y_val)
                X_te_3d,  y_te_3d  = create_sequences(y_te.reshape(-1, 1), y_te)
                log.info("LSTM: univariate scaled Close — input shape (look_back, 1)")

        # Ground-truth closes on test (same alignment for all models)
        _, y_te_close = create_sequences(y_te.reshape(-1, 1), y_te)
        test_actual = self.preprocessor.inverse_transform_prices(y_te_close)
        test_dates  = test_df.index[config.LOOK_BACK:]

        # 7 — Train each model
        for mname in self.model_names:
            log.info("--- Training: %s ---", mname)
            ModelClass = _import_model(MODEL_REGISTRY[mname])
            model      = ModelClass()

            is_lstm = mname == "LSTM"
            X_train_in  = X_tr_3d  if is_lstm else X_tr_2d[config.LOOK_BACK:]
            y_train_in  = y_tr_3d  if is_lstm else y_tr[config.LOOK_BACK:]
            X_val_in    = X_val_3d if is_lstm else X_val_2d[config.LOOK_BACK:]
            y_val_in    = y_val_3d if is_lstm else y_val[config.LOOK_BACK:]
            X_test_in   = X_te_3d  if is_lstm else X_te_2d[config.LOOK_BACK:]

            ckpt = str(config.MODELS_DIR / f"{self.ticker}_{mname.replace(' ','_')}_best.h5") \
                   if is_lstm else None
            train_kwargs: dict = {"X_val": X_val_in, "y_val": y_val_in}
            if is_lstm and ckpt:
                train_kwargs["checkpoint_path"] = ckpt
            if not is_lstm:
                train_kwargs["feature_names"] = feature_cols

            model.train(X_train_in, y_train_in, **train_kwargs)
            self.trained_models[mname] = model

            y_pred_sc = model.predict(X_test_in)
            if (
                is_lstm
                and config.LSTM_TARGET_MODE == "return"
                and self.lstm_ret_scaler is not None
            ):
                from src.lstm_data import decode_return_predictions_to_close

                y_pred = decode_return_predictions_to_close(
                    y_pred_sc,
                    self.lstm_ret_scaler,
                    test_df["Close"].values,
                    config.LOOK_BACK,
                )
            else:
                y_pred = self.preprocessor.inverse_transform_prices(y_pred_sc)

            metrics = self.evaluator.add_result(mname, test_actual, y_pred, test_dates)

            # Persist
            if self.save_models:
                save_path = config.MODELS_DIR / f"{self.ticker}_{mname.replace(' ','_')}.pkl"
                if not is_lstm:          # LSTM uses Keras save
                    model.save(str(save_path))

            database.save_predictions(
                self.ticker, mname, test_dates, test_actual, y_pred, metrics
            )

        log.info("=== Pipeline DONE ===")
        return self.evaluator

    def get_comparison_table(self) -> pd.DataFrame:
        return self.evaluator.comparison_table()

    def get_backtest(self, model_name: str) -> dict:
        return self.evaluator.simulate_trading(model_name)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stock Price Prediction Pipeline")
    parser.add_argument("--ticker",  default="RELIANCE")
    parser.add_argument("--start",   default=config.DEFAULT_START_DATE)
    parser.add_argument("--end",     default=config.DEFAULT_END_DATE)
    parser.add_argument("--models",  nargs="+",
                        default=["Random Forest", "XGBoost", "LSTM"])
    parser.add_argument("--sentiment", action="store_true")
    args = parser.parse_args()

    pipe = StockPredictionPipeline(
        ticker        = args.ticker,
        start_date    = args.start,
        end_date      = args.end,
        model_names   = args.models,
        use_sentiment = args.sentiment,
    )
    evaluator = pipe.run()
    print("\n" + "="*60)
    print(evaluator.comparison_table().to_string(index=False))
    print("="*60)

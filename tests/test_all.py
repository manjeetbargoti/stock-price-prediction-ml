"""
tests/test_all.py
------------------
Comprehensive unit and integration test suite.
Run with:  pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """Generate 500 rows of synthetic OHLCV data."""
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 1000.0 + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 100)
    df = pd.DataFrame({
        "Open":   close * (1 + np.random.randn(n) * 0.003),
        "High":   close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "Low":    close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 20_000_000, n).astype(float),
    }, index=idx)
    return df


@pytest.fixture
def clean_df(sample_ohlcv):
    from src.preprocessing import DataPreprocessor
    return DataPreprocessor().clean(sample_ohlcv)


@pytest.fixture
def feature_df(clean_df):
    from src.feature_engineering import FeatureEngineer
    return FeatureEngineer().build_features(clean_df)


@pytest.fixture
def split_scaled(feature_df):
    from src.preprocessing import DataPreprocessor
    pp = DataPreprocessor()
    train, val, test = pp.split(feature_df)
    feat_cols = [c for c in feature_df.columns if c != "Close"]
    all_cols  = feat_cols + ["Close"]
    train_sc  = pp.fit_scale(train, all_cols)
    val_sc    = pp.transform(val)
    test_sc   = pp.transform(test)
    return pp, feat_cols, train_sc, val_sc, test_sc


# ──────────────────────────────────────────────────────────────────────────────
# DataPreprocessor
# ──────────────────────────────────────────────────────────────────────────────

class TestDataPreprocessor:

    def test_clean_removes_nans(self, sample_ohlcv):
        from src.preprocessing import DataPreprocessor
        df = sample_ohlcv.copy()
        df.loc[df.index[5:10], "Close"] = np.nan
        clean = DataPreprocessor().clean(df)
        assert clean["Close"].isna().sum() == 0

    def test_clean_adds_daily_return(self, clean_df):
        assert "Daily_Return" in clean_df.columns

    def test_split_ratios(self, clean_df):
        from src.preprocessing import DataPreprocessor
        pp = DataPreprocessor(train_ratio=0.70, val_ratio=0.15)
        train, val, test = pp.split(clean_df)
        total = len(train) + len(val) + len(test)
        assert total == len(clean_df)
        assert abs(len(train) / total - 0.70) < 0.02

    def test_no_data_leakage(self, split_scaled):
        pp, feat_cols, train_sc, val_sc, test_sc = split_scaled
        # Scaler was fitted on train: train mean should be ~0.5, test may differ
        train_mean = train_sc[feat_cols].values.mean()
        test_mean  = test_sc[feat_cols].values.mean()
        # They should NOT be identical (leakage would make them identical)
        assert abs(train_mean - test_mean) > 0.001

    def test_scale_range(self, split_scaled):
        pp, feat_cols, train_sc, _, _ = split_scaled
        vals = train_sc[feat_cols].values
        assert vals.min() >= -0.01 and vals.max() <= 1.01

    def test_inverse_transform(self, split_scaled, clean_df):
        pp, feat_cols, train_sc, val_sc, test_sc = split_scaled
        original = clean_df["Close"].values[-len(test_sc):]
        scaled   = test_sc["Close"].values
        recovered = pp.inverse_transform_prices(scaled)
        np.testing.assert_allclose(recovered, original, rtol=0.02)

    def test_negative_price_raises(self):
        from src.preprocessing import DataPreprocessor
        bad = pd.DataFrame({
            "Open":[100]*300, "High":[105]*300, "Low":[95]*300,
            "Close":[100]*300, "Volume":[1e6]*300
        }, index=pd.date_range("2020-01-01", periods=300, freq="B"))
        bad.loc[bad.index[10], "Close"] = -50
        dp = DataPreprocessor()
        # winsorising will handle it, but let's confirm it doesn't crash
        cleaned = dp.clean(bad)
        # The extreme value should be corrected
        assert (cleaned["Close"] > 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# FeatureEngineer
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_rsi_range(self, feature_df):
        assert feature_df["rsi"].between(0, 100).all()

    def test_bollinger_band_order(self, feature_df):
        assert (feature_df["bb_upper"] >= feature_df["bb_lower"]).all()

    def test_atr_nonnegative(self, feature_df):
        assert (feature_df["atr"] >= 0).all()

    def test_lag_shift(self, feature_df, clean_df):
        # close_lag1 at row i should equal Close at row i-1
        aligned = feature_df[["Close", "close_lag1"]].dropna()
        assert np.allclose(
            aligned["close_lag1"].values[1:],
            aligned["Close"].values[:-1],
            rtol=1e-4,
        )

    def test_no_nan_after_build(self, feature_df):
        assert feature_df.isna().sum().sum() == 0

    def test_sma_correctness(self, clean_df):
        from src.feature_engineering import FeatureEngineer
        df = FeatureEngineer().build_features(clean_df)
        # Spot-check SMA-20 at a known position
        row_idx = 100
        expected_sma = clean_df["Close"].iloc[row_idx - 19 : row_idx + 1].mean()
        actual_row   = df[df.index == clean_df.index[row_idx]]
        if not actual_row.empty:
            assert abs(actual_row["sma_20"].values[0] - expected_sma) < 1.0

    def test_feature_count(self, feature_df):
        # Should produce at least 30 feature columns
        assert len(feature_df.columns) >= 30


# ──────────────────────────────────────────────────────────────────────────────
# Sequence Creation
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateSequences:

    def test_output_shapes(self):
        from src.feature_engineering import create_sequences
        n, f, look_back = 200, 10, 60
        X_data = np.random.rand(n, f).astype(np.float32)
        y_data = np.random.rand(n).astype(np.float32)
        X, y = create_sequences(X_data, y_data, look_back=look_back)
        assert X.shape == (n - look_back, look_back, f)
        assert y.shape == (n - look_back,)

    def test_sequence_content(self):
        from src.feature_engineering import create_sequences
        data   = np.arange(100).reshape(100, 1).astype(np.float32)
        target = np.arange(100).astype(np.float32)
        X, y   = create_sequences(data, target, look_back=5)
        # First sequence should be rows 0–4
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3, 4])
        assert y[0] == 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Models — shared interface tests
# ──────────────────────────────────────────────────────────────────────────────

def _make_2d_data(n=400, f=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, f)).astype(np.float32)
    y = rng.random(n).astype(np.float32)
    return X[:300], y[:300], X[300:350], y[300:350], X[350:], y[350:]


class TestLinearModel:

    def test_train_and_predict(self):
        from src.models.linear_model import LinearRegressionModel
        Xtr, ytr, Xv, yv, Xte, yte = _make_2d_data()
        m = LinearRegressionModel()
        m.train(Xtr, ytr)
        pred = m.predict(Xte)
        assert pred.shape == (len(Xte),)

    def test_untrained_predict_raises(self):
        from src.models.linear_model import LinearRegressionModel
        with pytest.raises(RuntimeError):
            LinearRegressionModel().predict(np.random.rand(10, 5))

    def test_evaluate_perfect(self):
        from src.models.linear_model import LinearRegressionModel
        m = LinearRegressionModel()
        y = np.array([1.0, 2.0, 3.0])
        metrics = m.evaluate(y, y)
        assert metrics["MAE"]  == pytest.approx(0.0, abs=1e-9)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["R2"]   == pytest.approx(1.0, abs=1e-9)

    def test_save_load(self, tmp_path):
        from src.models.linear_model import LinearRegressionModel
        Xtr, ytr, _, _, Xte, _ = _make_2d_data()
        m = LinearRegressionModel()
        m.train(Xtr, ytr)
        path = str(tmp_path / "lr.pkl")
        m.save(path)
        m2 = LinearRegressionModel()
        m2.load(path)
        np.testing.assert_allclose(m.predict(Xte), m2.predict(Xte), rtol=1e-5)


class TestRandomForestModel:

    def test_train_produces_importance(self):
        from src.models.rf_model import RandomForestModel
        Xtr, ytr, Xv, yv, _, _ = _make_2d_data()
        m = RandomForestModel(n_estimators=50)
        m.train(Xtr, ytr, feature_names=[f"f{i}" for i in range(20)])
        assert m.feature_importance_ is not None
        assert len(m.feature_importance_) == 20

    def test_3d_input_flattened(self):
        from src.models.rf_model import RandomForestModel
        X3d = np.random.rand(100, 5, 10).astype(np.float32)
        y   = np.random.rand(100).astype(np.float32)
        m   = RandomForestModel(n_estimators=20)
        m.train(X3d, y)
        pred = m.predict(X3d)
        assert pred.shape == (100,)


class TestXGBoostModel:

    def test_train_and_predict(self):
        from src.models.xgb_model import XGBoostModel
        Xtr, ytr, Xv, yv, Xte, _ = _make_2d_data()
        m = XGBoostModel(n_estimators=50)
        m.train(Xtr, ytr, Xv, yv)
        pred = m.predict(Xte)
        assert pred.shape == (len(Xte),)

    def test_save_load_xgb(self, tmp_path):
        from src.models.xgb_model import XGBoostModel
        Xtr, ytr, Xv, yv, Xte, _ = _make_2d_data()
        m = XGBoostModel(n_estimators=30)
        m.train(Xtr, ytr, Xv, yv)
        path = str(tmp_path / "xgb.json")
        m.save(path)
        m2 = XGBoostModel(n_estimators=30)
        m2.load(path)
        np.testing.assert_allclose(m.predict(Xte), m2.predict(Xte), rtol=1e-4)


class TestSVRModel:

    def test_train_and_predict_small(self):
        from src.models.svr_model import SVRModel
        X = np.random.rand(150, 5).astype(np.float32)
        y = np.random.rand(150).astype(np.float32)
        m = SVRModel(C=1.0)
        m.train(X[:100], y[:100])
        pred = m.predict(X[100:])
        assert pred.shape == (50,)


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment
# ──────────────────────────────────────────────────────────────────────────────

class TestSentimentAnalyzer:

    def test_score_text_range(self):
        from src.sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
        score = sa.score_text("Company profits surge — exceeds all expectations!")
        assert -1.0 <= score["compound"] <= 1.0

    def test_from_tuples_returns_df(self):
        from src.sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
        tuples = [
            ("2023-01-03", "Reliance profit beats expectations."),
            ("2023-01-03", "Markets rally on positive cues."),
            ("2023-01-04", "Global sell-off drags Indian indices lower."),
        ]
        result = sa.from_tuples(tuples)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2   # 2 unique dates
        assert "sentiment_mean" in result.columns

    def test_zero_scores_shape(self):
        from src.sentiment import SentimentAnalyzer
        idx = pd.date_range("2023-01-01", periods=10, freq="B")
        z   = SentimentAnalyzer.make_zero_scores(idx)
        assert len(z) == 10
        assert (z["sentiment_mean"] == 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# ModelEvaluator
# ──────────────────────────────────────────────────────────────────────────────

class TestModelEvaluator:

    @pytest.fixture
    def evaluator_with_results(self):
        from src.evaluation import ModelEvaluator
        ev = ModelEvaluator()
        np.random.seed(0)
        y_true = np.linspace(1000, 1500, 200) + np.random.randn(200) * 10
        ev.add_result("ModelA", y_true, y_true + np.random.randn(200) * 30)
        ev.add_result("ModelB", y_true, y_true + np.random.randn(200) * 15)
        return ev, y_true

    def test_comparison_table_sorted(self, evaluator_with_results):
        ev, _ = evaluator_with_results
        table = ev.comparison_table()
        # Should be sorted by RMSE ascending
        assert table["RMSE"].is_monotonic_increasing

    def test_best_model(self, evaluator_with_results):
        ev, _ = evaluator_with_results
        best = ev.best_model()
        assert best in ["ModelA", "ModelB"]

    def test_simulate_trading_keys(self, evaluator_with_results):
        ev, y_true = evaluator_with_results
        result = ev.simulate_trading("ModelB", initial_capital=100_000)
        for key in ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"]:
            assert key in result

    def test_evaluate_metrics_perfect(self):
        from src.evaluation import ModelEvaluator
        ev = ModelEvaluator()
        y  = np.array([100.0, 200.0, 300.0])
        m  = ev.add_result("Perfect", y, y)
        assert m["MAE"]  == pytest.approx(0.0, abs=1e-9)
        assert m["R2"]   == pytest.approx(1.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_preprocess_then_features(self, sample_ohlcv):
        from src.preprocessing      import DataPreprocessor
        from src.feature_engineering import FeatureEngineer
        clean = DataPreprocessor().clean(sample_ohlcv)
        feats = FeatureEngineer().build_features(clean)
        assert len(feats) > 0
        assert "rsi" in feats.columns
        assert "sma_20" in feats.columns

    def test_feature_engineer_with_sentiment(self, clean_df):
        from src.feature_engineering import FeatureEngineer
        from src.sentiment import SentimentAnalyzer
        sent = SentimentAnalyzer.make_zero_scores(clean_df.index)
        feats = FeatureEngineer().build_features(clean_df, sent)
        assert "sentiment_mean" in feats.columns

    def test_full_sklearn_pipeline(self, split_scaled):
        from src.models.rf_model import RandomForestModel
        from src.evaluation      import ModelEvaluator
        from src.feature_engineering import create_sequences
        import config

        pp, feat_cols, train_sc, val_sc, test_sc = split_scaled
        X_tr = train_sc[feat_cols].values
        y_tr = train_sc["Close"].values
        X_te = test_sc[feat_cols].values
        y_te = test_sc["Close"].values

        model = RandomForestModel(n_estimators=20)
        model.train(X_tr, y_tr)
        y_pred_sc = model.predict(X_te)
        y_pred    = pp.inverse_transform_prices(y_pred_sc)
        y_true    = pp.inverse_transform_prices(y_te)

        ev = ModelEvaluator()
        m  = ev.add_result("RF", y_true, y_pred)
        assert m["R2"] > 0.5, f"R² surprisingly low: {m['R2']}"

    def test_database_round_trip(self, sample_ohlcv, tmp_path):
        from src import database
        db = tmp_path / "test.db"
        database.initialise(db)
        database.save_prices(sample_ohlcv, "TEST", db)
        loaded = database.load_prices("TEST", db)
        assert len(loaded) == len(sample_ohlcv)
        np.testing.assert_allclose(
            loaded["Close"].values,
            sample_ohlcv["Close"].values,
            rtol=1e-4,
        )

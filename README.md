# Algorithmic Trading — Stock Price Prediction with Machine Learning

**MSc (Data Science) Capstone Project · Chandigarh University**  
Developed with Linear Regression, Random Forest, XGBoost, SVR & LSTM

---

## Overview

An end-to-end machine learning pipeline for predicting NIFTY 50 stock closing prices using five algorithms spanning linear, ensemble, kernel, and deep learning families. The project includes an interactive Streamlit dashboard, a CLI pipeline runner, a SQLite persistence layer, VADER-based sentiment analysis, and a simulated trading backtest.

---

## Project Structure

```
stock_predictor/
├── config.py                    # All tunable hyperparameters and paths
├── requirements.txt             # Python dependencies
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py       # Yahoo Finance fetcher with Parquet caching
│   ├── preprocessing.py         # Cleaning, Winsorisation, MinMax scaling, splits
│   ├── feature_engineering.py   # 40+ technical indicators, lag & rolling features
│   ├── sentiment.py             # VADER news sentiment scorer
│   ├── evaluation.py            # Metrics, Plotly charts, trading backtest
│   ├── pipeline.py              # End-to-end orchestrator (CLI + Python API)
│   ├── database.py              # SQLite persistence (prices & predictions)
│   ├── logger.py                # Structured logging (file + console)
│   └── models/
│       ├── __init__.py
│       ├── base_model.py        # Abstract base class (shared interface)
│       ├── linear_model.py      # Ridge Regression baseline
│       ├── rf_model.py          # Random Forest with OOB score & feature importance
│       ├── xgb_model.py         # XGBoost with early stopping
│       ├── svr_model.py         # Support Vector Regression (RBF kernel)
│       └── lstm_model.py        # Stacked LSTM (Keras) with callbacks
│
├── dashboard/
│   └── app.py                   # Streamlit interactive web dashboard
│
├── data/
│   ├── raw/                     # Cached Parquet files (auto-created)
│   ├── processed/               # Processed data (auto-created)
│   └── stock_predictor.db       # SQLite database (auto-created)
│
├── saved_models/                # Trained model artefacts (auto-created)
├── logs/
│   └── stock_predictor.log      # Application log (auto-created)
└── tests/
    └── test_all.py              # pytest unit + integration test suite
```

---

## Python Version Requirement

> **Python 3.10 is required.**

TensorFlow (used by the LSTM model) does not yet support Python 3.11+. Use the `py` launcher to target 3.10 explicitly:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
```

---

## Setup

### 1. Create virtual environment (Python 3.10)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Download VADER lexicon (one-time)

```powershell
python -c "import nltk; nltk.download('vader_lexicon')"
```

The SQLite database and all required directories (`data/`, `saved_models/`, `logs/`) are created automatically on first run.

---

## Usage

### Option A — Interactive Dashboard

```powershell
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Dashboard features:**
- Select any NIFTY 50 ticker and custom date range
- Choose one or more models to train simultaneously
- Toggle VADER sentiment analysis
- View tabs: Data Overview · Technical Indicators · Model Predictions · Performance Metrics · Trading Signals

### Option B — Command-Line Pipeline

```powershell
python -m src.pipeline `
    --ticker   RELIANCE `
    --start    2021-01-01 `
    --end      2026-04-25 `
    --models   "Random Forest" XGBoost LSTM `
    --sentiment
```

Prints a formatted comparison table of MAE, RMSE, MAPE, and R² for all selected models.

### Option C — Python API

```python
from src.pipeline import StockPredictionPipeline

pipe = StockPredictionPipeline(
    ticker        = "TCS",
    start_date    = "2021-01-01",
    end_date      = "2026-04-25",
    model_names   = ["Random Forest", "XGBoost", "LSTM"],
    use_sentiment = True,
)
evaluator = pipe.run()

# Comparison table sorted by RMSE
print(evaluator.comparison_table())

# Trading backtest on the best model
best   = evaluator.best_model()
result = evaluator.simulate_trading(best, initial_capital=100_000)
print(f"Total Return : {result['Total Return']}%")
print(f"Sharpe Ratio : {result['Sharpe Ratio']}")
print(f"Max Drawdown : {result['Max Drawdown']}%")
print(f"Win Rate     : {result['Win Rate']}%")
```

---

## Models Implemented

| Model | Class | Algorithm | Key Features |
|---|---|---|---|
| Linear Regression | `LinearRegressionModel` | Ridge Regression (α=1.0) | Fast baseline, interpretable coefficients |
| Random Forest | `RandomForestModel` | Ensemble of 500 trees | OOB R², feature importance, robust to noise |
| XGBoost | `XGBoostModel` | Gradient Boosting (1000 estimators) | Early stopping, feature importance, L1/L2 regularisation |
| SVR | `SVRModel` | Support Vector Regression (RBF) | Effective on smaller datasets, kernel trick |
| LSTM | `LSTMModel` | Stacked LSTM (Keras/TensorFlow) | Captures long-range temporal dependencies |

All models share a common `BaseModel` interface: `train()`, `predict()`, `evaluate()`, `save()`, `load()`.

### LSTM Architecture

```
Input  → (batch, look_back=60, n_features)
LSTM-1 → 128 units, return_sequences=True, dropout=0.2
BatchNorm
LSTM-2 → 64 units, return_sequences=False, dropout=0.2
BatchNorm
Dense  → 32 units, ReLU
Dropout → 0.2
Output → 1 unit, linear
```

Compiled with Adam (lr=0.001), MSE loss. Callbacks: EarlyStopping (patience=15), ReduceLROnPlateau, ModelCheckpoint.

---

## Feature Engineering (40+ Features)

| Category | Features |
|---|---|
| **Raw OHLCV** | Open, High, Low, Volume, Daily Return |
| **Trend** | SMA-10/20/50, EMA-12/26, MACD, MACD Signal, MACD Histogram |
| **Momentum** | RSI-14, Stochastic %K (14-day), Stochastic %D (3-day) |
| **Volatility** | Bollinger Upper/Lower/Width/%, ATR-14 |
| **Volume** | OBV, VWAP (20-day rolling), Volume Ratio (vs 20-day avg) |
| **Lag Features** | Close Lag 1/2/3/5/10, Return Lag 1/2/3/5/10 |
| **Rolling Stats** | Rolling Mean 5/10/20 days, Rolling Std 5/10/20 days |
| **Calendar** | Day of Week, Month, Quarter |
| **Sentiment** *(optional)* | Sentiment Mean, Sentiment Std, Positive Ratio, Negative Ratio |

All parameters (SMA windows, RSI period, ATR period, Stochastic period, etc.) are defined in `config.py` — no magic numbers in source code.

---

## Data Pipeline

```
Yahoo Finance (yfinance)
        │
        ▼
Parquet Cache (data/raw/)    ← skipped on cache hit
        │
        ▼
DataPreprocessor.clean()
  • Forward/backward fill OHLC gaps
  • Zero-fill Volume on non-trading days
  • Remove duplicate dates
  • Winsorise outliers (rolling 30-day, ±4σ → rolling median)
  • Compute Daily Return
        │
        ▼
FeatureEngineer.build_features()
  • Add all 40+ technical features
  • Optional: merge VADER sentiment with exponential decay
  • Drop NaN rows
        │
        ▼
Chronological 70/15/15 split  ← NO shuffling, NO leakage
        │
        ▼
MinMaxScaler fitted on TRAIN only → applied to Val & Test
        │
        ▼
Model Training (per selected model)
        │
        ▼
Inverse-transform predictions → actual INR prices
        │
        ▼
ModelEvaluator  (MAE / RMSE / MAPE / R²)
        │
        ▼
SQLite (prices + predictions)  &  saved_models/
```

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | mean(|y − ŷ|) | Average absolute error in INR |
| RMSE | √mean((y − ŷ)²) | Penalises large errors more |
| MAPE | mean(|y − ŷ| / y) × 100 | Scale-free percentage error |
| R² | 1 − SS_res/SS_tot | Proportion of variance explained |

### Trading Backtest (`simulate_trading`)

A simple long-only directional strategy:
- **Buy** when the model predicts the next day's price will rise
- **Sell** (exit) when the model predicts a fall
- 0.1% transaction cost per trade

Reported statistics: Total Return, Buy & Hold Return, Sharpe Ratio (annualised, √252), Maximum Drawdown, Win Rate.

---

## Sentiment Analysis

`SentimentAnalyzer` uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner), which is pre-calibrated for financial short text.

**Three input modes:**

```python
analyzer = SentimentAnalyzer()

# From a DataFrame with 'date' and 'headline' columns
daily_df = analyzer.from_dataframe(news_df)

# From a list of (date_str, headline) tuples
daily_df = analyzer.from_tuples(headline_tuples)

# Neutral placeholder (when no news data is available)
daily_df = SentimentAnalyzer.make_zero_scores(date_index)
```

**Output columns per day:** `sentiment_mean`, `sentiment_std`, `news_count`, `positive_ratio`, `negative_ratio`

Gaps (weekends, holidays) are forward-filled with exponential decay (`SENTIMENT_DECAY = 0.9` per day) so stale sentiment loses influence gradually.

---

## Database Layer

SQLite database at `data/stock_predictor.db` with WAL journal mode.

| Table | Stores |
|---|---|
| `stock_prices` | OHLCV rows per ticker/date, upserted (no duplicates) |
| `model_predictions` | Predicted and actual prices, MAE, RMSE, R² per model run |

```python
from src import database

database.save_prices(df, "RELIANCE")
df = database.load_prices("RELIANCE")

database.save_predictions("RELIANCE", "XGBoost", dates, y_true, y_pred, metrics)
df = database.load_predictions("RELIANCE", "XGBoost")
```

---

## Configuration (`config.py`)

All tunable parameters are centralised. Key settings:

```python
# Data
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE   = "2026-04-25"
MIN_REQUIRED_ROWS  = 300

# Splits
TRAIN_RATIO = 0.70   # 70% training
VAL_RATIO   = 0.15   # 15% validation
             # 0.15  # 15% test (implicit)

# LSTM
LOOK_BACK           = 60     # timesteps per input sequence
LSTM_EPOCHS         = 100
EARLY_STOP_PATIENCE = 15
LSTM_UNITS_1        = 128
LSTM_UNITS_2        = 64
DROPOUT_RATE        = 0.2

# XGBoost
XGB_N_ESTIMATORS   = 1000
XGB_EARLY_STOPPING = 50

# Random Forest
RF_N_ESTIMATORS = 500
```

---

## Running Tests

```powershell
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Test coverage includes:**

| Test Class | What Is Tested |
|---|---|
| `TestDataPreprocessor` | NaN removal, split ratios, data leakage, scale range, inverse-transform, outlier handling |
| `TestFeatureEngineer` | RSI bounds, Bollinger Band order, ATR non-negativity, lag correctness, zero NaN after build, SMA correctness, feature count |
| `TestCreateSequences` | Output shapes, sequence content correctness |
| `TestLinearModel` | Train/predict, untrained guard, perfect-prediction metrics, save/load round-trip |
| `TestRandomForestModel` | Feature importance, 3-D input flattening |
| `TestXGBoostModel` | Train/predict, save/load round-trip |
| `TestSVRModel` | Train/predict on small data |
| `TestSentimentAnalyzer` | Score range, DataFrame output, zero-score placeholder |
| `TestModelEvaluator` | RMSE-sorted table, best model, backtest keys, perfect-score metrics |
| `TestIntegration` | Full preprocess→features chain, sentiment merge, sklearn pipeline, SQLite round-trip |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation and numerical computing |
| `scikit-learn` | LinearRegression, SVR, MinMaxScaler, metrics |
| `xgboost` | Gradient boosting (≥3.x) |
| `tensorflow` / `keras` | LSTM deep learning model |
| `yfinance` | Yahoo Finance data download |
| `ta` | Additional technical indicators (optional) |
| `nltk`, `textblob` | VADER sentiment analysis |
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive charts |
| `matplotlib`, `seaborn` | Static plots |
| `statsmodels` | Plotly trendline support |
| `joblib` | Model serialisation (sklearn/RF/SVR) |
| `pytest`, `pytest-cov` | Testing framework |

---

## Known Limitations

- **TensorFlow support:** TF does not yet support Python 3.11+. Python 3.10 is required to use the LSTM model.
- **LSTM training time:** The stacked LSTM can take several minutes on CPU for large date ranges. Training progress is visible in the Streamlit dashboard.
- **News sentiment:** The pipeline works fully without real news data (uses neutral zero scores by default). Integrating a live news API would require a custom `from_dataframe()` call.
- **Live trading:** This project is for research and education only. The trading backtest is a simplified simulation and does not account for slippage, taxes, or liquidity constraints.

---

## Disclaimer

This project is developed for **educational and academic research purposes** as part of an MSc (Data Science) capstone at Chandigarh University. Stock price predictions produced by this system do **not** constitute financial advice. Past performance is not indicative of future results.

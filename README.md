# 📈 Algorithmic Trading — Stock Price Prediction with Machine Learning

**MSc (Data Science) Capstone Project · Chandigarh University**

A production-ready, end-to-end machine learning pipeline for predicting NIFTY 50 stock prices, with an interactive Streamlit dashboard.

---

## 🗂️ Project Structure

```
stock_predictor/
├── config.py                   # All tunable parameters
├── requirements.txt
├── src/
│   ├── data_collection.py      # Yahoo Finance fetcher + caching
│   ├── preprocessing.py        # Cleaning, scaling, train/val/test split
│   ├── feature_engineering.py  # Technical indicators + lag features
│   ├── sentiment.py            # VADER-based news sentiment scoring
│   ├── evaluation.py           # Metrics, comparison charts, backtest
│   ├── pipeline.py             # End-to-end orchestrator (CLI + API)
│   ├── database.py             # SQLite persistence layer
│   └── models/
│       ├── base_model.py       # Abstract base class
│       ├── linear_model.py     # Ridge Regression (baseline)
│       ├── rf_model.py         # Random Forest
│       ├── xgb_model.py        # XGBoost (early stopping)
│       ├── svr_model.py        # Support Vector Regression
│       └── lstm_model.py       # Stacked LSTM (TensorFlow/Keras)
├── dashboard/
│   └── app.py                  # Streamlit web dashboard
└── tests/
    └── test_all.py             # pytest unit + integration tests
```

---

## ⚙️ Setup

### 1. Clone & create virtual environment
```bash
git clone <repo-url>
cd stock_predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 🚀 Usage

### Option A — Interactive Dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501 in your browser
```

### Option B — Command-Line Pipeline
```bash
# Train Random Forest + XGBoost + LSTM on RELIANCE (2016-2023)
python -m src.pipeline \
    --ticker RELIANCE \
    --start 2016-01-01 \
    --end   2023-12-31 \
    --models "Random Forest" XGBoost LSTM \
    --sentiment
```

### Option C — Programmatic API
```python
from src.pipeline import StockPredictionPipeline

pipe = StockPredictionPipeline(
    ticker        = "TCS",
    start_date    = "2016-01-01",
    end_date      = "2023-12-31",
    model_names   = ["Random Forest", "XGBoost", "LSTM"],
    use_sentiment = True,
)
evaluator = pipe.run()
print(evaluator.comparison_table())

# Backtest the best model
best = evaluator.best_model()
result = evaluator.simulate_trading(best, initial_capital=100_000)
print(f"Return: {result['Total Return']}%  Sharpe: {result['Sharpe Ratio']}")
```

---

## 🧪 Run Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🤖 Models Implemented

| Model | Type | Key Strengths |
|---|---|---|
| Linear Regression (Ridge) | Baseline | Fast, interpretable |
| Random Forest | Ensemble | Feature importance, robust |
| XGBoost | Gradient Boosting | High accuracy, early stopping |
| SVR | Kernel Method | Good on small datasets |
| LSTM | Deep Learning | Captures temporal dependencies |

---

## 📊 Feature Set (18+ Features)

- **Trend**: SMA-10/20/50, EMA-12/26, MACD, MACD Signal
- **Momentum**: RSI-14, Stochastic %K/%D
- **Volatility**: Bollinger Bands (upper/lower/width/pct), ATR-14
- **Volume**: OBV, VWAP, Volume Ratio
- **Lag**: Close Lag 1/2/3/5/10, Return Lag 1/2/3/5/10
- **Rolling Stats**: Rolling Mean/Std (5/10/20 days)
- **Calendar**: Day of Week, Month, Quarter
- **Sentiment** *(optional)*: VADER compound score, positive/negative ratio

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
Predictions do not constitute financial advice.
Past performance is not indicative of future results.

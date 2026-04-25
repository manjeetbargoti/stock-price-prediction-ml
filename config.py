"""
config.py
---------
Central configuration for the Stock Price Prediction System.
All tunable parameters live here — no magic numbers scattered in code.
"""

from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = BASE_DIR / "saved_models"
LOGS_DIR       = BASE_DIR / "logs"
DB_PATH        = DATA_DIR / "stock_predictor.db"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Stocks ────────────────────────────────────────────────────────────────────
NIFTY_50_TICKERS = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
    "WIPRO", "BAJFINANCE", "ASIANPAINT", "HINDUNILVR", "SUNPHARMA",
    "SBILIFE", "KOTAKBANK", "LT", "AXISBANK", "MARUTI",
]
DEFAULT_TICKER = "RELIANCE"
EXCHANGE_SUFFIX = ".NS"          # NSE India suffix for Yahoo Finance

# ── Data Collection ──────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE   = "2026-04-25"
DATA_INTERVAL      = "1d"
MIN_REQUIRED_ROWS  = 300         # Minimum trading days for reliable training

# ── Preprocessing ─────────────────────────────────────────────────────────────
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO      = 1 - TRAIN_RATIO - VAL_RATIO  # (implicitly 0.15)
OUTLIER_STD_THRESHOLD = 4.0      # Flag values beyond N std devs

# ── Feature Engineering ───────────────────────────────────────────────────────
SMA_WINDOWS       = [10, 20, 50]
EMA_WINDOWS       = [12, 26]
RSI_PERIOD        = 14
STOCH_PERIOD      = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BOLLINGER_WINDOW  = 20
BOLLINGER_STD     = 2
ATR_PERIOD        = 14
LAG_PERIODS       = [1, 2, 3, 5, 10]
ROLLING_WINDOWS   = [5, 10, 20]

# ── LSTM ──────────────────────────────────────────────────────────────────────
LOOK_BACK          = 60          # Timesteps in each input sequence
LSTM_UNITS_1       = 128
LSTM_UNITS_2       = 64
DENSE_UNITS        = 32
DROPOUT_RATE       = 0.2
LSTM_BATCH_SIZE    = 32
LSTM_EPOCHS        = 100
LSTM_LR            = 0.001
EARLY_STOP_PATIENCE = 15

# ── Random Forest ─────────────────────────────────────────────────────────────
RF_N_ESTIMATORS    = 500
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF  = 2
RF_MAX_FEATURES    = "sqrt"

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS   = 1000
XGB_LR             = 0.05
XGB_MAX_DEPTH      = 6
XGB_SUBSAMPLE      = 0.8
XGB_COLSAMPLE      = 0.8
XGB_REG_ALPHA      = 0.1
XGB_REG_LAMBDA     = 1.0
XGB_EARLY_STOPPING = 50

# ── SVR ───────────────────────────────────────────────────────────────────────
SVR_KERNEL         = "rbf"
SVR_C              = 100.0
SVR_EPSILON        = 0.1
SVR_GAMMA          = "scale"

# ── Sentiment ─────────────────────────────────────────────────────────────────
SENTIMENT_DECAY    = 0.9         # Daily decay factor for forward-filled sentiment
NEWS_MAX_HEADLINES = 10          # Headlines per day to fetch (if API available)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE   = LOGS_DIR / "stock_predictor.log"

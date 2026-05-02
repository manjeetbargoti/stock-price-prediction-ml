"""
config.py
---------
Central configuration for the Stock Price Prediction System.
All tunable parameters live here — no magic numbers scattered in code.
"""

from pathlib import Path
from datetime import date

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
]
DEFAULT_TICKER = "RELIANCE"
EXCHANGE_SUFFIX = ".NS"          # NSE India suffix for Yahoo Finance

# ── Data Collection ──────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE   = date.today().strftime("%Y-%m-%d")
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

# ── Forward forecast (dashboard / inference beyond last bar) ─────────────────
FORECAST_WINDOW_DAYS = 7       # Trading days shown per page; "Next" advances one window
FORECAST_MAX_HORIZON = 56      # Total trading days computed (paginated in WINDOW-sized chunks)

# ── LSTM ──────────────────────────────────────────────────────────────────────
LOOK_BACK          = 60          # Timesteps in each input sequence
LSTM_UNITS_1       = 160         # Wider first layer for multivariate sequences
LSTM_UNITS_2       = 80
DENSE_UNITS        = 48
# Input dropout vs recurrent dropout: high recurrent dropout often hurts returns forecasting
LSTM_DROPOUT           = 0.15
LSTM_RECURRENT_DROPOUT = 0.08
DENSE_DROPOUT          = 0.18
LSTM_BATCH_SIZE    = 24
LSTM_EPOCHS        = 200
# EarlyStopping only considers stopping after this many epochs (Keras start_from_epoch)
LSTM_MIN_EPOCHS_BEFORE_EARLY_STOP = 82
LSTM_LR            = 0.0005      # Slower LR; use with ReduceLROnPlateau
LSTM_CLIPNORM      = 1.0         # Stabilizes BPTT on noisy finance series
LSTM_L2            = 1e-5        # Mild L2 on LSTM kernels
EARLY_STOP_PATIENCE   = 28       # Allow longer fit; finance val curves are noisy
EARLY_STOP_MIN_DELTA  = 1e-6     # Ignore tiny val_loss fluctuations
LSTM_LR_PLATEAU_PATIENCE = 6
LSTM_LR_PLATEAU_FACTOR   = 0.5
LSTM_LR_MIN              = 1e-7
# "huber" or "mse" — mse is usually better for univariate scaled Close
LSTM_LOSS          = "mse"
LSTM_HUBER_DELTA   = 0.04
# "return" = predict z-scored daily simple return, reconstruct Close (recommended)
# "close"  = predict MinMax-scaled Close (use with LSTM_UNIVARIATE_INPUT = True)
LSTM_TARGET_MODE   = "return"
# Past scaled Close only when LSTM_TARGET_MODE == "close"
LSTM_UNIVARIATE_INPUT = True
# Nominal ~95% band (normal approx) vs validation residuals; half-width uses max(z·σ, |res| q97.5)
LSTM_CI_Z          = 1.96
# Legacy alias (tabular RF / XGB / SVR still use global dropout if any)
DROPOUT_RATE       = 0.2

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

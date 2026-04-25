# src/models package — optional deps imported lazily to avoid hard failures
from src.models.linear_model import LinearRegressionModel
from src.models.rf_model     import RandomForestModel
from src.models.svr_model    import SVRModel

try:
    from src.models.xgb_model import XGBoostModel
except ImportError:
    XGBoostModel = None  # type: ignore

try:
    from src.models.lstm_model import LSTMModel
except ImportError:
    LSTMModel = None  # type: ignore

__all__ = [
    "LinearRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
    "SVRModel",
    "LSTMModel",
]

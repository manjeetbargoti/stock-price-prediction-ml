"""
src/evaluation.py
------------------
ModelEvaluator — aggregates results from multiple models,
generates comparison tables, plots, and a simulated trading backtest.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.logger import get_logger

log = get_logger(__name__)


class ModelEvaluator:
    """
    Collect and compare results from multiple predictive models.

    Typical Usage
    -------------
    evaluator = ModelEvaluator()
    evaluator.add_result("RandomForest", y_true, y_pred_rf)
    evaluator.add_result("LSTM",         y_true, y_pred_lstm)
    table = evaluator.comparison_table()
    fig   = evaluator.plot_comparison()
    """

    def __init__(self):
        self._results: dict[str, dict] = {}     # model_name -> metrics + arrays

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_result(
        self,
        model_name: str,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        dates:      Optional[pd.DatetimeIndex] = None,
    ) -> dict:
        """Compute metrics and store result for one model."""
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        nz   = y_true != 0
        mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else np.nan
        r2   = r2_score(y_true, y_pred)

        self._results[model_name] = {
            "MAE":   mae,
            "RMSE":  rmse,
            "MAPE":  mape,
            "R2":    r2,
            "y_true": y_true,
            "y_pred": y_pred,
            "dates":  dates,
        }
        log.info("[%s] MAE=%.4f RMSE=%.4f MAPE=%.2f%% R²=%.4f",
                 model_name, mae, rmse, mape, r2)
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    # ── Comparison ────────────────────────────────────────────────────────────

    def comparison_table(self) -> pd.DataFrame:
        """Return a DataFrame with one row per model, sorted by RMSE."""
        rows = []
        for name, res in self._results.items():
            rows.append({
                "Model": name,
                "MAE":   round(res["MAE"],  4),
                "RMSE":  round(res["RMSE"], 4),
                "MAPE":  round(res["MAPE"], 2),
                "R2":    round(res["R2"],   4),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("RMSE")
            .reset_index(drop=True)
        )

    def best_model(self) -> str:
        """Return the name of the model with the lowest RMSE."""
        if not self._results:
            raise ValueError("No results stored yet.")
        return min(self._results, key=lambda n: self._results[n]["RMSE"])

    # ── Plotting (Plotly) ─────────────────────────────────────────────────────

    def plot_predictions(self, model_name: str, ticker: str = ""):
        """Return a Plotly figure: actual vs predicted prices."""
        import plotly.graph_objects as go

        res   = self._results[model_name]
        dates = res["dates"] if res["dates"] is not None else range(len(res["y_true"]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=res["y_true"], name="Actual",
            line=dict(color="#FF6B35", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=res["y_pred"], name=f"{model_name} Prediction",
            line=dict(color="#1F77B4", width=2, dash="dash"),
        ))
        fig.update_layout(
            title=f"{ticker} — {model_name}: Actual vs Predicted",
            xaxis_title="Date", yaxis_title="Price (INR)",
            hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
        )
        return fig

    def plot_comparison_bar(self, metric: str = "RMSE"):
        """Return a Plotly bar chart comparing all models on one metric."""
        import plotly.express as px

        df  = self.comparison_table()
        fig = px.bar(
            df, x="Model", y=metric, color="Model",
            title=f"Model Comparison — {metric}",
            text_auto=".4f", template="plotly_white",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=metric)
        return fig

    def plot_all_predictions(self, ticker: str = ""):
        """Single chart overlaying predictions from every model."""
        import plotly.graph_objects as go

        COLOURS = ["#FF6B35", "#1F77B4", "#2CA02C", "#9467BD", "#D62728", "#8C564B"]
        fig = go.Figure()

        # Actual (from first stored result)
        first = next(iter(self._results.values()))
        dates = first["dates"] if first["dates"] is not None else range(len(first["y_true"]))
        fig.add_trace(go.Scatter(
            x=dates, y=first["y_true"], name="Actual",
            line=dict(color="#000000", width=2.5),
        ))

        for i, (name, res) in enumerate(self._results.items()):
            fig.add_trace(go.Scatter(
                x=dates, y=res["y_pred"], name=name,
                line=dict(color=COLOURS[i % len(COLOURS)], width=1.5, dash="dash"),
            ))

        fig.update_layout(
            title=f"{ticker} — All Model Predictions",
            xaxis_title="Date", yaxis_title="Price (INR)",
            hovermode="x unified", template="plotly_white",
        )
        return fig

    # ── Trading Backtest ──────────────────────────────────────────────────────

    def simulate_trading(
        self,
        model_name:      str,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,     # 0.1 % per trade
    ) -> dict:
        """
        Simple long-only directional strategy:
        - Buy when model predicts price will rise next day.
        - Sell (exit) when model predicts price will fall.

        Returns a dict with Total Return, Sharpe Ratio,
        Max Drawdown, Win Rate, and a trades DataFrame.
        """
        res    = self._results[model_name]
        actual = res["y_true"]
        pred   = res["y_pred"]

        capital   = initial_capital
        position  = 0.0          # shares held
        in_market = False
        portfolio_values = [capital]
        trades    = []

        for i in range(1, len(pred)):
            # Signal: predict tomorrow's direction
            going_up = pred[i] > pred[i - 1]

            if going_up and not in_market:
                # BUY
                shares   = (capital * (1 - transaction_cost)) / actual[i]
                position = shares
                capital  = 0.0
                in_market = True
                trades.append({"day": i, "action": "BUY", "price": actual[i]})

            elif not going_up and in_market:
                # SELL
                capital   = position * actual[i] * (1 - transaction_cost)
                position  = 0.0
                in_market = False
                trades.append({"day": i, "action": "SELL", "price": actual[i]})

            pv = capital + position * actual[i]
            portfolio_values.append(pv)

        # Close any open position at end
        if in_market:
            capital = position * actual[-1] * (1 - transaction_cost)
            portfolio_values[-1] = capital

        portfolio = np.array(portfolio_values)
        returns   = np.diff(portfolio) / portfolio[:-1]

        total_return = (portfolio[-1] - initial_capital) / initial_capital * 100
        bh_return    = (actual[-1] - actual[0]) / actual[0] * 100

        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0 else 0.0
        )

        rolling_max  = np.maximum.accumulate(portfolio)
        drawdowns    = (portfolio - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100

        buy_prices  = [t["price"] for t in trades if t["action"] == "BUY"]
        sell_prices = [t["price"] for t in trades if t["action"] == "SELL"]
        n_pairs     = min(len(buy_prices), len(sell_prices))
        wins        = sum(s > b for b, s in zip(buy_prices[:n_pairs], sell_prices[:n_pairs]))
        win_rate    = wins / n_pairs * 100 if n_pairs > 0 else 0.0

        result = {
            "Model":          model_name,
            "Total Return":   round(total_return, 2),
            "Buy & Hold":     round(bh_return, 2),
            "Sharpe Ratio":   round(sharpe, 3),
            "Max Drawdown":   round(max_drawdown, 2),
            "Win Rate":       round(win_rate, 2),
            "n_trades":       len(trades),
            "portfolio":      portfolio,
            "trades":         pd.DataFrame(trades),
        }
        log.info(
            "[%s] Backtest → Return=%.1f%%  B&H=%.1f%%  Sharpe=%.2f  MaxDD=%.1f%%",
            model_name, total_return, bh_return, sharpe, max_drawdown,
        )
        return result

    def plot_portfolio(self, backtest_result: dict):
        """Return a Plotly line chart of portfolio value over time."""
        import plotly.graph_objects as go

        port  = backtest_result["portfolio"]
        model = backtest_result["Model"]
        fig   = go.Figure()
        fig.add_trace(go.Scatter(
            y=port, name=f"{model} Portfolio",
            fill="tozeroy", line=dict(color="#2CA02C", width=2),
        ))
        fig.update_layout(
            title=f"Simulated Portfolio — {model}",
            xaxis_title="Trading Day", yaxis_title="Portfolio Value (INR)",
            template="plotly_white",
        )
        return fig

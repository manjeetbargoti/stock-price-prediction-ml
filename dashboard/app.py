"""
dashboard/app.py
-----------------
Streamlit web dashboard for the Stock Price Prediction System.
Run with:  streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

import config
from src.data_collection     import StockDataFetcher
from src.preprocessing       import DataPreprocessor
from src.feature_engineering import FeatureEngineer, create_sequences
from src.sentiment           import SentimentAnalyzer
from src.evaluation          import ModelEvaluator
from src.pipeline            import StockPredictionPipeline, MODEL_REGISTRY

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
        border-left: 4px solid #1F4E79;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #1F4E79; }
    .metric-label { font-size: 13px; color: #555; margin-top: 4px; }
    .best-badge {
        background: #28a745; color: white;
        border-radius: 12px; padding: 2px 10px; font-size: 12px;
    }
    h1, h2, h3 { color: #1F4E79; }
</style>
""", unsafe_allow_html=True)


# ── Session State Helpers ─────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "pipeline_run":  False,
        "evaluator":     None,
        "pipe":          None,
        "feature_df":    None,
        "raw_df":        None,
        "test_actual":   None,
        "test_dates":    None,
        "predictions":   {},     # model_name → y_pred (unscaled)
        "preprocessor":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://cdn-websites.onlinecu.in/ONLINECU/public_html/new-assets/images/cu-logo.webp", width=220)
    st.title("⚙️ Configuration")

    ticker = st.selectbox(
        "🏢 Select Stock (NIFTY 50)",
        config.NIFTY_50_TICKERS,
        index=0,
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2021, 1, 1),
            min_value=date(2010, 1, 1),
            max_value=date.today() - timedelta(days=365),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2025, 12, 31),
            min_value=date(2011, 1, 1),
            max_value=date.today(),
        )

    st.divider()
    st.subheader("🤖 Models")
    model_names = st.multiselect(
        "Select models to train",
        list(MODEL_REGISTRY.keys()),
        default=["Random Forest", "XGBoost", "LSTM"],
    )

    st.divider()
    use_sentiment = st.toggle("💬 Enable Sentiment Analysis", value=False)

    st.divider()
    run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📈 Algorithmic Trading — Stock Price Prediction")
st.caption(
    "MSc (Data Science) Capstone Project · Chandigarh University · "
    "Developed with Linear Regression, Random Forest, XGBoost, SVR & LSTM"
)

# ── Run Pipeline ──────────────────────────────────────────────────────────────

if run_btn:
    if not model_names:
        st.error("Please select at least one model.")
        st.stop()

    with st.status("Running prediction pipeline…", expanded=True) as status:
        try:
            st.write(f"📥 Fetching {ticker} data from Yahoo Finance…")
            fetcher = StockDataFetcher(ticker, str(start_date), str(end_date))
            raw_df  = fetcher.fetch()
            st.session_state["raw_df"] = raw_df
            st.write(f"✅ Fetched {len(raw_df)} trading days.")

            st.write("🔧 Preprocessing & feature engineering…")
            preprocessor = DataPreprocessor()
            clean_df     = preprocessor.clean(raw_df)

            s_df = None
            if use_sentiment:
                st.write("💬 Computing sentiment scores…")
                s_df = SentimentAnalyzer.make_zero_scores(clean_df.index)

            engineer   = FeatureEngineer()
            feature_df = engineer.build_features(clean_df, s_df)
            st.session_state["feature_df"] = feature_df

            train_df, val_df, test_df = preprocessor.split(feature_df)
            feature_cols = [c for c in feature_df.columns if c != "Close"]
            all_cols     = feature_cols + ["Close"]

            train_sc = preprocessor.fit_scale(train_df, all_cols)
            val_sc   = preprocessor.transform(val_df)
            test_sc  = preprocessor.transform(test_df)
            st.session_state["preprocessor"] = preprocessor

            X_tr_2d  = train_sc[feature_cols].values
            y_tr     = train_sc["Close"].values
            X_val_2d = val_sc[feature_cols].values
            y_val    = val_sc["Close"].values
            X_te_2d  = test_sc[feature_cols].values
            y_te     = test_sc["Close"].values

            X_tr_3d,  y_tr_3d  = create_sequences(X_tr_2d,  y_tr)
            X_val_3d, y_val_3d = create_sequences(X_val_2d, y_val)
            X_te_3d,  y_te_3d  = create_sequences(X_te_2d,  y_te)

            test_actual = preprocessor.inverse_transform_prices(y_te_3d)
            test_dates  = test_df.index[config.LOOK_BACK:]
            st.session_state["test_actual"] = test_actual
            st.session_state["test_dates"]  = test_dates

            evaluator = ModelEvaluator()

            for mname in model_names:
                st.write(f"🧠 Training {mname}…")
                import importlib
                module_path, class_name = MODEL_REGISTRY[mname].rsplit(".", 1)
                ModelClass = getattr(importlib.import_module(module_path), class_name)
                model      = ModelClass()

                is_lstm = mname == "LSTM"
                X_tr_in   = X_tr_3d  if is_lstm else X_tr_2d[config.LOOK_BACK:]
                y_tr_in   = y_tr_3d  if is_lstm else y_tr[config.LOOK_BACK:]
                X_val_in  = X_val_3d if is_lstm else X_val_2d[config.LOOK_BACK:]
                y_val_in  = y_val_3d if is_lstm else y_val[config.LOOK_BACK:]
                X_te_in   = X_te_3d  if is_lstm else X_te_2d[config.LOOK_BACK:]

                kw = {"X_val": X_val_in, "y_val": y_val_in}
                if not is_lstm:
                    kw["feature_names"] = feature_cols
                model.train(X_tr_in, y_tr_in, **kw)

                y_pred_sc = model.predict(X_te_in)
                y_pred    = preprocessor.inverse_transform_prices(y_pred_sc)

                evaluator.add_result(mname, test_actual, y_pred, test_dates)
                st.session_state["predictions"][mname] = y_pred

                st.write(f"   ✅ {mname} done.")

            st.session_state["evaluator"]    = evaluator
            st.session_state["pipeline_run"] = True
            status.update(label="✅ Pipeline complete!", state="complete")

        except Exception as exc:
            status.update(label=f"❌ Error: {exc}", state="error")
            st.exception(exc)


# ── Main Content ──────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 Data Overview",
    "📉 Technical Indicators",
    "🎯 Model Predictions",
    "🏆 Performance Metrics",
    "💹 Trading Signals",
])


# ── Tab 1: Data Overview ──────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"📊 {ticker} — Historical Price Data")
    raw_df = st.session_state.get("raw_df")

    if raw_df is None:
        st.info("Configure the sidebar and click **Run Prediction** to load data.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trading Days", f"{len(raw_df):,}")
        c2.metric("Latest Close",  f"₹{raw_df['Close'].iloc[-1]:,.2f}")
        c3.metric("All-time High", f"₹{raw_df['High'].max():,.2f}")
        c4.metric("All-time Low",  f"₹{raw_df['Low'].min():,.2f}")

        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=raw_df.index,
            open=raw_df["Open"], high=raw_df["High"],
            low=raw_df["Low"],   close=raw_df["Close"],
            increasing_line_color="#26A65B",
            decreasing_line_color="#E74C3C",
            name="OHLC",
        )])
        fig.update_layout(
            title=f"{ticker} — Candlestick Chart",
            xaxis_title="Date", yaxis_title="Price (INR)",
            template="plotly_white", xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume bar
        vol_fig = px.bar(raw_df, y="Volume", title="Daily Volume",
                         template="plotly_white", color_discrete_sequence=["#1F77B4"])
        st.plotly_chart(vol_fig, use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(raw_df.describe().round(2), use_container_width=True)


# ── Tab 2: Technical Indicators ───────────────────────────────────────────────
with tabs[1]:
    st.subheader("📉 Technical Indicators")
    feat_df = st.session_state.get("feature_df")

    if feat_df is None:
        st.info("Run the pipeline first to compute indicators.")
    else:
        indicator_choice = st.selectbox(
            "Select Indicator Group",
            ["Price + Moving Averages", "RSI", "MACD", "Bollinger Bands", "Volume (OBV)"]
        )

        if indicator_choice == "Price + Moving Averages":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["Close"],
                                     name="Close", line=dict(color="#000", width=1.5)))
            for col, colour in zip(["sma_20","sma_50"], ["#1F77B4","#FF7F0E"]):
                if col in feat_df.columns:
                    fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df[col],
                                             name=col.upper(), line=dict(color=colour)))
            if "ema_12" in feat_df.columns:
                fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["ema_12"],
                                         name="EMA-12", line=dict(color="#2CA02C", dash="dot")))
            fig.update_layout(title="Price & Moving Averages", template="plotly_white",
                              yaxis_title="Price (INR)")
            st.plotly_chart(fig, use_container_width=True)

        elif indicator_choice == "RSI":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["rsi"],
                                     name="RSI-14", line=dict(color="#9467BD")))
            fig.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig.update_layout(title="Relative Strength Index (14-day)",
                              yaxis_title="RSI", yaxis_range=[0, 100],
                              template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        elif indicator_choice == "MACD":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["macd"],
                                     name="MACD", line=dict(color="#1F77B4")))
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["macd_signal"],
                                     name="Signal", line=dict(color="#FF7F0E")))
            if "macd_diff" in feat_df:
                fig.add_trace(go.Bar(x=feat_df.index, y=feat_df["macd_diff"],
                                     name="Histogram", marker_color="#AAD4F5", opacity=0.6))
            fig.update_layout(title="MACD", template="plotly_white", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

        elif indicator_choice == "Bollinger Bands":
            fig = go.Figure([
                go.Scatter(x=feat_df.index, y=feat_df["bb_upper"],
                           name="Upper Band", line=dict(color="#E74C3C", dash="dot")),
                go.Scatter(x=feat_df.index, y=feat_df["Close"],
                           name="Close", fill="tonexty", fillcolor="rgba(31,119,180,0.1)",
                           line=dict(color="#1F77B4")),
                go.Scatter(x=feat_df.index, y=feat_df["bb_lower"],
                           name="Lower Band", fill="tonexty", fillcolor="rgba(31,119,180,0.1)",
                           line=dict(color="#2CA02C", dash="dot")),
            ])
            fig.update_layout(title="Bollinger Bands (20-day, 2σ)",
                              yaxis_title="Price (INR)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        elif indicator_choice == "Volume (OBV)":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["obv"],
                                     name="OBV", line=dict(color="#8C564B")))
            fig.update_layout(title="On-Balance Volume", yaxis_title="OBV",
                              template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Predictions ────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("🎯 Model Predictions vs Actual")
    evaluator = st.session_state.get("evaluator")

    if evaluator is None:
        st.info("Run the pipeline first to see predictions.")
    else:
        test_actual = st.session_state["test_actual"]
        test_dates  = st.session_state["test_dates"]
        predictions = st.session_state["predictions"]

        # All-models overlay
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_dates, y=test_actual, name="Actual Price",
            line=dict(color="black", width=2.5),
        ))
        colours = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD"]
        for i, (mname, y_pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=test_dates, y=y_pred, name=mname,
                line=dict(color=colours[i % len(colours)], width=1.5, dash="dash"),
            ))
        fig.update_layout(
            title=f"{ticker} — All Models: Actual vs Predicted",
            xaxis_title="Date", yaxis_title="Price (INR)",
            hovermode="x unified", template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-model scatter
        st.subheader("Scatter: Actual vs Predicted (per model)")
        model_sel = st.selectbox("Choose model", list(predictions.keys()), key="scatter_sel")
        y_pred = predictions[model_sel]
        scatter = px.scatter(
            x=test_actual, y=y_pred,
            labels={"x": "Actual Price (INR)", "y": "Predicted Price (INR)"},
            title=f"{model_sel} — Actual vs Predicted Scatter",
            opacity=0.6, trendline="ols", template="plotly_white",
        )
        st.plotly_chart(scatter, use_container_width=True)


# ── Tab 4: Performance Metrics ────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🏆 Model Performance Comparison")
    evaluator = st.session_state.get("evaluator")

    if evaluator is None:
        st.info("Run the pipeline first.")
    else:
        table = evaluator.comparison_table()
        best  = evaluator.best_model()

        # Metric cards
        cols = st.columns(len(table))
        for i, row in table.iterrows():
            with cols[i]:
                badge = "🥇 " if row["Model"] == best else ""
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-value">{badge}{row['R2']:.4f}</div>
                    <div class="metric-label">{row['Model']}<br>R² Score</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.dataframe(
            table.style
            .highlight_min(subset=["RMSE","MAE","MAPE"], color="#d4edda")
            .highlight_max(subset=["R2"], color="#d4edda")
            .format({"MAE":"₹{:.2f}", "RMSE":"₹{:.2f}", "MAPE":"{:.2f}%", "R2":"{:.4f}"}),
            use_container_width=True,
        )

        # Bar charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(evaluator.plot_comparison_bar("RMSE"), use_container_width=True)
        with c2:
            st.plotly_chart(evaluator.plot_comparison_bar("R2"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(evaluator.plot_comparison_bar("MAE"), use_container_width=True)
        with c4:
            st.plotly_chart(evaluator.plot_comparison_bar("MAPE"), use_container_width=True)


# ── Tab 5: Trading Signals ────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("💹 Simulated Trading Strategy")
    evaluator = st.session_state.get("evaluator")

    if evaluator is None:
        st.info("Run the pipeline first.")
    else:
        bt_model = st.selectbox(
            "Select model for backtest",
            list(st.session_state["predictions"].keys()),
            key="bt_model",
        )
        capital  = st.number_input("Initial Capital (INR)", value=100_000, step=10_000)

        if st.button("▶ Run Backtest"):
            result = evaluator.simulate_trading(bt_model, float(capital))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Return",   f"{result['Total Return']:.1f}%")
            c2.metric("Buy & Hold",     f"{result['Buy & Hold']:.1f}%")
            c3.metric("Sharpe Ratio",   f"{result['Sharpe Ratio']:.3f}")
            c4.metric("Max Drawdown",   f"{result['Max Drawdown']:.1f}%")
            c5.metric("Win Rate",       f"{result['Win Rate']:.1f}%")

            st.plotly_chart(evaluator.plot_portfolio(result), use_container_width=True)

            if not result["trades"].empty:
                st.subheader("Trade Log")
                st.dataframe(result["trades"], use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Disclaimer: This application is for **educational and research purposes only**. "
    "Predictions do not constitute financial advice. Past performance is not indicative "
    "of future results."
)

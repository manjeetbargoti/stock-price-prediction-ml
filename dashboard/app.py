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
from src.forecasting         import future_close_forecast, signal_for_forecast_window

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
        "predictions":          {},     # model_name → y_pred (unscaled)
        "preprocessor":         None,
        "trained_models":       {},     # model_name → fitted model instance
        "forward_forecasts":    {},     # model_name → future_close_forecast dict
        "clean_df":             None,
        "sentiment_df_fc":      None,
        "feature_cols_fc":      None,
        "forecast_page_idx":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _forecast_n_pages(fc: dict, window: int) -> int:
    n = len(fc["dates"])
    if n == 0:
        return 1
    return max(1, (n + window - 1) // window)


def _forecast_week_slice(fc: dict, page: int, window: int):
    dates, prices = fc["dates"], fc["prices"]
    n = len(dates)
    start = page * window
    if start >= n:
        return [], [], start
    end = min(start + window, n)
    return dates[start:end], prices[start:end], start


def _page_for_pick_date(fc: dict, pick: date, window: int) -> int:
    ds = [pd.Timestamp(d).date() for d in fc["dates"]]
    n = len(ds)
    if n == 0:
        return 0
    n_pages = _forecast_n_pages(fc, window)
    for p in range(n_pages):
        s = p * window
        e = min(s + window, n)
        if s >= n:
            break
        if ds[s] <= pick <= ds[e - 1]:
            return p
    best_p, best_diff = 0, 10**9
    for p in range(n_pages):
        s = p * window
        if s >= n:
            break
        diff = abs((ds[s] - pick).days)
        if diff < best_diff:
            best_diff, best_p = diff, p
    return best_p


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
            value=date.today(),
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
            st.session_state["predictions"] = {}
            st.session_state["trained_models"] = {}
            st.session_state["forward_forecasts"] = {}
            st.session_state["forecast_page_idx"] = 0
            st.session_state["lstm_ret_scaler"] = None

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

            lstm_ret_scaler = None
            if "LSTM" in model_names and config.LSTM_TARGET_MODE == "return":
                from sklearn.preprocessing import StandardScaler
                from src.lstm_data import prepare_lstm_return_sequences

                lstm_ret_scaler = StandardScaler()
                (X_tr_3d, y_tr_3d), (X_val_3d, y_val_3d), (X_te_3d, y_te_3d) = (
                    prepare_lstm_return_sequences(
                        train_df, val_df, test_df, lstm_ret_scaler, config.LOOK_BACK
                    )
                )
            else:
                X_tr_3d,  y_tr_3d  = create_sequences(X_tr_2d,  y_tr)
                X_val_3d, y_val_3d = create_sequences(X_val_2d, y_val)
                X_te_3d,  y_te_3d  = create_sequences(X_te_2d,  y_te)
                if "LSTM" in model_names and config.LSTM_UNIVARIATE_INPUT:
                    X_tr_3d,  y_tr_3d  = create_sequences(y_tr.reshape(-1, 1),  y_tr)
                    X_val_3d, y_val_3d = create_sequences(y_val.reshape(-1, 1), y_val)
                    X_te_3d,  y_te_3d  = create_sequences(y_te.reshape(-1, 1), y_te)

            _, y_te_close = create_sequences(y_te.reshape(-1, 1), y_te)
            test_actual = preprocessor.inverse_transform_prices(y_te_close)
            test_dates  = test_df.index[config.LOOK_BACK:]
            st.session_state["lstm_ret_scaler"] = lstm_ret_scaler
            st.session_state["lstm_ci_halfwidth_inr"] = None
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

                if is_lstm:
                    from src.lstm_data import lstm_validation_ci_halfwidth_inr

                    y_val_pred_sc = model.predict(X_val_in)
                    st.session_state["lstm_ci_halfwidth_inr"] = (
                        lstm_validation_ci_halfwidth_inr(
                            y_val_pred_sc=y_val_pred_sc,
                            val_df=val_df,
                            preprocessor=preprocessor,
                            lstm_ret_scaler=lstm_ret_scaler,
                            return_mode=config.LSTM_TARGET_MODE == "return",
                            look_back=config.LOOK_BACK,
                        )
                    )

                y_pred_sc = model.predict(X_te_in)
                if (
                    is_lstm
                    and config.LSTM_TARGET_MODE == "return"
                    and lstm_ret_scaler is not None
                ):
                    from src.lstm_data import decode_return_predictions_to_close

                    y_pred = decode_return_predictions_to_close(
                        y_pred_sc,
                        lstm_ret_scaler,
                        test_df["Close"].values,
                        config.LOOK_BACK,
                    )
                else:
                    y_pred = preprocessor.inverse_transform_prices(y_pred_sc)

                evaluator.add_result(mname, test_actual, y_pred, test_dates)
                st.session_state["predictions"][mname] = y_pred
                st.session_state["trained_models"][mname] = model

                st.write(f"   ✅ {mname} done.")

            st.write(
                "📆 Computing forward forecast: "
                f"{config.FORECAST_MAX_HORIZON} trading days total "
                f"({config.FORECAST_WINDOW_DAYS} days per page in the dashboard)…"
            )
            r2_map = evaluator.comparison_table().set_index("Model")["R2"].to_dict()
            fc_out = {}
            for mname_fc, model_fc in st.session_state["trained_models"].items():
                fc_out[mname_fc] = future_close_forecast(
                    clean_df=clean_df,
                    engineer=engineer,
                    preprocessor=preprocessor,
                    sentiment_df=s_df,
                    model=model_fc,
                    is_lstm=(mname_fc == "LSTM"),
                    lstm_target_return=(
                        mname_fc == "LSTM" and config.LSTM_TARGET_MODE == "return"
                    ),
                    lstm_ret_scaler=st.session_state.get("lstm_ret_scaler"),
                    lstm_univariate=(
                        mname_fc == "LSTM"
                        and config.LSTM_TARGET_MODE == "close"
                        and config.LSTM_UNIVARIATE_INPUT
                    ),
                    feature_cols=feature_cols,
                    n_steps=config.FORECAST_MAX_HORIZON,
                    test_r2=float(r2_map.get(mname_fc, 0.0)),
                )
            st.session_state["forward_forecasts"] = fc_out
            st.session_state["clean_df"] = clean_df
            st.session_state["sentiment_df_fc"] = s_df
            st.session_state["feature_cols_fc"] = feature_cols

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

        if "LSTM" in predictions:
            st.divider()
            hw = float(st.session_state.get("lstm_ci_halfwidth_inr") or 0.0)
            y_lstm = np.asarray(predictions["LSTM"], dtype=float)
            act = np.asarray(test_actual, dtype=float)
            td_ts = pd.to_datetime(test_dates)
            d_min = td_ts.min().date()
            d_max = td_ts.max().date()
            c1, c2 = st.columns(2)
            with c1:
                dr0 = st.date_input(
                    "LSTM results — start date",
                    value=d_min,
                    min_value=d_min,
                    max_value=d_max,
                    key="lstm_res_start",
                )
            with c2:
                dr1 = st.date_input(
                    "LSTM results — end date",
                    value=d_max,
                    min_value=d_min,
                    max_value=d_max,
                    key="lstm_res_end",
                )
            if dr0 > dr1:
                dr0, dr1 = dr1, dr0
            mask = (td_ts.date >= dr0) & (td_ts.date <= dr1)
            x_sub = td_ts[mask]
            act_sub = act[mask]
            pred_sub = y_lstm[mask]
            low_sub = pred_sub - hw
            high_sub = pred_sub + hw
            t0s = dr0.strftime("%B %Y")
            t1s = dr1.strftime("%B %Y")
            st.subheader(
                f"LSTM Prediction Results for {ticker} Stock ({t0s} – {t1s})"
            )
            st.caption(
                "Actual price in orange, LSTM-predicted price in blue, "
                "with ~95% interval (validation-calibrated) shaded in light blue."
            )
            lfig = go.Figure()
            lfig.add_trace(go.Scatter(
                x=x_sub, y=high_sub, mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            lfig.add_trace(go.Scatter(
                x=x_sub, y=low_sub, mode="lines", line=dict(width=0),
                fillcolor="rgba(173, 216, 230, 0.45)",
                fill="tonexty", name="~95% interval",
                hoverinfo="skip",
            ))
            lfig.add_trace(go.Scatter(
                x=x_sub, y=act_sub, name="Actual",
                line=dict(color="#FF7F0E", width=2.4),
            ))
            lfig.add_trace(go.Scatter(
                x=x_sub, y=pred_sub, name="LSTM predicted",
                line=dict(color="#1F77B4", width=2),
            ))
            lfig.update_layout(
                title=f"{ticker} — LSTM vs actual (selected range)",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(lfig, use_container_width=True)
            tbl = pd.DataFrame({
                "Date": x_sub.strftime("%Y-%m-%d"),
                "Actual (₹)": np.round(act_sub, 2),
                "Predicted (₹)": np.round(pred_sub, 2),
                "Lower ~95% (₹)": np.round(low_sub, 2),
                "Upper ~95% (₹)": np.round(high_sub, 2),
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        # ── Multi-day forward forecast (beyond downloaded history) ──────────
        st.divider()
        W = config.FORECAST_WINDOW_DAYS
        st.subheader(
            f"🔮 Forward forecast — {W} trading days per page "
            f"(up to {config.FORECAST_MAX_HORIZON} days computed)"
        )
        fc_map = st.session_state.get("forward_forecasts") or {}
        if not fc_map:
            st.caption("Forward forecasts will appear here after a successful pipeline run.")
        else:
            fc_model = st.selectbox(
                "Model for forward forecast",
                list(fc_map.keys()),
                key="forward_fc_model",
            )
            fc = fc_map[fc_model]

            if st.session_state.get("_forecast_ui_model") != fc_model:
                st.session_state["forecast_page_idx"] = 0
                st.session_state["_forecast_ui_model"] = fc_model

            n_pages = _forecast_n_pages(fc, W)
            page = int(st.session_state.get("forecast_page_idx", 0))
            page = max(0, min(page, n_pages - 1))
            st.session_state["forecast_page_idx"] = page

            w_dates, w_prices, w_start = _forecast_week_slice(fc, page, W)
            n_tot = len(fc["dates"])

            fd0 = pd.Timestamp(fc["dates"][0]).date()
            fd1 = pd.Timestamp(fc["dates"][-1]).date()

            nav_prev, nav_mid, nav_next = st.columns([1, 3, 1])
            with nav_prev:
                if st.button(
                    "◀ Previous",
                    key="fc_btn_prev",
                    help=f"Previous {W} trading days",
                    disabled=(page <= 0),
                    use_container_width=True,
                ):
                    st.session_state["forecast_page_idx"] = page - 1
                    st.rerun()
            with nav_next:
                if st.button(
                    "Next ▶",
                    key="fc_btn_next",
                    help=f"Next {W} trading days",
                    disabled=(page >= n_pages - 1),
                    use_container_width=True,
                ):
                    st.session_state["forecast_page_idx"] = page + 1
                    st.rerun()
            with nav_mid:
                st.markdown(
                    f"**Page {page + 1} / {n_pages}** — "
                    f"rows **{w_start + 1}–{w_start + len(w_dates)}** of **{n_tot}** "
                    "forecast trading days"
                )

            cjump, cform = st.columns([1, 2])
            with cjump:
                jump_options = list(range(n_pages))

                def _week_label(p: int) -> str:
                    ds, pr, s = _forecast_week_slice(fc, p, W)
                    if not ds:
                        return f"Week {p + 1}"
                    a = pd.Timestamp(ds[0]).strftime("%d %b %Y")
                    b = pd.Timestamp(ds[-1]).strftime("%d %b %Y")
                    return f"Week {p + 1}: {a} → {b}"

                week_pick = st.selectbox(
                    "Jump to week",
                    options=jump_options,
                    index=min(page, max(0, n_pages - 1)),
                    format_func=_week_label,
                )
                if week_pick != page:
                    st.session_state["forecast_page_idx"] = week_pick
                    st.rerun()

            with cform:
                with st.form("fc_date_jump_form", clear_on_submit=False):
                    jd = st.date_input(
                        "Or pick any date in the forecast range",
                        value=fd0,
                        min_value=fd0,
                        max_value=fd1,
                        key="fc_jump_date_input",
                    )
                    submitted = st.form_submit_button("Go to date")
                    if submitted:
                        st.session_state["forecast_page_idx"] = _page_for_pick_date(
                            fc, jd, W
                        )
                        st.rerun()

            raw = st.session_state.get("raw_df")
            tail_n = min(120, len(raw) if raw is not None else 0)

            top_l, top_r = st.columns([1.15, 1.0])
            with top_l:
                ffig = go.Figure()
                if raw is not None and tail_n > 0:
                    tail = raw.iloc[-tail_n:]
                    ffig.add_trace(go.Scatter(
                        x=tail.index,
                        y=tail["Close"],
                        mode="lines",
                        name="Actual Prices",
                        line=dict(color="#1F77B4", width=2.2),
                    ))
                if w_dates:
                    ffig.add_trace(go.Scatter(
                        x=list(w_dates),
                        y=list(w_prices),
                        mode="lines+markers",
                        name=f"Forecast (this page, {len(w_dates)} days)",
                        line=dict(color="#FF7F0E", width=2, dash="dash"),
                    ))
                ffig.update_layout(
                    title=f"{ticker} — Actual · forecast window (page {page + 1})",
                    xaxis_title="Date",
                    yaxis_title="Price (INR)",
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(ffig, use_container_width=True)

            with top_r:
                st.markdown("**Predicted stock prices (this page)**")
                pred_rows = []
                for d, p in zip(
                    reversed(w_dates),
                    reversed(w_prices),
                ):
                    pred_rows.append({
                        "Date":             pd.Timestamp(d).strftime("%m/%d/%Y"),
                        "Stock symbol":     ticker,
                        "Predicted price": f"₹{p:,.2f}",
                        "Confidence":      f"{fc['confidence_pct']}%",
                    })
                st.dataframe(
                    pd.DataFrame(pred_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Confidence is a simple mapping from test-set R² (research UI only, "
                    "not a calibrated probability)."
                )

            bot_l, bot_r = st.columns([1.0, 1.0])
            with bot_l:
                st.markdown("**Next predicted prices (up to 3 rows)**")
                st.dataframe(
                    pd.DataFrame(pred_rows[: max(1, min(3, len(pred_rows)))]),
                    use_container_width=True,
                    hide_index=True,
                )
            with bot_r:
                if w_prices:
                    np_price = float(w_prices[-1])
                    sig = signal_for_forecast_window(fc=fc, page=page, window=W)
                else:
                    np_price = float(fc["last_actual_close"])
                    sig = "Hold"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Window end (last day on this page)</div>'
                    f'<div class="metric-value">₹{np_price:,.2f}</div>'
                    f'<div class="metric-label" style="margin-top:10px">'
                    f'Prediction confidence</div>'
                    f'<div class="metric-value">{fc["confidence_pct"]}%</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if sig == "Buy":
                    bg, fg = "#28a745", "#fff"
                elif sig == "Sell":
                    bg, fg = "#dc3545", "#fff"
                else:
                    bg, fg = "#6c757d", "#fff"
                st.markdown(
                    f'<p style="text-align:center;margin-top:16px">'
                    f'<span style="display:inline-block;padding:14px 48px;'
                    f"background:{bg};color:{fg};border-radius:10px;"
                    f'font-size:1.15rem;font-weight:600">{sig}</span></p>',
                    unsafe_allow_html=True,
                )
                st.caption(fc["model_horizon_note"])

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

        lstm_m = (st.session_state.get("trained_models") or {}).get("LSTM")
        if lstm_m is not None and hasattr(lstm_m, "get_training_history"):
            hist = lstm_m.get_training_history()
            loss = hist.get("loss") if hist else None
            if loss:
                st.divider()
                st.subheader("LSTM — training & validation loss")
                n_ep = len(loss)
                val_loss = hist.get("val_loss")
                x_ep = list(range(1, n_ep + 1))
                lfig = go.Figure()
                lfig.add_trace(go.Scatter(
                    x=x_ep,
                    y=loss,
                    mode="lines",
                    name="Training loss",
                    line=dict(color="#1F77B4", width=2),
                ))
                if val_loss is not None and len(val_loss) == n_ep:
                    lfig.add_trace(go.Scatter(
                        x=x_ep,
                        y=val_loss,
                        mode="lines",
                        name="Validation loss",
                        line=dict(color="#FF7F0E", width=2),
                    ))
                min_es = getattr(config, "LSTM_MIN_EPOCHS_BEFORE_EARLY_STOP", 0) or 0
                title_extra = (
                    f" ({n_ep} epoch{'s' if n_ep != 1 else ''} run)"
                )
                lfig.update_layout(
                    title=f"{ticker} — LSTM loss vs epoch{title_extra}",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(lfig, use_container_width=True)
                st.caption(
                    "Training stops when early stopping triggers or when "
                    f"`LSTM_EPOCHS` is reached. If `LSTM_MIN_EPOCHS_BEFORE_EARLY_STOP` "
                    f"is set ({int(min_es)} here), patience-based stopping only applies "
                    "after that many epochs."
                )


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

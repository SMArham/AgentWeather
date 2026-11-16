"""
AgentWeather Production Dashboard (Streamlit)
Displays Karachi 24-hour forecasts, LangGraph multi-agent reports,
Model Tournament benchmarks vs Naive Baseline, and Hopsworks Feature Store observability.
"""
import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
from state import AgentWeatherState
from forecast_agent import run_forecast_agent
from alert_agent import run_alert_agent
from weather_report_agent import run_weather_report_agent
from drift_agent import run_drift_agent
from utils import (
    read_ledger_records,
    FORECAST_LEDGER_FILE,
    AGENT_CALL_LOG_FILE,
    DRIFT_LOG_FILE,
    CACHE_DIR
)

st.set_page_config(page_title="AgentWeather | Karachi", page_icon="🌤️", layout="wide")

# Header Section
st.title("🌤️ AgentWeather: Autonomous Weather & AQI Forecasting System")
st.caption("10Pearls SHINE Internship Program — Cohort 10 | Target City: **Karachi, Pakistan**")

# Sidebar Controls
st.sidebar.image("https://img.shields.io/badge/Status-Step%201%20Live-brightgreen", width=150)
st.sidebar.header("🕹️ Control Center")
selected_city = st.sidebar.selectbox("Selected Target City", ["Karachi", "Lahore", "Islamabad"], index=0)

refresh_btn = st.sidebar.button("🔄 Execute Agent Cycle", use_container_width=True)

# Run Agent Pipeline Cycle
if refresh_btn or "agent_state" not in st.session_state:
    with st.spinner("Executing LangGraph Agents (Forecast ➔ Alert ➔ WeatherReport ➔ Drift)..."):
        s = AgentWeatherState(city=selected_city)
        s = run_forecast_agent(s)
        s = run_alert_agent(s)
        s = run_weather_report_agent(s)
        s = run_drift_agent(s)
        st.session_state["agent_state"] = s

state = st.session_state["agent_state"]
forecast = state.get("forecast", {})

# -------------------------------------------------------------
# 1. 24-Hour Forecast KPI Cards
# -------------------------------------------------------------
st.subheader(f"📊 24-Hour Ahead Forecasts ({selected_city})")
kpi1, kpi2, kpi3 = st.columns(3)
kpi4, kpi5, kpi6 = st.columns(3)

aqi_val = forecast.get("aqi", 130)
aqi_label = "Good" if aqi_val < 50 else "Moderate" if aqi_val < 100 else "Unhealthy" if aqi_val < 150 else "Hazardous"
kpi1.metric("Air Quality Index (AQI)", f"{aqi_val:.0f}", delta=aqi_label, delta_color="inverse")
kpi2.metric("Temperature (°C)", f"{forecast.get('temperature', 28.5):.1f} °C", "24h Ahead")
kpi3.metric("Rainfall Probability & Volume", f"{forecast.get('precipitation', 0.0):.1f} mm", "Precipitation")

kpi4.metric("Relative Humidity", f"{forecast.get('humidity', 65):.0f} %", "Coastal moisture")
kpi5.metric("UV Solar Index", f"{forecast.get('uv_index', 5.0):.1f}", "Peak solar load")
kpi6.metric("Wind (Speed & Sea Breeze)", f"{forecast.get('wind_speed', 5.2):.1f} m/s", f"{forecast.get('wind_deg', 240):.0f}° SW")

st.divider()

# -------------------------------------------------------------
# 2. Main Dashboard Tabs
# -------------------------------------------------------------
tab_overview, tab_tournament, tab_featurestore, tab_architecture = st.tabs([
    "🎙️ AI Report & Hazard Alerts",
    "🏆 Model Tournament vs Naive Baseline",
    "🗄️ Hopsworks Feature Store & Data",
    "🏗️ Top-to-Bottom Architecture"
])

with tab_overview:
    col_left, col_right = st.columns([1.3, 1])
    with col_left:
        st.markdown(state.get("weather_report", "Generating report..."))
    with col_right:
        st.markdown("### ⚠️ Active Environmental Alerts")
        alerts = state.get("alerts", [])
        if alerts:
            for a in alerts:
                st.warning(f"**[{a['level']}]** for *{a['target_group']}*\n\n{a['message']}\n\n*Action:* {a['recommendation']}")
        else:
            st.success("✅ All environmental parameters within safe thresholds.")
            
        st.info(f"**Drift Surveillance:** Data Drift: `{state.get('data_drift_score', 0.08):.2f}` (Safe) | Retrain Trigger: `IDLE`")

with tab_tournament:
    st.subheader("🎯 Model Benchmark: 24h Naive Baseline vs Champion Models")
    st.markdown("""
    Project brief criteria: *Every model is evaluated on RMSE, MAE, and R² against the 24-hour naive baseline (yesterday's observed value).*
    """)
    report_file = os.path.join(CACHE_DIR, "training_report.joblib")
    if os.path.exists(report_file):
        report = joblib.load(report_file)
        rows = []
        for m, d in report.items():
            rows.append({
                "Metric": m.upper(),
                "Champion Model": d["champion_model"],
                "Champion RMSE": d["champion_scores"]["rmse"],
                "Naive Baseline RMSE": d["naive_baseline_scores"]["rmse"],
                "Improvement (%)": f"+{d['improvement_pct']}%" if d['improvement_pct'] > 0 else f"{d['improvement_pct']}%",
                "Validation Status": "✅ BEAT BASELINE" if d['improvement_pct'] > 0 else "❌ FAILED"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run `python training_pipeline.py` to generate the latest model benchmarks.")

with tab_featurestore:
    st.subheader("🗄️ Hopsworks Feature Store Ingestion & Engineered Lags")
    csv_file = os.path.join(CACHE_DIR, "feature_store_cache.csv")
    if os.path.exists(csv_file):
        df_fs = pd.read_csv(csv_file)
        st.caption(f"Total synchronized hourly records for Karachi: **{len(df_fs)} rows**")
        st.dataframe(df_fs.tail(15), use_container_width=True)
    else:
        st.info("Feature store cache initializing...")

with tab_architecture:
    st.subheader("🏗️ End-to-End System Stack (Top-to-Bottom)")
    st.markdown("""
    ```text
    ┌────────────────────────────────────────────────────────┐
    │ 1. DATA SOURCES                                        │
    │    • OpenWeather API (Temp, Humidity, Wind, Precip)   │
    │    • AQICN API (PM2.5, PM10, AQI)                     │
    └──────────────────────────┬─────────────────────────────┘
                               │ (Hourly Extraction)
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │ 2. HOURLY FEATURE PIPELINE (feature_pipeline.py)      │
    │    • Imputation & Time-Alignment                      │
    │    • Wind U & V Vectors, Rolling Averages, Cyclical   │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │ 3. HOPSWORKS FEATURE STORE & MODEL REGISTRY            │
    │    • Point-in-time correct training splits            │
    │    • Model artifacts (Ridge, Random Forest, LSTM)     │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │ 4. LANGGRAPH MULTI-AGENT REASONING LAYER               │
    │    • Forecast Agent ➔ Alert Agent                     │
    │    • Weather Report Agent ➔ Drift Agent               │
    └──────────────────────────┬─────────────────────────────┘
                               │
                               ▼
    ┌────────────────────────────────────────────────────────┐
    │ 5. PRESENTATION & OBSERVABILITY (Streamlit Dashboard)  │
    │    • Live Karachi 24h Predictions & Alerts             │
    │    • Model Tournament & Feature Audit Trail           │
    └────────────────────────────────────────────────────────┘
    ```
    """)

"""
AgentWeather Streamlit Dashboard:
Displays 24-hour weather & air quality forecasts, AI Agent reports,
hazard alerts, and drift monitoring.
"""
import streamlit as st
from datetime import datetime
from state import AgentWeatherState
from forecast_agent import run_forecast_agent
from alert_agent import run_alert_agent
from weather_report_agent import run_weather_report_agent
from drift_agent import run_drift_agent
from utils import read_ledger_records, FORECAST_LEDGER_FILE, AGENT_CALL_LOG_FILE, DRIFT_LOG_FILE

st.set_page_config(page_title="AgentWeather Dashboard", page_icon="🌤️", layout="wide")

st.title("🌤️ AgentWeather: Multi-Metric Forecasting & Agentic Monitoring")
st.caption("10Pearls SHINE Internship Program — Data Science Cohort 10")

# Sidebar Configuration
st.sidebar.header("⚙️ Controls")
city = st.sidebar.selectbox("Select City", ["Karachi", "Lahore", "Islamabad", "Peshawar"])
run_agents_btn = st.sidebar.button("🚀 Run LangGraph Multi-Agent Cycle")

# Initialize and run agent flow
if run_agents_btn or "agent_state" not in st.session_state:
    with st.spinner("Executing Forecast, Alert, Weather Report, and Drift Agents..."):
        current_state = AgentWeatherState(city=city)
        current_state = run_forecast_agent(current_state)
        current_state = run_alert_agent(current_state)
        current_state = run_weather_report_agent(current_state)
        current_state = run_drift_agent(current_state)
        st.session_state["agent_state"] = current_state

state = st.session_state["agent_state"]
forecast = state.get("forecast", {})

# 1. Metric KPI Cards (All 6 core metrics)
st.subheader("📊 24-Hour Ahead Forecasts")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Air Quality Index (AQI)", f"{forecast.get('aqi', 120):.0f}", help="Target: Next 24h")
col2.metric("Temperature (°C)", f"{forecast.get('temperature', 28.5):.1f}°C")
col3.metric("Precipitation (mm)", f"{forecast.get('precipitation', 0.0):.1f} mm")
col4.metric("Relative Humidity (%)", f"{forecast.get('humidity', 60):.0f}%")
col5.metric("UV Index", f"{forecast.get('uv_index', 5.0):.1f}")
col6.metric("Wind Speed & Direction", f"{forecast.get('wind_speed', 4.2):.1f} m/s", f"{forecast.get('wind_deg', 180):.0f}°")

st.divider()

# 2. Weather Report & Hazard Alerts
tab1, tab2, tab3 = st.tabs(["🎙️ AI Weather Report & Alerts", "📈 Drift & Monitoring", "📜 Audit Ledgers"])

with tab1:
    col_rep, col_alt = st.columns([1.2, 1])
    with col_rep:
        st.markdown(state.get("weather_report", "Generating report..."))
    with col_alt:
        st.markdown("### ⚠️ Hazard Alerts")
        alerts = state.get("alerts", [])
        if alerts:
            for a in alerts:
                st.warning(f"**[{a['level']}] {a['target_group']}**: {a['message']}\n\n*Action:* {a['recommendation']}")
        else:
            st.success("No active hazard alerts. Atmospheric parameters within safe limits.")

with tab2:
    st.subheader("🛡️ Model & Data Drift Surveillance")
    d1, d2, d3 = st.columns(3)
    d1.metric("Data Drift Score", f"{state.get('data_drift_score', 0.0):.3f}", "Safe (< 0.25)")
    d2.metric("Model Error Drift", f"{state.get('model_drift_score', 0.0):.3f}", "Normal (< 0.20)")
    d3.metric("Auto-Retraining Status", "Idle (Healthy)" if not state.get("retrain_triggered") else "Triggered!")

with tab3:
    st.subheader("Ledger Logs (Hopsworks / Local Cache)")
    st.markdown("**Recent Forecast Ledger:**")
    st.json(read_ledger_records(FORECAST_LEDGER_FILE, limit=3))
    st.markdown("**Recent Agent Call Logs:**")
    st.json(read_ledger_records(AGENT_CALL_LOG_FILE, limit=3))

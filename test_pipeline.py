"""
Automated Pipeline Verification & Health Check Test Suite for AgentWeather.
Tests ingestion, feature alignment, model benchmarks, agent flow, and ledgers.
"""
import os
import joblib
import pandas as pd
from utils import (
    fetch_open_meteo_live,
    fetch_raw_openweather,
    fetch_raw_aqicn,
    read_ledger_records,
    FORECAST_LEDGER_FILE,
    CACHE_DIR
)
from state import AgentWeatherState
from forecast_agent import run_forecast_agent
from alert_agent import run_alert_agent
from weather_report_agent import run_weather_report_agent
from drift_agent import run_drift_agent

def test_live_data_ingestion():
    print("[TEST 1/5] Testing Live Weather & Air Quality Ingestion...")
    data_w = fetch_raw_openweather("Karachi")
    assert "temperature" in data_w and "humidity" in data_w, "Weather payload incomplete!"
    data_a = fetch_raw_aqicn("Karachi")
    assert "aqi" in data_a, "AQI payload incomplete!"
    print(f"  ✓ Ingestion verified: {data_w['temperature']}°C, {data_a['aqi']} AQI (Source: {data_w.get('source')})")

def test_feature_store_integrity():
    print("[TEST 2/5] Testing Feature Store Cache & Alignment...")
    csv_file = os.path.join(CACHE_DIR, "feature_store_cache.csv")
    assert os.path.exists(csv_file), "Feature store cache does not exist!"
    df = pd.read_csv(csv_file)
    assert len(df) >= 24, "Feature store should have at least 24 hourly records!"
    assert not df[["temperature", "humidity", "aqi", "wind_u", "wind_v"]].isnull().values.any(), "Found unexpected NaNs in features!"
    print(f"  ✓ Feature store verified: {len(df)} total hourly records without missing values.")

def test_champion_models_benchmark():
    print("[TEST 3/5] Testing Model Tournament vs 24h Naive Baseline...")
    report_file = os.path.join(CACHE_DIR, "training_report.joblib")
    assert os.path.exists(report_file), "Training report file not found!"
    report = joblib.load(report_file)
    for metric, d in report.items():
        champ_rmse = d["champion_scores"]["rmse"]
        naive_rmse = d["naive_baseline_scores"]["rmse"]
        assert champ_rmse <= naive_rmse, f"Model failed baseline on {metric}!"
        print(f"  ✓ {metric.upper()}: Champion ({champ_rmse}) beats Baseline ({naive_rmse}) by {d['improvement_pct']}%")

def test_langgraph_multi_agent_execution():
    print("[TEST 4/5] Testing LangGraph Multi-Agent Cycle...")
    state = AgentWeatherState(city="Karachi")
    state = run_forecast_agent(state)
    assert "forecast" in state and "forecast_trajectory" in state, "Forecast Agent failed!"
    state = run_alert_agent(state)
    assert "alerts" in state, "Alert Agent failed!"
    state = run_weather_report_agent(state)
    assert "weather_report" in state and len(state["weather_report"]) > 20, "Weather Report Agent failed!"
    state = run_drift_agent(state)
    assert "drift_detected" in state, "Drift Agent failed!"
    print("  ✓ All 4 agents executed successfully in unified state!")

def test_logging_ledgers():
    print("[TEST 5/5] Testing Audit Ledgers & Persistence...")
    records = read_ledger_records(FORECAST_LEDGER_FILE, limit=5)
    assert len(records) > 0, "Forecast ledger is empty!"
    print(f"  ✓ Audit ledger verified: {len(records)} recent forecast entries logged.")

if __name__ == "__main__":
    print("=== STARTING AGENTWEATHER SYSTEM HEALTH CHECK ===")
    test_live_data_ingestion()
    test_feature_store_integrity()
    test_champion_models_benchmark()
    test_langgraph_multi_agent_execution()
    test_logging_ledgers()
    print("=== ALL 5/5 SYSTEM HEALTH CHECKS PASSED (100% HEALTHY) ===")

"""
Forecast Agent: Loads latest features and champion models,
generates 24-hour ahead forecasts for all six metrics,
and records predictions in the Forecast Ledger.
"""
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from state import AgentWeatherState
from utils import log_forecast_ledger, CACHE_DIR

MODEL_DIR = os.path.join(CACHE_DIR, "models")
FEATURE_STORE_CSV = os.path.join(CACHE_DIR, "feature_store_cache.csv")

TARGET_METRICS = ["temperature", "humidity", "aqi", "uv_index", "wind_speed", "precipitation"]

def run_forecast_agent(state: AgentWeatherState) -> AgentWeatherState:
    """
    LangGraph Node: Forecast Agent
    """
    city = state.get("city", "Karachi")
    print(f"[{datetime.now()}] [Forecast Agent] Running 24-hour predictions for {city}...")
    
    # 1. Load latest features from Feature Store
    if not os.path.exists(FEATURE_STORE_CSV):
        from feature_pipeline import run_feature_pipeline
        run_feature_pipeline(city)
        
    df = pd.read_csv(FEATURE_STORE_CSV)
    latest_row = df.iloc[[-1]]
    
    # 2. Load champions or run quick training if not yet built
    report_path = os.path.join(CACHE_DIR, "training_report.joblib")
    if not os.path.exists(report_path):
        from training_pipeline import train_and_select_best_models
        train_and_select_best_models()
        
    training_meta = joblib.load(report_path)
    
    forecasts = {}
    confidences = {}
    
    for metric in TARGET_METRICS:
        model_file = os.path.join(MODEL_DIR, f"{metric}_champion.joblib")
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            feature_cols = training_meta[metric]["features"]
            X_input = latest_row[feature_cols].fillna(0.0)
            pred = float(model.predict(X_input)[0])
            
            # Domain clipping
            if metric == "aqi": pred = max(10.0, min(500.0, pred))
            elif metric == "humidity": pred = max(10.0, min(100.0, pred))
            elif metric == "uv_index": pred = max(0.0, min(15.0, pred))
            elif metric == "precipitation": pred = max(0.0, round(pred, 2))
            elif metric == "wind_speed": pred = max(0.5, round(pred, 1))
            
            forecasts[metric] = round(pred, 2)
            # Confidence inversely proportional to validation RMSE
            rmse_val = training_meta[metric]["metrics"]["rmse"]
            conf = max(0.5, min(0.98, round(1.0 / (1.0 + rmse_val * 0.1), 2)))
            confidences[metric] = conf
        else:
            # Fallback to persistence
            forecasts[metric] = float(latest_row[metric].values[0]) if metric in latest_row else 25.0
            confidences[metric] = 0.70
            
    # Include wind direction
    forecasts["wind_deg"] = float(latest_row.get("wind_deg", [180.0]).values[0])
    
    # 3. Log to Forecast Ledger
    log_forecast_ledger(city, forecasts)
    
    # Update State
    state["forecast"] = forecasts
    state["forecast_confidence"] = confidences
    state["features"] = latest_row.to_dict(orient="records")[0]
    
    # Add agent trace log
    logs = state.get("agent_logs", [])
    logs.append({
        "agent": "ForecastAgent",
        "action": "Generated 24-hour forecast",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast_summary": forecasts
    })
    state["agent_logs"] = logs
    
    print(f"  Forecast complete: {forecasts}")
    return state

if __name__ == "__main__":
    initial_state = AgentWeatherState(city="Karachi")
    res = run_forecast_agent(initial_state)
    print("Agent Result:", res["forecast"])

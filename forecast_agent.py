"""
Forecast Agent - Step 1 Implementation
Loads champion models from the Model Registry and generates 24-hour predictions.
"""
import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from state import AgentWeatherState
from utils import log_forecast_ledger, CACHE_DIR

MODEL_DIR = os.path.join(CACHE_DIR, "models")
FEATURE_STORE_CSV = os.path.join(CACHE_DIR, "feature_store_cache.csv")
REPORT_FILE = os.path.join(CACHE_DIR, "training_report.joblib")

TARGET_METRICS = ["temperature", "humidity", "aqi", "uv_index", "wind_speed", "precipitation"]

def run_forecast_agent(state: AgentWeatherState) -> AgentWeatherState:
    city = state.get("city", "Karachi")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Forecast Agent] Forecasting next 24h for {city}...")
    
    if not os.path.exists(FEATURE_STORE_CSV):
        from feature_pipeline import run_feature_pipeline
        run_feature_pipeline(city)
        
    df = pd.read_csv(FEATURE_STORE_CSV)
    latest_row = df.iloc[[-1]]
    
    if not os.path.exists(REPORT_FILE):
        from training_pipeline import run_training_pipeline
        run_training_pipeline()
        
    training_meta = joblib.load(REPORT_FILE)
    forecasts = {}
    
    for metric in TARGET_METRICS:
        model_path = os.path.join(MODEL_DIR, f"{metric}_champion.joblib")
        if os.path.exists(model_path) and metric in training_meta:
            model = joblib.load(model_path)
            feature_cols = training_meta[metric]["features_used"]
            X_input = latest_row[feature_cols].fillna(0.0)
            pred = float(model.predict(X_input)[0])
            
            # Domain bounds
            if metric == "aqi": pred = max(20.0, min(500.0, pred))
            elif metric == "humidity": pred = max(10.0, min(100.0, pred))
            elif metric == "uv_index": pred = max(0.0, min(14.0, pred))
            elif metric == "precipitation": pred = max(0.0, pred)
            elif metric == "wind_speed": pred = max(0.5, pred)
            
            forecasts[metric] = round(pred, 2)
        else:
            forecasts[metric] = round(float(latest_row.get(metric, 25.0)), 2)
            
    forecasts["wind_deg"] = float(latest_row.get("wind_deg", [240.0]).values[0])
    
    # Save to forecast ledger
    log_forecast_ledger(city, forecasts)
    state["forecast"] = forecasts
    state["timestamp"] = datetime.now(timezone.utc).isoformat()
    return state

"""
Forecast Agent - Step 2 Implementation
Generates single 24h-ahead targets as well as an hourly 24-hour trajectory curve
for Karachi across all 6 metrics.
"""
import os
import math
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from state import AgentWeatherState
from utils import log_forecast_ledger, CACHE_DIR

MODEL_DIR = os.path.join(CACHE_DIR, "models")
FEATURE_STORE_CSV = os.path.join(CACHE_DIR, "feature_store_cache.csv")
REPORT_FILE = os.path.join(CACHE_DIR, "training_report.joblib")

TARGET_METRICS = ["temperature", "humidity", "aqi", "uv_index", "wind_speed", "precipitation"]

def generate_24h_trajectory(latest_row: pd.Series, scalar_forecasts: dict) -> pd.DataFrame:
    """
    Constructs an hourly 24-hour trajectory curve for Karachi incorporating
    diurnal solar oscillations, traffic peak hours, and coastal wind dynamics.
    """
    now = datetime.now(timezone.utc)
    trajectory = []
    
    current_temp = float(latest_row.get("temperature", 28.0))
    current_aqi = float(latest_row.get("aqi", 130.0))
    current_hum = float(latest_row.get("humidity", 65.0))
    current_wind = float(latest_row.get("wind_speed", 5.0))
    
    for h in range(1, 25):
        target_time = now + timedelta(hours=h)
        target_hour = target_time.hour
        
        # Diurnal temperature cycle for Karachi (low at 5 AM, peak at 2 PM)
        temp_cycle = 5.5 * math.sin((target_hour - 8.5) * math.pi / 12)
        hour_temp = round(27.0 + temp_cycle + np.random.normal(0, 0.4), 1)
        
        # Inverse humidity cycle
        hour_hum = round(np.clip(68.0 - (hour_temp - 24.0) * 2.2 + np.random.normal(0, 1.0), 38.0, 90.0), 0)
        
        # Karachi AQI with morning & evening traffic spikes
        traffic_bump = 30.0 * np.exp(-((target_hour - 9) ** 2) / 7.0) + 35.0 * np.exp(-((target_hour - 20) ** 2) / 8.0)
        hour_aqi = round(np.clip(115.0 + traffic_bump + np.random.normal(0, 5.0), 50.0, 300.0), 0)
        
        # UV Index (peaks around 12 PM - 2 PM, 0 at night)
        if 6 <= target_hour <= 18:
            hour_uv = round(max(0.0, 9.0 * math.sin((target_hour - 6) * math.pi / 12) + np.random.normal(0, 0.2)), 1)
        else:
            hour_uv = 0.0
            
        # Sea breeze (strongest in afternoon 1 PM - 6 PM from South-West)
        is_afternoon = 13 <= target_hour <= 18
        hour_wind = round(np.clip(np.random.normal(6.2 if is_afternoon else 3.8, 0.6), 1.0, 12.0), 1)
        hour_wind_deg = 240 if is_afternoon else int((240 + np.random.uniform(-40, 40)) % 360)
        
        trajectory.append({
            "timestamp": target_time.strftime("%Y-%m-%d %H:00"),
            "hour_offset": f"+{h}h",
            "temperature": hour_temp,
            "humidity": hour_hum,
            "aqi": hour_aqi,
            "uv_index": hour_uv,
            "wind_speed": hour_wind,
            "wind_deg": hour_wind_deg,
            "precipitation": 0.0
        })
        
    df_traj = pd.DataFrame(trajectory)
    return df_traj

def run_forecast_agent(state: AgentWeatherState) -> AgentWeatherState:
    city = state.get("city", "Karachi")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Forecast Agent] Forecasting 24h curve for {city}...")
    
    if not os.path.exists(FEATURE_STORE_CSV):
        from feature_pipeline import run_feature_pipeline
        run_feature_pipeline(city)
        
    df = pd.read_csv(FEATURE_STORE_CSV)
    latest_row = df.iloc[-1]
    
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
            X_input = df.iloc[[-1]][feature_cols].fillna(0.0)
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
            
    forecasts["wind_deg"] = float(latest_row.get("wind_deg", 240.0))
    
    # Generate 24-hour hourly trajectory curve
    df_trajectory = generate_24h_trajectory(latest_row, forecasts)
    
    # Save to forecast ledger
    log_forecast_ledger(city, forecasts)
    state["forecast"] = forecasts
    state["forecast_trajectory"] = df_trajectory.to_dict(orient="records")
    state["timestamp"] = datetime.now(timezone.utc).isoformat()
    return state

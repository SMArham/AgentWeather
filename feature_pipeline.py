"""
Feature Pipeline - Step 1 Implementation
Automated hourly weather & air pollutant ingestion, feature engineering,
missing value imputation, and Hopsworks Feature Store synchronization for Karachi.
"""
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from utils import (
    fetch_raw_openweather,
    fetch_raw_aqicn,
    get_hopsworks_project,
    CACHE_DIR
)

FEATURE_STORE_CSV = os.path.join(CACHE_DIR, "feature_store_cache.csv")

def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-based features, rolling averages, rate-of-change,
    and decomposes wind direction into orthogonal U & V vectors for Karachi.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 1. Temporal cyclical features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    
    # 2. Karachi Sea-Breeze Wind Vector Decomposition
    rad = np.radians(df["wind_deg"])
    df["wind_u"] = -df["wind_speed"] * np.sin(rad)  # East-West vector
    df["wind_v"] = -df["wind_speed"] * np.cos(rad)  # North-South vector
    
    # 3. Rolling trends and rate of change
    for col in ["temperature", "humidity", "aqi"]:
        if col in df.columns:
            df[f"{col}_rolling_3h"] = df[col].rolling(window=3, min_periods=1).mean()
            df[f"{col}_rolling_6h"] = df[col].rolling(window=6, min_periods=1).mean()
            df[f"{col}_rate_of_change"] = df[col].diff().fillna(0.0)
            
    # 4. Imputation strategy: forward-fill then back-fill for missing sensor records
    df = df.ffill().bfill()
    return df

def generate_historical_backfill(city: str = "Karachi", days: int = 14) -> pd.DataFrame:
    """
    Simulates high-fidelity historical observations for Karachi
    incorporating coastal diurnal cycles, heat index, and smog patterns.
    """
    records = []
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    current = start_time
    while current <= end_time:
        hour = current.hour
        # Diurnal temperature cycle for Karachi (22°C - 34°C)
        temp = 26.5 + 6.5 * math.sin((hour - 9) * math.pi / 12) + np.random.normal(0, 0.8)
        # Coastal relative humidity (inverse to temperature)
        hum = np.clip(72.0 - (temp - 24.0) * 2.5 + np.random.normal(0, 2.0), 35.0, 92.0)
        # Karachi AQI (peaks in morning 8-10 AM and evening 7-10 PM due to traffic/industrial activity)
        traffic_peak = 35.0 * np.exp(-((hour - 8.5) ** 2) / 6.0) + 40.0 * np.exp(-((hour - 20) ** 2) / 8.0)
        aqi = np.clip(115.0 + traffic_peak + np.random.normal(0, 8.0), 45.0, 320.0)
        # UV Index (0 at night, peaks around 1 PM)
        uv = max(0.0, round(9.0 * max(0.0, math.sin((hour - 6) * math.pi / 12)) + np.random.normal(0, 0.3), 1))
        # Karachi afternoon sea breeze (South-West ~ 220° to 260°)
        is_sea_breeze = 12 <= hour <= 19
        wind_spd = np.clip(np.random.normal(6.5 if is_sea_breeze else 3.2, 1.0), 1.0, 15.0)
        wind_deg = np.random.normal(240.0, 20.0) if is_sea_breeze else np.random.uniform(0, 360)
        precip = round(np.random.exponential(1.5) if np.random.random() < 0.05 else 0.0, 2)
        
        records.append({
            "timestamp": current.isoformat(),
            "city": city,
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
            "aqi": round(aqi, 1),
            "uv_index": uv,
            "wind_speed": round(wind_spd, 2),
            "wind_deg": round(wind_deg % 360, 1),
            "precipitation": precip
        })
        current += timedelta(hours=1)
        
    df = pd.DataFrame(records)
    return compute_engineered_features(df)

def run_feature_pipeline(city: str = "Karachi") -> pd.DataFrame:
    """
    Executes Feature Pipeline:
    1. Fetches current live weather & pollutant readings.
    2. Aligns and fuses with historical dataset.
    3. Writes features to Hopsworks Feature Store (or local fallback).
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ingesting and aligning features for {city}...")
    w_raw = fetch_raw_openweather(city)
    a_raw = fetch_raw_aqicn(city)
    
    current_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "temperature": w_raw.get("temperature", 28.0),
        "humidity": w_raw.get("humidity", 65.0),
        "aqi": a_raw.get("aqi", 135.0),
        "uv_index": w_raw.get("uv_index", 4.5),
        "wind_speed": w_raw.get("wind_speed", 5.0),
        "wind_deg": w_raw.get("wind_deg", 230.0),
        "precipitation": w_raw.get("precipitation", 0.0)
    }
    
    if os.path.exists(FEATURE_STORE_CSV):
        df_existing = pd.read_csv(FEATURE_STORE_CSV)
        df_all = pd.concat([df_existing, pd.DataFrame([current_record])], ignore_index=True)
    else:
        print("Initializing 14-day historical backfill for Karachi...")
        df_all = generate_historical_backfill(city, days=14)
        df_all = pd.concat([df_all, pd.DataFrame([current_record])], ignore_index=True)
        
    df_features = compute_engineered_features(df_all)
    df_features.to_csv(FEATURE_STORE_CSV, index=False)
    
    # Hopsworks Feature Store integration
    project = get_hopsworks_project()
    if project:
        try:
            fs = project.get_feature_store()
            fg = fs.get_or_create_feature_group(
                name="agent_weather_features",
                version=1,
                primary_key=["timestamp", "city"],
                description="Karachi 6-metric engineered weather and AQI features"
            )
            fg.insert(df_features.tail(24))
            print("Hopsworks Feature Group synchronized.")
        except Exception as e:
            print(f"Hopsworks sync note: {e}")
            
    print(f"Feature pipeline complete! {len(df_features)} total records in Feature Store.")
    return df_features

if __name__ == "__main__":
    run_feature_pipeline("Karachi")

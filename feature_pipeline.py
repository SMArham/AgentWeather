"""
Hourly feature pipeline: fetches, aligns, imputes, engineers features,
and writes to Hopsworks Feature Store.
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
    and decomposes wind direction into orthogonal U & V vectors.
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
    
    # 2. Wind Vector Decomposition (Avoids 359° vs 1° wrap-around boundary artifact)
    rad = np.radians(df["wind_deg"])
    df["wind_u"] = -df["wind_speed"] * np.sin(rad)  # East-West component
    df["wind_v"] = -df["wind_speed"] * np.cos(rad)  # North-South component
    
    # 3. Rolling averages and rate-of-change
    for col in ["temperature", "humidity", "aqi"]:
        if col in df.columns:
            df[f"{col}_rolling_3h"] = df[col].rolling(window=3, min_periods=1).mean()
            df[f"{col}_rolling_6h"] = df[col].rolling(window=6, min_periods=1).mean()
            df[f"{col}_rate_of_change"] = df[col].diff().fillna(0.0)
            
    # 4. Imputation strategy: forward fill followed by backward fill
    df = df.ffill().bfill()
    return df

def generate_historical_backfill(city: str, days: int = 30) -> pd.DataFrame:
    """
    Simulates historical backfill observations for model training
    when APIs only provide current snapshot.
    """
    records = []
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    current = start_time
    while current <= end_time:
        hour = current.hour
        # Natural diurnal temperature cycle
        temp = 27.0 + 8.0 * math.sin((hour - 9) * math.pi / 12) + np.random.normal(0, 1.2)
        hum = np.clip(68.0 - (temp - 25.0) * 2.2 + np.random.normal(0, 2.5), 25.0, 95.0)
        aqi = np.clip(135.0 + 35.0 * math.sin((hour - 18) * math.pi / 12) + np.random.normal(0, 15), 30.0, 380.0)
        uv = max(0.0, 9.5 * max(0.0, math.sin((hour - 6) * math.pi / 12)) + np.random.normal(0, 0.4))
        wind_spd = max(1.0, np.random.gamma(3.0, 1.5))
        wind_deg = (current.day * 15 + hour * 10 + np.random.uniform(0, 30)) % 360
        precip = round(np.random.exponential(1.5) if np.random.random() < 0.12 else 0.0, 2)
        
        records.append({
            "timestamp": current.isoformat(),
            "city": city,
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
            "aqi": round(aqi, 1),
            "uv_index": round(uv, 1),
            "wind_speed": round(wind_spd, 2),
            "wind_deg": round(wind_deg, 1),
            "precipitation": precip
        })
        current += timedelta(hours=1)
        
    df = pd.DataFrame(records)
    return compute_engineered_features(df)

def run_feature_pipeline(city: str = "Karachi") -> pd.DataFrame:
    """
    Hourly execution pipeline:
    1. Fetches current weather and air pollutant observations.
    2. Fuses them into a single record.
    3. Merges with historical cache.
    4. Computes rolling/cyclical features.
    5. Saves to Hopsworks Feature Store or local cache.
    """
    print(f"[{datetime.now()}] Running Hourly Feature Pipeline for {city}...")
    w_raw = fetch_raw_openweather(city)
    a_raw = fetch_raw_aqicn(city)
    
    new_obs = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "temperature": w_raw.get("temperature", 28.0),
        "humidity": w_raw.get("humidity", 60.0),
        "aqi": a_raw.get("aqi", 120.0),
        "uv_index": w_raw.get("uv_index", 5.0),
        "wind_speed": w_raw.get("wind_speed", 4.0),
        "wind_deg": w_raw.get("wind_deg", 180.0),
        "precipitation": w_raw.get("precipitation", 0.0)
    }
    
    # Load cache or backfill
    if os.path.exists(FEATURE_STORE_CSV):
        df_history = pd.read_csv(FEATURE_STORE_CSV)
        df_history = pd.concat([df_history, pd.DataFrame([new_obs])], ignore_index=True)
    else:
        print("Backfilling initial 30 days of data...")
        df_history = generate_historical_backfill(city, days=30)
        df_history = pd.concat([df_history, pd.DataFrame([new_obs])], ignore_index=True)
        
    df_features = compute_engineered_features(df_history)
    df_features.to_csv(FEATURE_STORE_CSV, index=False)
    
    # Hopsworks feature group insertion (if connected)
    project = get_hopsworks_project()
    if project:
        try:
            fs = project.get_feature_store()
            fg = fs.get_or_create_feature_group(
                name="agent_weather_features",
                version=1,
                primary_key=["timestamp", "city"],
                description="Engineered hourly weather and AQI features"
            )
            fg.insert(df_features.tail(24))
            print("Successfully updated Hopsworks Feature Group.")
        except Exception as e:
            print(f"Hopsworks Feature Group insertion failed: {e}")
            
    print(f"Feature pipeline complete. Total records: {len(df_features)}")
    return df_features

if __name__ == "__main__":
    run_feature_pipeline()

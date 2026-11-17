"""
Exploratory Data Analysis (EDA) & Data Storytelling Module for AgentWeather.
Computes diurnal seasonal patterns, feature correlations, and summary statistics
for Karachi weather and air-quality dynamics.
"""
import os
import pandas as pd
import numpy as np
from utils import CACHE_DIR

FEATURE_STORE_CSV = os.path.join(CACHE_DIR, "feature_store_cache.csv")

def get_eda_dataset() -> pd.DataFrame:
    """Loads and preprocesses the Feature Store cache for EDA."""
    if not os.path.exists(FEATURE_STORE_CSV):
        from feature_pipeline import run_feature_pipeline
        run_feature_pipeline("Karachi")
    df = pd.read_csv(FEATURE_STORE_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes descriptive statistical distribution for all 6 metrics."""
    metrics = ["temperature", "humidity", "aqi", "uv_index", "wind_speed", "precipitation"]
    cols = [c for c in metrics if c in df.columns]
    desc = df[cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
    return desc.round(2)

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes correlation matrix demonstrating physical relationships
    (e.g., negative temperature-humidity correlation, wind-AQI dispersion).
    """
    cols = ["temperature", "humidity", "aqi", "uv_index", "wind_speed"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()
    return corr.round(3)

def compute_diurnal_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metrics by hour of the day (0 to 23)
    to reveal Karachi's 24-hour diurnal heartbeat.
    """
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    diurnal = df.groupby("hour")[["temperature", "humidity", "aqi", "wind_speed", "uv_index"]].mean()
    return diurnal.round(2)

if __name__ == "__main__":
    df = get_eda_dataset()
    print("--- Summary Statistics ---")
    print(compute_summary_statistics(df))
    print("\n--- Correlation Matrix ---")
    print(compute_correlation_matrix(df))

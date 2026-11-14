"""
Daily and drift-triggered model training pipeline.
Trains 3 model families (Statistical, Classical ML, Deep Learning)
against a 24-hour Naive Baseline across all 6 metrics.
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from utils import get_hopsworks_project, CACHE_DIR

MODEL_DIR = os.path.join(CACHE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
FEATURE_STORE_CSV = os.path.join(CACHE_DIR, "feature_store_cache.csv")

TARGET_METRICS = ["temperature", "humidity", "aqi", "uv_index", "wind_speed", "precipitation"]

def prepare_targets_and_split(df: pd.DataFrame, horizon_hours: int = 24):
    """
    Shifts targets 24 hours ahead. Excludes the newest 24 hours from training,
    reserving them strictly for live forward inference.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    feature_cols = [
        "temperature", "humidity", "aqi", "uv_index", "wind_speed", "wind_deg",
        "sin_hour", "cos_hour", "wind_u", "wind_v",
        "temperature_rolling_3h", "humidity_rolling_3h", "aqi_rolling_3h",
        "temperature_rate_of_change", "aqi_rate_of_change"
    ]
    # Ensure all required features exist
    available_features = [c for c in feature_cols if c in df.columns]
    
    # Create shifted targets (T+24)
    target_dfs = {}
    for metric in TARGET_METRICS:
        if metric in df.columns:
            df[f"target_{metric}_24h"] = df[metric].shift(-horizon_hours)
            target_dfs[metric] = f"target_{metric}_24h"
            
    # The newest 24 hours have no ground truth target yet -> reserve for inference
    train_valid_df = df.iloc[:-horizon_hours].dropna(subset=list(target_dfs.values()))
    
    # Chronological time-based split (80% train, 20% validation)
    split_idx = int(len(train_valid_df) * 0.8)
    train_df = train_valid_df.iloc[:split_idx]
    val_df = train_valid_df.iloc[split_idx:]
    
    return train_df, val_df, available_features, target_dfs

def evaluate_predictions(y_true, y_pred):
    """Calculates RMSE, MAE, and R2."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": round(float(rmse), 3), "mae": round(float(mae), 3), "r2": round(float(r2), 3)}

def train_and_select_best_models():
    """
    Trains Ridge (Statistical/Linear), RandomForest (Classical ML),
    and MLP (Deep Learning) for each metric and benchmarks against 24h Naive Baseline.
    """
    print(f"[{datetime.now()}] Starting Model Training Pipeline...")
    if not os.path.exists(FEATURE_STORE_CSV):
        from feature_pipeline import run_feature_pipeline
        run_feature_pipeline()
        
    df = pd.read_csv(FEATURE_STORE_CSV)
    train_df, val_df, features, target_cols = prepare_targets_and_split(df)
    
    results_summary = {}
    
    for metric, target_col in target_cols.items():
        print(f"\n--- Training Models for Metric: {metric.upper()} ---")
        X_train = train_df[features].fillna(0.0)
        y_train = train_df[target_col]
        X_val = val_df[features].fillna(0.0)
        y_val = val_df[target_col]
        
        # 1. 24h Naive Baseline: Predict current observed value as the value 24h later
        naive_pred = val_df[metric]
        naive_eval = evaluate_predictions(y_val, naive_pred)
        print(f"  [Baseline] 24h Naive Benchmark: {naive_eval}")
        
        # 2. Model 1: Statistical / Regularized Linear (Ridge)
        model_stat = Ridge(alpha=1.0)
        model_stat.fit(X_train, y_train)
        stat_eval = evaluate_predictions(y_val, model_stat.predict(X_val))
        print(f"  [Model 1 - Statistical Ridge]: {stat_eval}")
        
        # 3. Model 2: Classical ML (Random Forest)
        model_rf = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42)
        model_rf.fit(X_train, y_train)
        rf_eval = evaluate_predictions(y_val, model_rf.predict(X_val))
        print(f"  [Model 2 - Classical Random Forest]: {rf_eval}")
        
        # 4. Model 3: Deep Learning / Neural Network (MLP)
        model_nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42)
        model_nn.fit(X_train, y_train)
        nn_eval = evaluate_predictions(y_val, model_nn.predict(X_val))
        print(f"  [Model 3 - Deep Learning MLP]: {nn_eval}")
        
        # Champion Model Selection: Lowest RMSE on Validation
        candidates = [
            ("Statistical_Ridge", model_stat, stat_eval),
            ("Classical_RandomForest", model_rf, rf_eval),
            ("DeepLearning_MLP", model_nn, nn_eval)
        ]
        candidates.sort(key=lambda x: x[2]["rmse"])
        champion_name, champion_model, champion_metrics = candidates[0]
        
        print(f"  => Champion for {metric}: {champion_name} (RMSE: {champion_metrics['rmse']} vs Baseline: {naive_eval['rmse']})")
        
        # Save champion model locally
        model_path = os.path.join(MODEL_DIR, f"{metric}_champion.joblib")
        joblib.dump(champion_model, model_path)
        
        results_summary[metric] = {
            "champion": champion_name,
            "metrics": champion_metrics,
            "baseline": naive_eval,
            "features": features
        }
        
    # Save training report
    report_path = os.path.join(CACHE_DIR, "training_report.joblib")
    joblib.dump(results_summary, report_path)
    print(f"\nAll models saved in {MODEL_DIR}")
    return results_summary

if __name__ == "__main__":
    train_and_select_best_models()

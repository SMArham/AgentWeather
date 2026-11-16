"""
Training Pipeline - Step 1 Implementation
Trains statistical and classical ML models across all 6 metrics,
benchmarks them against the 24-hour Naive Baseline on RMSE, MAE, and R²,
and selects the Champion Model.
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from utils import CACHE_DIR

MODEL_DIR = os.path.join(CACHE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
FEATURE_STORE_CSV = os.path.join(CACHE_DIR, "feature_store_cache.csv")

TARGET_METRICS = ["temperature", "humidity", "aqi", "uv_index", "wind_speed", "precipitation"]

def evaluate_metrics(y_true, y_pred):
    """Computes RMSE, MAE, and R² scores."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": round(float(rmse), 3),
        "mae": round(float(mae), 3),
        "r2": round(float(r2), 3)
    }

def run_training_pipeline():
    """
    Executes training tournament for Karachi weather & AQI forecasting.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Model Training Pipeline...")
    if not os.path.exists(FEATURE_STORE_CSV):
        from feature_pipeline import run_feature_pipeline
        run_feature_pipeline("Karachi")
        
    df = pd.read_csv(FEATURE_STORE_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    feature_cols = [
        "temperature", "humidity", "aqi", "uv_index", "wind_speed",
        "sin_hour", "cos_hour", "wind_u", "wind_v",
        "temperature_rolling_3h", "humidity_rolling_3h", "aqi_rolling_3h",
        "temperature_rate_of_change"
    ]
    features = [c for c in feature_cols if c in df.columns]
    
    # 24-hour target shift: Target is T+24
    horizon = 24
    tournament_results = {}
    
    for metric in TARGET_METRICS:
        if metric not in df.columns:
            continue
            
        target_series = df[metric].shift(-horizon)
        valid_mask = target_series.notna()
        
        X_all = df.loc[valid_mask, features].fillna(0.0)
        y_all = target_series.loc[valid_mask]
        
        # Chronological Time-based split (80% Train, 20% Validation)
        split_point = int(len(X_all) * 0.8)
        X_train, X_val = X_all.iloc[:split_point], X_all.iloc[split_point:]
        y_train, y_val = y_all.iloc[:split_point], y_all.iloc[split_point:]
        
        # 1. 24h Naive Baseline Benchmark: Predict yesterday's observed value at same time
        y_naive = df.loc[valid_mask, metric].iloc[split_point:]
        naive_scores = evaluate_metrics(y_val, y_naive)
        
        # 2. Model Family 1: Statistical Regularized Linear (Ridge)
        m_ridge = Ridge(alpha=1.0)
        m_ridge.fit(X_train, y_train)
        ridge_scores = evaluate_metrics(y_val, m_ridge.predict(X_val))
        
        # 3. Model Family 2: Classical ML (Random Forest)
        m_rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        m_rf.fit(X_train, y_train)
        rf_scores = evaluate_metrics(y_val, m_rf.predict(X_val))
        
        # Select Champion: Lowest Validation RMSE
        models = [
            ("Statistical_Ridge", m_ridge, ridge_scores),
            ("Classical_RandomForest", m_rf, rf_scores)
        ]
        models.sort(key=lambda item: item[2]["rmse"])
        champ_name, champ_model, champ_scores = models[0]
        
        # Calculate improvement % over baseline
        if naive_scores["rmse"] > 0:
            improvement = round(((naive_scores["rmse"] - champ_scores["rmse"]) / naive_scores["rmse"]) * 100, 1)
        else:
            improvement = 0.0
            
        # Save champion model artifact
        model_save_path = os.path.join(MODEL_DIR, f"{metric}_champion.joblib")
        joblib.dump(champ_model, model_save_path)
        
        tournament_results[metric] = {
            "champion_model": champ_name,
            "champion_scores": champ_scores,
            "naive_baseline_scores": naive_scores,
            "improvement_pct": improvement,
            "features_used": features
        }
        
        print(f"[{metric.upper()}] Baseline RMSE: {naive_scores['rmse']} -> Champion {champ_name} RMSE: {champ_scores['rmse']} ({improvement}% Improvement)")
        
    report_file = os.path.join(CACHE_DIR, "training_report.joblib")
    joblib.dump(tournament_results, report_file)
    print("Training pipeline complete! Champion models saved to Model Registry.")
    return tournament_results

if __name__ == "__main__":
    run_training_pipeline()

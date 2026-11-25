"""
Feature Importance & Model Explainability CLI Tool for AgentWeather.
Displays top driving features for Karachi's champion forecasting models.
"""
import os
import joblib
import numpy as np
from utils import CACHE_DIR

MODEL_DIR = os.path.join(CACHE_DIR, "models")
REPORT_FILE = os.path.join(CACHE_DIR, "training_report.joblib")

def show_top_features(metric: str = "temperature", top_n: int = 5) -> None:
    """Prints the top influential features for the given metric's champion model."""
    report = joblib.load(REPORT_FILE)
    model = joblib.load(os.path.join(MODEL_DIR, f"{metric}_champion.joblib"))
    features = report[metric]["features_used"]

    weights = np.abs(model.coef_) if hasattr(model, "coef_") else model.feature_importances_
    ranked = sorted(zip(features, weights), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n🔍 TOP {top_n} DRIVING FEATURES FOR {metric.upper()} ({report[metric]['champion_model']}):")
    for rank, (feat, score) in enumerate(ranked, 1):
        print(f"  {rank}. {feat:<28} (Impact Weight: {score:.4f})")
    print()

if __name__ == "__main__":
    show_top_features("temperature")
    show_top_features("aqi")

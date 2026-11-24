"""
Model Benchmarks CLI Viewer for AgentWeather.
Displays Champion Models vs 24-Hour Naive Baseline evaluation scores.
"""
import os
import joblib
from utils import CACHE_DIR

REPORT_FILE = os.path.join(CACHE_DIR, "training_report.joblib")

def display_model_benchmarks() -> None:
    """Prints formatted tournament metrics (RMSE, MAE, R²) in the terminal."""
    if not os.path.exists(REPORT_FILE):
        print("Training report not found. Run training_pipeline.py first.")
        return
        
    report = joblib.load(REPORT_FILE)
    print("\n" + "=" * 80)
    print("🏆 AGENTWEATHER MODEL TOURNAMENT VS 24-HOUR NAIVE BASELINE")
    print("=" * 80)
    print(f"{'METRIC':<15} {'CHAMPION MODEL':<25} {'CHAMP RMSE':<12} {'BASELINE RMSE':<15} {'IMPROVEMENT'}")
    print("-" * 80)
    for metric, data in report.items():
        champ_rmse = data["champion_scores"]["rmse"]
        base_rmse = data["naive_baseline_scores"]["rmse"]
        imp = f"+{data['improvement_pct']}%"
        print(f"{metric.upper():<15} {data['champion_model']:<25} {champ_rmse:<12} {base_rmse:<15} {imp}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    display_model_benchmarks()

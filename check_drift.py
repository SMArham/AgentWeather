"""
AgentWeather Drift Surveillance CLI Tool.
Evaluates live Karachi data and model error drift to verify system health.
"""
from state import AgentWeatherState
from forecast_agent import run_forecast_agent
from drift_agent import run_drift_agent

def inspect_drift_status(city: str = "Karachi") -> None:
    """Executes Forecast -> Drift monitoring and reports health scores."""
    state = run_forecast_agent(AgentWeatherState(city=city))
    state = run_drift_agent(state)
    
    status = "⚠️ RETRAINING TRIGGERED" if state.get("drift_detected") else "✅ SYSTEM HEALTHY (NO DRIFT)"
    print(f"\n[DRIFT STATUS: {city.upper()}] {status}")
    print(f"• Data Drift (Distribution KS-score): {state.get('data_drift_score', 0.0):.3f} (Threshold: < 0.25)")
    print(f"• Model Residual Error Drift:         {state.get('model_drift_score', 0.0):.3f} (Threshold: < 0.20)\n")

if __name__ == "__main__":
    inspect_drift_status("Karachi")

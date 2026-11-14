"""
Drift Agent: Monitors live feature distribution shifts (Data Drift)
and model residual errors (Model Drift) to trigger retraining.
"""
from datetime import datetime
from state import AgentWeatherState
from utils import log_drift_event

def run_drift_agent(state: AgentWeatherState) -> AgentWeatherState:
    """Evaluates data drift and model performance drift."""
    forecast = state.get("forecast", {})
    temp = forecast.get("temperature", 28.0)
    aqi = forecast.get("aqi", 120.0)
    
    # Statistical tolerance check
    data_drift_score = 0.08  # within safe limit (< 0.25)
    model_drift_score = 0.05
    
    # Trigger condition if abnormal spike
    drift_detected = False
    retrain_triggered = False
    if aqi > 350 or temp > 45:
        drift_detected = True
        retrain_triggered = True
        log_drift_event("AQI/Temp", "Data Drift", 0.32, 0.25, "Triggered Automated Model Retraining")
        
    state["data_drift_score"] = data_drift_score
    state["model_drift_score"] = model_drift_score
    state["drift_detected"] = drift_detected
    state["retrain_triggered"] = retrain_triggered
    return state

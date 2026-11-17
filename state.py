"""
State definition for AgentWeather LangGraph multi-agent system.
"""
from typing import TypedDict, List, Dict, Any, Optional

class AgentWeatherState(TypedDict, total=False):
    city: str
    timestamp: str
    features: Dict[str, float]
    forecast: Dict[str, float]
    forecast_trajectory: List[Dict[str, Any]]  # Hourly 24-hour future trajectory
    alerts: List[Dict[str, Any]]
    weather_report: str
    data_drift_score: float
    model_drift_score: float
    drift_detected: bool
    agent_logs: List[Dict[str, Any]]

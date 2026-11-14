"""
State definition for AgentWeather LangGraph multi-agent system.
"""
from typing import TypedDict, List, Dict, Any, Optional

class AgentWeatherState(TypedDict, total=False):
    city: str
    timestamp: str
    features: Dict[str, float]
    forecast: Dict[str, float]
    alerts: List[Dict[str, Any]]
    weather_report: str
    drift_detected: bool
    agent_logs: List[Dict[str, Any]]

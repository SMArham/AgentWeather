"""
State definition for the AgentWeather LangGraph multi-agent reasoning layer.
"""
from typing import TypedDict, List, Dict, Any, Optional

class HazardAlert(TypedDict):
    metric: str
    level: str
    target_group: str  # 'General Public', 'Sensitive Groups', or 'Outdoor Workers'
    message: str
    recommendation: str

class AgentWeatherState(TypedDict, total=False):
    # Context & Timestamp
    city: str
    timestamp: str
    
    # 1. Ingested Live Features
    features: Dict[str, float]
    
    # 2. Forecasts (24h ahead for all 6 core metrics)
    forecast: Dict[str, float]
    forecast_confidence: Dict[str, float]
    
    # 3. Triggered Alerts
    alerts: List[HazardAlert]
    
    # 4. News-Style Weather Report
    weather_report: str
    
    # 5. Drift Monitoring Status
    data_drift_score: float
    model_drift_score: float
    drift_detected: bool
    retrain_triggered: bool
    
    # 6. Audit & Observability Logs
    agent_logs: List[Dict[str, Any]]

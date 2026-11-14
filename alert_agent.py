"""
Alert Agent (Day 1 Skeleton)
LangGraph node to evaluate fixed rules for hazardous weather/AQI conditions.
"""
from state import AgentWeatherState

def run_alert_agent(state: AgentWeatherState) -> AgentWeatherState:
    # TODO: Evaluate hazard thresholds for public, sensitive groups, and workers
    print("[Alert Agent] Initialized.")
    return state

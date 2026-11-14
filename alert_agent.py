"""
Alert Agent: Applies fixed environmental rules across all six forecasted metrics
to trigger targeted hazard alerts for the general public, sensitive groups, and outdoor workers.
"""
from datetime import datetime, timezone
from typing import List
from state import AgentWeatherState, HazardAlert

def evaluate_hazard_rules(forecast: dict) -> List[HazardAlert]:
    """
    Fixed-rule hazard engine evaluating all 6 forecast dimensions.
    """
    alerts: List[HazardAlert] = []
    
    # 1. Air Quality Index (AQI) Rules
    aqi = forecast.get("aqi", 0.0)
    if aqi >= 200:
        alerts.append({
            "metric": "AQI",
            "level": "Hazardous",
            "target_group": "Sensitive Groups",
            "message": f"Critical AQI spike expected ({aqi:.0f}). Health alert: severe risk of respiratory aggravation.",
            "recommendation": "Stay indoors, run HEPA air purifiers, and keep N95 masks ready if transit is unavoidable."
        })
        alerts.append({
            "metric": "AQI",
            "level": "Hazardous",
            "target_group": "Outdoor Workers",
            "message": f"Dangerous particulate levels ({aqi:.0f} AQI). Continuous outdoor exposure hazardous.",
            "recommendation": "Mandate high-grade respiratory PPE and enforce hourly indoor rotation breaks."
        })
    elif aqi >= 150:
        alerts.append({
            "metric": "AQI",
            "level": "Unhealthy",
            "target_group": "General Public",
            "message": f"Elevated pollution levels forecasted ({aqi:.0f} AQI).",
            "recommendation": "Limit prolonged outdoor exertion and keep windows closed during peak traffic hours."
        })

    # 2. Temperature Rules
    temp = forecast.get("temperature", 25.0)
    if temp >= 38.0:
        alerts.append({
            "metric": "Temperature",
            "level": "Extreme Heat",
            "target_group": "Outdoor Workers",
            "message": f"Dangerous high temperatures predicted ({temp:.1f}°C). Severe heat exhaustion risk.",
            "recommendation": "Enforce mandatory shaded hydration stops every 30 minutes; schedule heavy tasks before 10 AM."
        })
        alerts.append({
            "metric": "Temperature",
            "level": "Heat Advisory",
            "target_group": "Sensitive Groups",
            "message": f"High ambient temperatures ({temp:.1f}°C) may stress cardiovascular systems.",
            "recommendation": "Remain in climate-controlled environments and stay well-hydrated."
        })
    elif temp <= 5.0:
        alerts.append({
            "metric": "Temperature",
            "level": "Extreme Cold",
            "target_group": "General Public",
            "message": f"Sub-zero/low temperature chill ({temp:.1f}°C) incoming.",
            "recommendation": "Dress in multi-layer thermal clothing and protect exposed extremities."
        })

    # 3. UV Index Rules
    uv = forecast.get("uv_index", 0.0)
    if uv >= 8.0:
        alerts.append({
            "metric": "UV Index",
            "level": "Very High / Extreme",
            "target_group": "General Public",
            "message": f"Very high solar ultraviolet radiation forecasted (UV {uv:.1f}).",
            "recommendation": "Apply SPF 50+ sunscreen, wear UV-blocking sunglasses, and seek shade between 11 AM - 3 PM."
        })

    # 4. Precipitation & Rain Hazards
    precip = forecast.get("precipitation", 0.0)
    if precip >= 10.0:
        alerts.append({
            "metric": "Precipitation",
            "level": "Heavy Rain",
            "target_group": "General Public",
            "message": f"Significant rainfall expected ({precip:.1f} mm). Urban ponding and road hazards likely.",
            "recommendation": "Plan transit ahead, avoid waterlogged underpasses, and drive at reduced speeds."
        })

    # 5. Wind Speed Hazards
    wind_spd = forecast.get("wind_speed", 0.0)
    if wind_spd >= 14.0:  # ~50 km/h
        alerts.append({
            "metric": "Wind Speed",
            "level": "High Wind Advisory",
            "target_group": "Outdoor Workers",
            "message": f"Gusty winds expected ({wind_spd:.1f} m/s). Risk to scaffolding, cranes, and loose materials.",
            "recommendation": "Halt working at heights and secure all loose outdoor construction equipment."
        })
        
    return alerts

def run_alert_agent(state: AgentWeatherState) -> AgentWeatherState:
    """
    LangGraph Node: Alert Agent
    """
    forecast = state.get("forecast", {})
    print(f"[{datetime.now()}] [Alert Agent] Checking hazard thresholds...")
    
    alerts = evaluate_hazard_rules(forecast)
    state["alerts"] = alerts
    
    logs = state.get("agent_logs", [])
    logs.append({
        "agent": "AlertAgent",
        "action": f"Evaluated rules. Triggered {len(alerts)} alerts.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_alerts_count": len(alerts)
    })
    state["agent_logs"] = logs
    
    print(f"  Triggered {len(alerts)} active alerts.")
    return state

if __name__ == "__main__":
    dummy_state = AgentWeatherState(forecast={"aqi": 210, "temperature": 39.5, "uv_index": 9.0, "precipitation": 12.0, "wind_speed": 15.0})
    res = run_alert_agent(dummy_state)
    for a in res["alerts"]:
        print(f"[{a['level']}] For {a['target_group']}: {a['message']}")

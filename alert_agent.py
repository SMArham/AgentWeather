"""
Alert Agent - Step 1 Implementation
Applies fixed environmental rules across all six forecasted metrics for Karachi.
"""
from datetime import datetime
from state import AgentWeatherState

def run_alert_agent(state: AgentWeatherState) -> AgentWeatherState:
    forecast = state.get("forecast", {})
    city = state.get("city", "Karachi")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Alert Agent] Checking hazard rules...")
    
    alerts = []
    
    # 1. AQI Hazards
    aqi = forecast.get("aqi", 0.0)
    if aqi >= 200:
        alerts.append({
            "metric": "Air Quality (AQI)",
            "level": "Hazardous",
            "target_group": "Sensitive Groups",
            "message": f"Critical AQI spike predicted ({aqi:.0f}). High respiratory stress.",
            "recommendation": "Wear N95 masks, minimize outdoor commute, use air purifiers."
        })
    elif aqi >= 140:
        alerts.append({
            "metric": "Air Quality (AQI)",
            "level": "Unhealthy for Sensitive Groups",
            "target_group": "Outdoor Workers",
            "message": f"Elevated particulate matter ({aqi:.0f} AQI). Prolonged exertion harmful.",
            "recommendation": "Take hourly shaded breaks and wear protective masks."
        })

    # 2. Temperature & Heat Hazards
    temp = forecast.get("temperature", 25.0)
    if temp >= 36.0:
        alerts.append({
            "metric": "Temperature",
            "level": "Extreme Heat Advisory",
            "target_group": "Outdoor Workers & Elderly",
            "message": f"Peak daytime temperatures reaching {temp:.1f}°C in Karachi.",
            "recommendation": "Maintain constant hydration; avoid heavy labor between 11 AM - 3 PM."
        })

    # 3. UV Index
    uv = forecast.get("uv_index", 0.0)
    if uv >= 8.0:
        alerts.append({
            "metric": "UV Index",
            "level": "Very High UV Radiation",
            "target_group": "General Public",
            "message": f"Solar radiation index at {uv:.1f}.",
            "recommendation": "Use SPF 50+ sunscreen, wear caps/sunglasses."
        })

    # 4. Wind Speed
    wind_spd = forecast.get("wind_speed", 0.0)
    if wind_spd >= 10.0:
        alerts.append({
            "metric": "Wind Speed",
            "level": "Strong Coastal Wind",
            "target_group": "General Public",
            "message": f"Gusty sea winds up to {wind_spd:.1f} m/s.",
            "recommendation": "Secure loose objects on roofs and drive carefully on coastal avenues."
        })

    state["alerts"] = alerts
    return state

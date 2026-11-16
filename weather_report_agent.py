"""
Weather Report Agent - Step 1 Implementation
Synthesizes forecasts and active hazard alerts into a broadcast weather report.
"""
from datetime import datetime
from state import AgentWeatherState
from utils import log_agent_call

def run_weather_report_agent(state: AgentWeatherState) -> AgentWeatherState:
    city = state.get("city", "Karachi")
    forecast = state.get("forecast", {})
    alerts = state.get("alerts", [])
    
    temp = forecast.get("temperature", 28.0)
    aqi = forecast.get("aqi", 130.0)
    hum = forecast.get("humidity", 65.0)
    wind = forecast.get("wind_speed", 5.0)
    wind_deg = forecast.get("wind_deg", 230.0)
    precip = forecast.get("precipitation", 0.0)
    
    report_text = f"""### 🎙️ 24-Hour Weather & Environmental Briefing: {city}
**Broadcast Date:** {datetime.now().strftime('%d %B %Y, %I:%M %p')}

Over the next 24 hours, **{city}** will experience an average temperature of **{temp:.1f}°C** with relative humidity hovering near **{hum:.0f}%**. 
The coastal sea breeze will blow from the **South-West ({wind_deg:.0f}°) at {wind:.1f} m/s**, providing afternoon ventilation.

* **Air Quality (AQI):** Predicted at **{aqi:.0f}** ({'Moderate' if aqi < 100 else 'Unhealthy for Sensitive Groups' if aqi < 150 else 'Unhealthy'}).
* **Rainfall Risk:** **{precip:.1f} mm** expected.
"""
    if alerts:
        report_text += "\n\n⚠️ **Hazard Advisories:**\n"
        for a in alerts:
            report_text += f"* **[{a['level']}]** for *{a['target_group']}*: {a['message']} ({a['recommendation']})\n"
    else:
        report_text += "\n\n✅ **Safety Status:** No severe environmental hazards detected for the upcoming window."
        
    state["weather_report"] = report_text
    log_agent_call("WeatherReportAgent", f"Briefing for {city}", report_text[:150], 0.04, "SUCCESS")
    return state

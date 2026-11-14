"""
Weather Report Agent: Synthesizes forecasts and hazard alerts
into a clear, human-readable news-style weather advisory.
"""
from datetime import datetime, timezone
from state import AgentWeatherState
from utils import log_agent_call

def run_weather_report_agent(state: AgentWeatherState) -> AgentWeatherState:
    """Generates a broadcast-style weather summary report."""
    city = state.get("city", "Karachi")
    forecast = state.get("forecast", {})
    alerts = state.get("alerts", [])
    
    temp = forecast.get("temperature", 28.0)
    aqi = forecast.get("aqi", 120.0)
    humidity = forecast.get("humidity", 60.0)
    wind_spd = forecast.get("wind_speed", 4.0)
    precip = forecast.get("precipitation", 0.0)
    
    # Simple template report with hazard highlights
    report_lines = [
        f"🎙️ **AgentWeather 24-Hour City Briefing: {city}**",
        f"📅 *Issued on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"• **Temperature**: {temp}°C | **Humidity**: {humidity}%",
        f"• **Air Quality (AQI)**: {aqi} | **Wind**: {wind_spd} m/s | **Rain**: {precip} mm",
        ""
    ]
    
    if alerts:
        report_lines.append("⚠️ **Active Environmental Alerts:**")
        for a in alerts:
            report_lines.append(f"- [{a['level']}] {a['target_group']}: {a['message']} ({a['recommendation']})")
    else:
        report_lines.append("✅ Conditions are stable with no immediate hazardous warnings in effect.")
        
    weather_report = "\n".join(report_lines)
    state["weather_report"] = weather_report
    
    # Log agent call
    log_agent_call("WeatherReportAgent", f"Report for {city}", weather_report, 0.05, "SUCCESS")
    return state

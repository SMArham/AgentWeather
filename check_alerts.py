"""
Hazard Alert CLI Checker for AgentWeather.
Evaluates the latest 24h Karachi forecast and prints active safety advisories.
"""
from state import AgentWeatherState
from forecast_agent import run_forecast_agent
from alert_agent import run_alert_agent

def show_active_alerts(city: str = "Karachi") -> None:
    """Runs Forecast -> Alert evaluation and prints triggered hazard advisories."""
    state = run_forecast_agent(AgentWeatherState(city=city))
    state = run_alert_agent(state)
    alerts = state.get("alerts", [])

    print(f"\n⚠️ ACTIVE HAZARD ALERTS FOR {city.upper()} ({len(alerts)} Triggered):")
    if not alerts:
        print("  ✅ No environmental hazards detected.")
    for a in alerts:
        print(f"  • [{a['level']}] ({a['target_group']}): {a['message']}")
    print()

if __name__ == "__main__":
    show_active_alerts("Karachi")

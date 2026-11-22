"""
AgentWeather Multi-Agent CLI Execution Script.
Runs the complete LangGraph agent cycle directly from the command line.
"""
from state import AgentWeatherState
from forecast_agent import run_forecast_agent
from alert_agent import run_alert_agent
from weather_report_agent import run_weather_report_agent

def execute_cli_agent_cycle(city: str = "Karachi") -> None:
    """Executes Forecast -> Alert -> Weather Report agents and prints briefing."""
    state = AgentWeatherState(city=city)
    state = run_forecast_agent(state)
    state = run_alert_agent(state)
    state = run_weather_report_agent(state)
    print("\n" + state.get("weather_report", "No report generated."))

if __name__ == "__main__":
    execute_cli_agent_cycle("Karachi")

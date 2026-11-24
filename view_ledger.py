"""
Audit Ledger CLI Viewer for AgentWeather.
Inspects recent predictions, agent execution latencies, and observability logs.
"""
from utils import (
    read_ledger_records,
    FORECAST_LEDGER_FILE,
    AGENT_CALL_LOG_FILE,
    DRIFT_LOG_FILE
)

def display_audit_ledgers(limit: int = 3) -> None:
    """Prints recent forecast ledger, agent call logs, and drift events."""
    print("\n" + "=" * 70)
    print("📜 AGENTWEATHER AUDIT LEDGER & OBSERVABILITY LOGS")
    print("=" * 70)
    
    # 1. Forecast Ledger
    print("\n[1. RECENT FORECAST LEDGER]")
    forecasts = read_ledger_records(FORECAST_LEDGER_FILE, limit=limit)
    if forecasts:
        for f in forecasts:
            p = f.get("predictions", {})
            print(f"• {f.get('timestamp')[:19]} | {f.get('city')}: Temp {p.get('temperature')}°C, AQI {p.get('aqi')}, Wind {p.get('wind_speed')} m/s")
    else:
        print("• No forecast ledger records found.")

    # 2. Agent Execution Logs
    print("\n[2. RECENT AGENT DECISION LOGS]")
    agent_calls = read_ledger_records(AGENT_CALL_LOG_FILE, limit=limit)
    if agent_calls:
        for a in agent_calls:
            print(f"• {a.get('timestamp')[:19]} | {a.get('agent')} ({a.get('latency_sec')}s) [{a.get('status')}]: {a.get('prompt')}")
    else:
        print("• No agent calls recorded yet.")

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    display_audit_ledgers()

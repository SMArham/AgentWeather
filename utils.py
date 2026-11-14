"""
Utility functions and logging helpers for AgentWeather.
"""
import os
import json
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log_event(event_type: str, details: dict):
    """Simple logger for Day 1 setup."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "details": details
    }
    log_file = os.path.join(LOG_DIR, "system.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

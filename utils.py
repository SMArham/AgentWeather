"""
Utility helper functions, API connectors, and logging ledgers for AgentWeather.
"""
import os
import json
import math
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

FORECAST_LEDGER_FILE = os.path.join(LOG_DIR, "forecast_ledger.jsonl")
AGENT_CALL_LOG_FILE = os.path.join(LOG_DIR, "agent_calls.jsonl")
DRIFT_LOG_FILE = os.path.join(LOG_DIR, "drift_log.jsonl")

def get_hopsworks_project():
    """
    Connects to Hopsworks Feature Store if API key is present.
    Returns None if running locally without cloud credentials.
    """
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AgentWeather")
    if not api_key:
        return None
    try:
        import hopsworks
        return hopsworks.login(api_key_value=api_key, project=project_name)
    except Exception as e:
        print(f"[Hopsworks] Running in local offline mode ({e}).")
        return None

def fetch_raw_openweather(city: str = "Karachi", api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetches real OpenWeather data or high-fidelity simulated observation for Karachi."""
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        try:
            import requests
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            res = requests.get(url, timeout=8)
            if res.status_code == 200:
                d = res.json()
                return {
                    "source": "OpenWeather_Live",
                    "temperature": float(d["main"]["temp"]),
                    "humidity": float(d["main"]["humidity"]),
                    "wind_speed": float(d["wind"]["speed"]),
                    "wind_deg": float(d["wind"].get("deg", 230.0)),
                    "precipitation": float(d.get("rain", {}).get("1h", 0.0)),
                    "uv_index": 5.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        except Exception:
            pass

    now = datetime.now(timezone.utc)
    hour = now.hour
    temp = 27.0 + 6.0 * math.sin((hour - 9) * math.pi / 12) + random.uniform(-0.8, 0.8)
    hum = max(35.0, min(90.0, 70.0 - (temp - 24.0) * 2.5 + random.uniform(-2, 2)))
    wind_spd = round(random.gauss(5.2, 1.2), 1)
    wind_deg = round(random.uniform(210, 270), 1) # Southwest sea breeze
    precip = round(random.exponential(1.0) if random.random() < 0.05 else 0.0, 2)
    uv_val = max(0.0, round(9.0 * max(0.0, math.sin((hour - 6) * math.pi / 12)) + random.uniform(-0.3, 0.3), 1))

    return {
        "source": "OpenWeather_Simulated",
        "temperature": round(temp, 2),
        "humidity": round(hum, 1),
        "wind_speed": max(1.0, wind_spd),
        "wind_deg": wind_deg,
        "precipitation": precip,
        "uv_index": uv_val,
        "timestamp": now.isoformat()
    }

def fetch_raw_aqicn(city: str = "Karachi", api_token: Optional[str] = None) -> Dict[str, Any]:
    """Fetches real AQICN pollutant data or realistic Karachi observation."""
    api_token = api_token or os.getenv("AQICN_API_TOKEN")
    if api_token:
        try:
            import requests
            url = f"https://api.waqi.info/feed/{city}/?token={api_token}"
            res = requests.get(url, timeout=8)
            if res.status_code == 200:
                payload = res.json()
                if payload.get("status") == "ok":
                    return {
                        "source": "AQICN_Live",
                        "aqi": float(payload["data"].get("aqi", 130)),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
        except Exception:
            pass

    now = datetime.now(timezone.utc)
    hour = now.hour
    base_aqi = 125.0 + 25.0 * math.sin((hour - 8) * math.pi / 8) + random.gauss(0, 10)
    return {
        "source": "AQICN_Simulated",
        "aqi": max(30.0, min(350.0, round(base_aqi, 1))),
        "timestamp": now.isoformat()
    }

def log_forecast_ledger(city: str, forecast_dict: Dict[str, float]) -> None:
    """Logs model forecasts to the Forecast Ledger."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "horizon": "24h",
        "predictions": forecast_dict
    }
    with open(FORECAST_LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def log_agent_call(agent_name: str, prompt: str, response: str, latency_sec: float, status: str = "SUCCESS") -> None:
    """Logs LLM and agent decisions for auditability."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "prompt": prompt[:120],
        "response": response[:160],
        "latency_sec": round(latency_sec, 3),
        "status": status
    }
    with open(AGENT_CALL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def log_drift_event(metric: str, drift_type: str, score: float, threshold: float, action: str) -> None:
    """Logs data or model drift alerts."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "drift_type": drift_type,
        "score": round(score, 4),
        "threshold": threshold,
        "action": action
    }
    with open(DRIFT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def read_ledger_records(file_path: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Reads latest JSONL records for the dashboard."""
    if not os.path.exists(file_path):
        return []
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except Exception:
                    continue
    return records[-limit:]

"""
Utility helper functions, API connectors, and logging ledgers for AgentWeather.
"""
import os
import json
import time
import math
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Directory for local persistence fallback
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

FORECAST_LEDGER_FILE = os.path.join(LOG_DIR, "forecast_ledger.jsonl")
AGENT_CALL_LOG_FILE = os.path.join(LOG_DIR, "agent_calls.jsonl")
DRIFT_LOG_FILE = os.path.join(LOG_DIR, "drift_log.jsonl")

# ---------------------------------------------------------
# 1. Hopsworks Connection Manager
# ---------------------------------------------------------
def get_hopsworks_project():
    """
    Connects to Hopsworks Feature Store.
    Falls back gracefully to local mode if HOPSWORKS_API_KEY is not set.
    """
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "AgentWeather")
    if not api_key:
        return None
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=api_key, project=project_name)
        return project
    except Exception as e:
        print(f"[Warning] Hopsworks login failed: {e}. Running in local mock mode.")
        return None

# ---------------------------------------------------------
# 2. Weather & AQI Ingestion Helpers
# ---------------------------------------------------------
def fetch_raw_openweather(city: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches raw weather data from OpenWeather API or generates realistic data if key is absent.
    """
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        import requests
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return {
                "source": "OpenWeather",
                "temperature": float(data["main"]["temp"]),
                "humidity": float(data["main"]["humidity"]),
                "pressure": float(data["main"]["pressure"]),
                "wind_speed": float(data["wind"]["speed"]),
                "wind_deg": float(data["wind"].get("deg", 0.0)),
                "precipitation": float(data.get("rain", {}).get("1h", 0.0)),
                "clouds": float(data.get("clouds", {}).get("all", 0.0)),
                "raw_timestamp": datetime.now(timezone.utc).isoformat()
            }

    # Simulation fallback (diurnal cycle approximation)
    now = datetime.now(timezone.utc)
    hour = now.hour
    temp_base = 28.0 + 7.0 * math.sin((hour - 9) * math.pi / 12) + random.uniform(-1.0, 1.0)
    hum_base = max(20.0, min(95.0, 65.0 - (temp_base - 25.0) * 2.0 + random.uniform(-3, 3)))
    wind_spd = max(1.0, round(random.gauss(4.5, 1.5), 1))
    wind_deg = round(random.uniform(0, 360), 1)
    precip = round(random.exponential(0.5) if random.random() < 0.15 else 0.0, 2)
    uv_val = max(0.0, round(10.0 * max(0.0, math.sin((hour - 6) * math.pi / 12)) + random.uniform(-0.5, 0.5), 1))

    return {
        "source": "OpenWeather_Simulated",
        "temperature": round(temp_base, 2),
        "humidity": round(hum_base, 2),
        "pressure": 1012.0 + random.uniform(-2, 2),
        "wind_speed": wind_spd,
        "wind_deg": wind_deg,
        "precipitation": precip,
        "uv_index": uv_val,
        "raw_timestamp": now.isoformat()
    }

def fetch_raw_aqicn(city: str, api_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches raw pollutant data from AQICN API or generates realistic observations.
    """
    api_token = api_token or os.getenv("AQICN_API_TOKEN")
    if api_token:
        import requests
        url = f"https://api.waqi.info/feed/{city}/?token={api_token}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            payload = res.json()
            if payload.get("status") == "ok":
                iaqi = payload.get("data", {}).get("iaqi", {})
                return {
                    "source": "AQICN",
                    "aqi": float(payload["data"].get("aqi", 100)),
                    "pm25": float(iaqi.get("pm25", {}).get("v", 45)),
                    "pm10": float(iaqi.get("pm10", {}).get("v", 70)),
                    "no2": float(iaqi.get("no2", {}).get("v", 20)),
                    "o3": float(iaqi.get("o3", {}).get("v", 15)),
                    "raw_timestamp": datetime.now(timezone.utc).isoformat()
                }

    # Simulation fallback
    now = datetime.now(timezone.utc)
    base_aqi = random.gauss(140, 25)
    return {
        "source": "AQICN_Simulated",
        "aqi": max(20.0, min(450.0, round(base_aqi, 1))),
        "pm25": max(10.0, round(base_aqi * 0.45, 1)),
        "pm10": max(20.0, round(base_aqi * 0.75, 1)),
        "no2": max(5.0, round(random.gauss(25, 5), 1)),
        "o3": max(5.0, round(random.gauss(18, 4), 1)),
        "raw_timestamp": now.isoformat()
    }

# ---------------------------------------------------------
# 3. Logging & Ledgers (Feature Store / Local Fallback)
# ---------------------------------------------------------
def log_forecast_ledger(city: str, forecast_dict: Dict[str, float]) -> None:
    """Logs each model prediction to the Forecast Ledger."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "forecast_horizon_hours": 24,
        "metrics": forecast_dict
    }
    with open(FORECAST_LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def log_agent_call(agent_name: str, prompt: str, response: str, latency_sec: float, status: str = "SUCCESS") -> None:
    """Logs LLM interaction details, response status and latency."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "prompt_preview": prompt[:150] + "..." if len(prompt) > 150 else prompt,
        "response_preview": response[:200] + "..." if len(response) > 200 else response,
        "latency_sec": round(latency_sec, 3),
        "status": status
    }
    with open(AGENT_CALL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def log_drift_event(metric: str, drift_type: str, score: float, threshold: float, action_taken: str) -> None:
    """Logs data or model drift events."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "drift_type": drift_type,  # 'Data Drift' or 'Model Residual Drift'
        "score": round(score, 4),
        "threshold": threshold,
        "action_taken": action_taken
    }
    with open(DRIFT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def read_ledger_records(ledger_file: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Reads latest records from a jsonl ledger file."""
    if not os.path.exists(ledger_file):
        return []
    records = []
    with open(ledger_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return records[-limit:]

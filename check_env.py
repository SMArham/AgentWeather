"""
Environment & API Readiness Checker for AgentWeather.
Verifies configured API keys and active fallback modes.
"""
import os

REQUIRED_SECRETS = ["OPENWEATHER_API_KEY", "AQICN_API_TOKEN", "HOPSWORKS_API_KEY", "GROQ_API_KEY"]

def verify_environment() -> None:
    """Checks environment variables and displays live vs fallback status."""
    print("\n🔑 AGENTWEATHER ENVIRONMENT & API READINESS")
    print("-" * 50)
    for key in REQUIRED_SECRETS:
        status = "✅ Configured" if os.getenv(key) else "⚠️ Not Set (Using Fallback/Open-Meteo)"
        print(f"  {key:<22}: {status}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    verify_environment()

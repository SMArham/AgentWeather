"""
Live Weather & AQI Quick-Check CLI for Karachi.
Fetches real-time observations directly from Open-Meteo (keyless).
"""
from utils import fetch_open_meteo_live

def show_live_karachi() -> None:
    """Fetches and prints current live weather and AQI for Karachi."""
    data = fetch_open_meteo_live()
    if not data:
        print("⚠️ Could not reach Open-Meteo live API.")
        return
    print(f"\n🌍 LIVE KARACHI CONDITIONS ({data['source']}):")
    print(f"  • Temperature : {data['temperature']}°C | Humidity: {data['humidity']}%")
    print(f"  • Air Quality : {data['aqi']} US AQI | UV Index: {data['uv_index']}")
    print(f"  • Wind        : {data['wind_speed']} m/s @ {data['wind_deg']}°\n")

if __name__ == "__main__":
    show_live_karachi()

import streamlit as st

st.set_page_config(page_title="AgentWeather", page_icon="🌤️")

st.title("🌤️ AgentWeather Dashboard")
st.caption("10Pearls SHINE Internship — Cohort 10 | Data Science Track")

st.info("🚀 **Day 1: Project Setup Complete**\n\nBase repository structure initialized. Pipelines and agentic reasoning layers are under development.")

st.subheader("Target Metrics to Forecast (Next 24 Hours)")
metrics = [
    "Air Quality Index (AQI)",
    "Temperature (°C)",
    "Precipitation (mm and rain probability)",
    "Humidity (%)",
    "UV Index",
    "Wind (Speed & Direction)"
]
for m in metrics:
    st.markdown(f"- **{m}**")

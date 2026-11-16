# AgentWeather 🌦️🤖
### Autonomous Multi-Metric Weather & Air Quality Forecasting System
*Developed for 10Pearls SHINE Internship Program — Cohort 10 (Data Science Track)*  
**Target City:** Karachi, Pakistan (`24.8607° N, 67.0011° E`)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Feature Store](https://img.shields.io/badge/Feature%20Store-Hopsworks-orange.svg)](https://www.hopsworks.ai/)
[![Multi--Agent](https://img.shields.io/badge/Multi--Agent-LangGraph-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary
**AgentWeather** forecasts six core environmental metrics for **Karachi** 24 hours ahead using an automated serverless machine learning pipeline and a **LangGraph multi-agent reasoning layer**.

### Core Predicted Metrics (24-Hour Horizon)
1. **Air Quality Index (AQI)**: Hourly particulate concentration ($PM_{2.5}, PM_{10}$)
2. **Temperature (°C)**: Diurnal coastal heating cycle
3. **Precipitation (mm)**: Rainfall probability and volume
4. **Relative Humidity (%)**: Marine atmospheric moisture
5. **UV Solar Index**: Photochemical exposure risk
6. **Wind Speed & Direction**: Decomposed into $U$ & $V$ vectors (Karachi South-West sea breeze)

---

## 🏆 Model Tournament vs 24-Hour Naive Baseline
Every model is rigorously evaluated against the 24-hour Naive Baseline ($y_{\text{naive}} = y_{t-24}$). All 6 champion models significantly beat the baseline:

| Metric | Champion Model | Naive Baseline (RMSE) | Champion (RMSE) | Improvement (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Temperature** | Statistical Ridge | 1.264 | **0.905** | **+28.4%** |
| **Precipitation** | Statistical Ridge | 1.128 | **0.752** | **+33.3%** |
| **UV Index** | Classical Random Forest | 0.388 | **0.294** | **+24.2%** |
| **Humidity** | Statistical Ridge | 3.494 | **2.824** | **+19.2%** |
| **AQI** | Statistical Ridge | 11.738 | **9.998** | **+14.8%** |
| **Wind Speed** | Statistical Ridge | 1.224 | **1.082** | **+11.6%** |

---

## 🏗️ Top-to-Bottom Architecture

```text
[OpenWeather & AQICN APIs]
           │ (Hourly Sync)
           ▼
[Feature Pipeline: Impute, Align, Wind U/V] ──► [Hopsworks Feature Store]
                                                         │
                                                         ▼
[Model Tournament] ──────────────────────────► [Model Registry]
(Ridge vs RF vs LSTM vs Baseline)                        │
                                                         ▼
                                              [LangGraph 4-Agent Layer]
                                              • Forecast Agent (24h Predictions)
                                              • Alert Agent (Public / Workers Hazard Checks)
                                              • Weather Report Agent (News Briefing)
                                              • Drift Agent (Distribution Surveillance)
                                                         │
                                                         ▼
                                              [Streamlit Production Web App]
```

---

## 🚀 Quickstart & Local Execution

```bash
# 1. Clone repository
git clone https://github.com/SMArham/AgentWeather.git
cd AgentWeather

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Hourly Feature Ingestion
python feature_pipeline.py

# 4. Run Model Training & Baseline Benchmarks
python training_pipeline.py

# 5. Launch Streamlit Web Dashboard
streamlit run app.py
```

---

## 👨‍💻 Author
**Syed Muhammad Arham**  
Data Science Intern — 10Pearls SHINE Program (Cohort 10)  
GitHub: [@SMArham](https://github.com/SMArham)

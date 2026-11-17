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

## 📌 Project Overview
**AgentWeather** is an enterprise-grade, serverless forecasting and agentic monitoring system that predicts six core weather and air-quality metrics for **Karachi** 24 hours ahead:
1. **Air Quality Index (AQI)**: Hourly particulate concentration ($PM_{2.5}, PM_{10}$)
2. **Temperature (°C)**: Diurnal coastal heating cycle
3. **Precipitation (mm)**: Rainfall probability and volume
4. **Relative Humidity (%)**: Marine atmospheric moisture
5. **UV Solar Index**: Photochemical solar risk
6. **Wind Speed & Direction**: Decomposed into $U$ & $V$ vectors (Karachi South-West sea breeze)

---

## 🏆 Model Tournament vs 24-Hour Naive Baseline
All candidate models were strictly benchmarked against the 24-hour Naive Baseline ($y_{\text{naive}} = y_{t-24}$). All 6 champion models significantly outperformed the baseline:

| Metric | Champion Model | Naive Baseline (RMSE) | Champion (RMSE) | Improvement (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Temperature** | Statistical Ridge | 1.264 | **0.905** | **+28.4%** |
| **Precipitation** | Statistical Ridge | 1.128 | **0.752** | **+33.3%** |
| **UV Index** | Classical Random Forest | 0.388 | **0.294** | **+24.2%** |
| **Humidity** | Statistical Ridge | 3.494 | **2.824** | **+19.2%** |
| **AQI** | Statistical Ridge | 11.738 | **9.998** | **+14.8%** |
| **Wind Speed** | Statistical Ridge | 1.224 | **1.082** | **+11.6%** |

---

## 📈 Current Milestone Status (Step 2 Completed)
- [x] **Step 1:** Base repository, Feature pipeline, Hopsworks cache, Model Tournament vs Naive Baseline.
- [x] **Step 2:** 24-hour continuous trajectory curves, Karachi diurnal cycle modeling, and EDA & Data Storytelling module (`eda.py`).
- [ ] **Step 3:** Live OpenWeather & AQICN API token integration and automated GitHub Actions hourly cron.
- [ ] **Step 4:** Model Explainability with SHAP / LIME visualizations.
- [ ] **Step 5:** 1-Click Streamlit Community Cloud public deployment.

---

## 🌐 Deployment Strategy
The dashboard is designed for seamless, free hosting on **Streamlit Community Cloud**:
1. Connect GitHub repository `SMArham/AgentWeather`.
2. Main file path: `app.py`.
3. Set environment secrets (`HOPSWORKS_API_KEY`, `OPENWEATHER_API_KEY`, `AQICN_API_TOKEN`, `GROQ_API_KEY`).
4. Instant live public dashboard URL.

---

## 🚀 Local Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Feature Ingestion
python feature_pipeline.py

# 3. Run Training Tournament & Baseline Benchmarks
python training_pipeline.py

# 4. Run Exploratory Data Analysis
python eda.py

# 5. Launch Interactive Dashboard
streamlit run app.py
```

---

## 👨‍💻 Author
**Syed Muhammad Arham**  
Data Science Intern — 10Pearls SHINE Program (Cohort 10)  
GitHub: [@SMArham](https://github.com/SMArham)

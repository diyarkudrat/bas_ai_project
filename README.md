# Data Center Cooling Optimizer

# Overview

A software-based platform that simulates and analyzes critical data center cooling systems using **AI and mock sensor data**.  
It detects inefficiencies, predicts maintenance issues, and simulates automated control actions — without needing access to real hardware.


## 📌 Why This Project Matters

Data centers consume massive amounts of energy to keep servers cool.  
Small inefficiencies in cooling systems can waste **thousands of dollars per year** and risk downtime.  
This project demonstrates how AI can:

- **Spot anomalies early** (e.g., faulty sensors, stuck valves)
- **Predict when maintenance is needed** before breakdowns occur
- **Optimize cooling settings** to reduce energy use without sacrificing performance


## 🖼 How It Works (Simple Version)

1. **Generate Realistic Sensor Data**  
   - Simulated readings for **liquid cooling loops**, **airflow systems**, and **IT load**  
   - Includes realistic ranges, daily patterns, and “fault scenarios” for testing

2. **Store & Process Data**  
   - Save time-series data locally  
   - Clean and prepare data for AI analysis

3. **Analyze with AI**  
   - **Anomaly Detection**: Finds unusual patterns that could mean a fault  
   - **Efficiency Scoring**: Flags low cooling efficiency  
   - **Predictive Maintenance**: Spots gradual equipment performance decline

4. **Control Simulation**  
   - Simulates safe, automated adjustments (like fan speed or valve position)  
   - Includes “safe mode” to protect the system during anomalies

5. **Visual Dashboard**  
   - Shows live metrics, efficiency scores, and alerts  
   - Easy-to-read charts for both engineers and management

---

# High-Level Architecture

### Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Mock Data       │───▶│ Data Ingestion  │───▶│ Data Storage    │───▶│ Processing &    │
│ Generator /     │    │ Layer           │    │ Layer           │    │ Feature         │
│ Real Sensor     │    │                 │    │                 │    │ Engineering     │
│ Interfaces      │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Integration &   │◀───│ Visualization   │◀───│ Control Logic   │◀───│ AI/ML Models    │
│ Deployment      │    │ & Dashboard     │    │ Layer           │    │                 │
│                 │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```
---

## ⚙️ Technology Stack

| Layer              | Demo Implementation        | Scalable/Production Option         |
|--------------------|----------------------------|-------------------------------------|
| Data Generation    | Python scripts (CSV)       | Live sensor feeds via BACnet/Modbus |
| Storage            | CSV + DuckDB               | TimescaleDB / InfluxDB              |
| Processing         | Pandas, NumPy               | Spark / Flink                       |
| AI/ML              | scikit-learn                | MLflow + ONNX Runtime               |
| Control Logic      | Python MPC-lite algorithm   | Industrial PLC integration          |
| Visualization      | Streamlit dashboard         | React + Plotly/ECharts              |
| Deployment         | Local Python scripts        | Kubernetes + Edge devices           |

---

## Key Design Principles
1. **Modularity** — Each layer can be swapped or scaled independently.
2. **Edge-Ready** — Interfaces and protocols chosen to support on-site deployment.
3. **Safety** — Fail-safes and alerting before automated control actions.
4. **Explainability** — Rules and models are transparent for operators.
5. **Scalability** — From a single laptop demo to multi-site data center deployment.
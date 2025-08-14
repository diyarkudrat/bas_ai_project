# Mock Data for Data Center Cooling Optimizer

This project uses **realistic simulated sensor data** to mimic the behavior of critical data center cooling components. It allows us to test AI algorithms and control logic.

---

## Purpose of the Mock Data

- **Test AI models**: Train and evaluate anomaly detection, efficiency scoring, and predictive maintence features.
- **Simulate real-world conditions**: Include normal operation, daily patterns, and rare fault events.
- **Validate control logic**: See how automated decisions respond to realistic data scenarios.

---

## What's Simulated

The mock data generator creates **time-series readings** at 1-5 second intervals for three main areas:

### 1. Liquid Cooling Loop
| Signal | Normal Range | Description |
|--------|--------------|-------------|
| `supply_temp_C` | 18–22 °C | Temperature supplied to cooling loop |
| `return_temp_C` | 24–30 °C | Temperature returning from IT equipment |
| `deltaT_C` | 5–10 °C | Return − Supply temperature; key efficiency metric |
| `flow_LPM` | 40–250 L/min | Flow rate of coolant |
| `valve_pct` | 0–100 % | Valve opening percentage |
| `supply_pressure_kPa` | 150–250 kPa | Loop pressure from pump output |

---

### 2. CRAH (Computer Room Air Handler) & Airflow
| Signal | Normal Range | Description |
|--------|--------------|-------------|
| `inlet_temp_C` | 22–26 °C target | Air temperature entering server racks |
| `fan_speed_pct` | 20–100 % | Fan speed setpoint |
| `airflow_CFM` | 5k–40k CFM | Air volume delivered |
| `rh_pct` | 30–55 % | Relative humidity in the cold aisle |

---

### 3. IT Load & Environment
| Signal | Normal Range | Description |
|--------|--------------|-------------|
| `rack_IT_kW` | 1–35 kW | IT equipment load |
| `ambient_C` | 15–35 °C | Outside air temperature |

---

## Fault & Anomly Scenarios

The generator **injects anomalies** at a configurable rate to test fault detection:

- **Drift**: Gradual increase or decrease in readings (e.g., +2.5 °C over 15 minutes)
- **Stuck Values**: Sensor or actuator output freezes temporarily
- **Step Changes**: Sudden jumps in readings
- **Dropouts**: Missing data (NaN values) for short periods
- **Pressure Drops**: Sharp decreases in pump or system pressure

These anomalies help simulate real operational challenges.

---

## How It’s Generated

- Written in **Python** using **NumPy** and **Pandas**
- Allows tuning of:
  - **Duration** of simulation (`--minutes`)
  - **Sampling rate** in seconds (`--dt`)
  - **Anomaly probability** per data point (`--anomaly-rate`)
  - **Random seed** for reproducibility
- Outputs to a `.csv` file for easy integration

---

## Running the Generator

### 1. **Install Dependencies**

```bash
pip install numpy pandas
```

### 2. Run the Script

```bash
python sim_generate.py --minutes 120 --dt 5 --anomaly-rate 0.02 --seed 42
```

### 3. Output

- Creates `sensor_stream.csv`
- Each row = 1 timestamp
- Each column = 1 sensor reading
- Includes normal data **and** injected anomalies

## Why This Matters

Real data from a data center is:
- **Hard to access** due to operational security constraints.
- **Expensive** to capture over long time periods.
- **Risky** to manipulate for testing.

Simulated data allows us to:
- Build and test AI models risk-free.
- Reproduce fault conditions on demand.
- Share the project publicly without exposing sensitive data.

## Future Enhancements

- Generate data **in real-time** over MQTT to mimic a live feed.
- Support **BACnet/IP** and **Modbus** output formats.
- Add **more sensor types** (e.g., chiller plant metrics, electrical meters, etc.)
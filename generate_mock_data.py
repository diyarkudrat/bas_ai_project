"""
generate_mock_data.py — generate realistic time-series for data center cooling + airflow systems.

Outputs a single CSV with columns covering liquid cooling loop, CRAH/airflow, and IT load.
Includes parameterizable anomalies (drift, stuck, step, dropout, pressure drops).

Usage examples:
  python generate_mock_data.py --minutes 120 --dt 5 --anomaly-rate 0.02 --seed 42
  python generate_mock_data.py --minutes 60 --dt 2 --anomaly-rate 0.04 --out sensor_stream_dense.csv

"""

import argparse
import numpy as np
import pandas as pd

# ------------------------------ Core generator ------------------------------ #
def generate_mock_data(n: int, dt_s: int, anomaly_rate: float, seed: int) -> pd.DataFrame:
    """Generate a simple synthetic time series.

    Parameters
    ----------
    n : int
        Number of samples (rows)
    dt_s : int
        Seconds per sample (cadence)
    anomaly_rate : float
        Probability at any tick to start an anomaly window
    seed : int
        RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed) # random number generator
    t = np.arange(n) * dt_s  # seconds since start of simulation

    # -------------------- Drivers: IT load & ambient -------------------- #
    # Daily sine baseline (6↔18 kW) + occasional spikes of 5–12 kW

    base_kw = 12 + 6 * np.sin(2 * np.pi * (t / (24 * 3600))) # baseline IT load
    bursts = (rng.random(n) < 0.010) * rng.uniform(5, 12, size=n) # occasional spikes
    it_kw = np.clip(base_kw + bursts, 1, 35) # IT load

    # Outside air: mean 22°C, ±5°C daily swing + small noise
    ambient = 22 + 5 * np.sin(2 * np.pi * (t / (24 * 3600) + 0.2)) + rng.normal(0, 0.3, n)

    # -------------------- Liquid cooling loop --------------------------- #
    # Supply water: setpoint ~19.5°C, slightly influenced by ambient
    supply = 19.5 + 0.1 * (ambient - 22) + rng.normal(0, 0.08, n)

    # ΔT grows a bit with IT load. Keep it in a sane band (3–12°C)
    deltaT = np.clip(7 + 0.12 * (it_kw - 12) + rng.normal(0, 0.2, n), 3, 12) # water temp rise across IT equipment
    ret = supply + deltaT # water temp coming back from IT equipment

    # Valve roughly follows load; flow follows valve (very simplified)
    valve = np.clip(20 + 1.6 * (it_kw - 12) + rng.normal(0, 4, n), 0, 100) # valve opening percentage in the cooling loop
    flow = np.clip(120 + 3.0 * (valve - 20) + rng.normal(0, 6, n), 30, 300) # flow rate through the cooling loop

    # Pressure correlates with flow
    pressure = np.clip(180 + 0.4 * (flow - 120) + rng.normal(0, 5, n), 120, 280) # chilled water loop pressure

    # -------------------- CRAH / airflow side --------------------------- #
    # Inlet temperature near target 23°C, nudged by load
    inlet = np.clip(23 + 0.03 * (it_kw - 12) + rng.normal(0, 0.2, n), 18, 29) # IT inlet air temperature

    # Fan speeds up when inlet rises; airflow follows fan
    fan = np.clip(35 + 4.0 * (inlet - 23) + rng.normal(0, 3, n), 20, 100) # fan speed percentage
    airflow = np.clip(15000 + 180 * (fan - 35) - rng.normal(0, 500, n), 5000, 40000) # air volume in CFM (cubic feet per minute)

    # Humidity floats around 45%
    rh = np.clip(45 + rng.normal(0, 3, n), 25, 65) # relative humidity in data center

    # -------------------- Simple anomalies ------------------------------ #
    # Keep these basic: drift on supply, stuck valve, and inlet dropout.
    supply_fault = supply.copy()
    valve_fault = valve.copy()
    inlet_fault = inlet.copy()

    # Helper: pick windows for anomalies (few minutes long)
    def window(start_idx: int, seconds: int) -> tuple[int, int]:
        s = int(start_idx)
        e = min(n, s + int(max(1, seconds // dt_s)))
        return s, e

    idxs = np.where(rng.random(n) < anomaly_rate)[0]

    # Supply drift: slowly rises by 1–3°C over ~15 min
    for idx in idxs[::3]:
        s, e = window(idx, 900)
        drift = np.linspace(0, rng.uniform(1.0, 3.0), e - s)
        supply_fault[s:e] += drift

    # Stuck valve: holds current position for ~10 min
    for idx in idxs[1::3]:
        s, e = window(idx, 600)
        valve_fault[s:e] = valve_fault[s]

    # Inlet dropout: NaNs for ~2 min (optional: downstream can interpolate)
    for idx in idxs[2::3]:
        s, e = window(idx, 120)
        inlet_fault[s:e] = np.nan

    # -------------------- Assemble DataFrame ---------------------------- #
    start_ts = pd.Timestamp.utcnow().floor("s")
    df = pd.DataFrame({
        "t_s": t,
        "timestamp": start_ts + pd.to_timedelta(t, unit="s"),
        "ambient_C": ambient,
        "it_kw": it_kw,
        "supply_C": supply_fault,
        "return_C": ret,
        "deltaT_C": ret - supply_fault,  # observed ΔT using faulted supply
        "valve_pct": valve_fault,
        "flow_LPM": flow,
        "pressure_kPa": pressure,
        "inlet_C": inlet_fault,
        "fan_pct": fan,
        "airflow_CFM": airflow,
        "rh_pct": rh,
    })

    # cooling system power use
    cooling_kw = 0.015 * df["airflow_CFM"] + 0.008 * df["flow_LPM"] + 0.12 * np.maximum(0.0, np.nan_to_num(df["inlet_C"]) - 22)
    df["pue_proxy"] = (df["it_kw"] + cooling_kw) / df["it_kw"]

    # Light touch: fill small gaps so charts don't break; comment this out to keep NaNs
    df.interpolate(limit_direction="both", inplace=True)

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate simple mock cooling/airflow data")
    ap.add_argument("--minutes", type=float, default=60, help="total minutes to simulate")
    ap.add_argument("--dt", type=int, default=5, help="timestep in seconds (1–10)")
    ap.add_argument("--anomaly-rate", type=float, default=0.02, help="probability per tick to start an anomaly (0.0–0.1)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--out", type=str, default="sensor_stream.csv", help="output CSV path")
    args = ap.parse_args()

    # Basic input hygiene
    if args.dt < 1 or args.dt > 10:
        raise ValueError("--dt must be between 1 and 10 seconds")
    if args.minutes <= 0:
        raise ValueError("--minutes must be > 0")
    if not (0.0 <= args.anomaly_rate <= 0.1):
        raise ValueError("--anomaly-rate must be between 0.0 and 0.1")

    n = int((args.minutes * 60) // args.dt)
    df = generate_mock_data(n=n, dt_s=args.dt, anomaly_rate=args.anomaly_rate, seed=args.seed)
    df.to_csv(args.out, index=False)

    print(f"Wrote {args.out} | rows={len(df):,} | dt={args.dt}s | minutes={args.minutes} | anomaly_rate={args.anomaly_rate} | seed={args.seed}")


if __name__ == "__main__":
    main()

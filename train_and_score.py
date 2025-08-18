#!/usr/bin/env python3
"""
Version 1.0 (basic) — train_and_score.py

Goal: keep the first ML pass simple and readable. We do two things:
  1) Unsupervised anomaly detection (IsolationForest) on a few intuitive features
  2) Inefficiency flag using a simple model of expected ΔT given load/flow/supply

Inputs:
  - sensor_stream.csv  (from sim_generate.py v1.0)

Outputs:
  - scored_stream.csv  (adds anomaly scores/flags and inefficiency columns)

How to run:
  python train_and_score.py --in sensor_stream.csv --out scored_stream.csv

Notes:
  - This is NOT a production pipeline. It's a teaching/demo baseline.
  - We interpolate small gaps to keep the first version frictionless.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ------------------------------ features ---------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create light, human-intuitive features.
    """

    d = df.copy()

    # Short context window. Adjusts automatically with dt.
    window = min(60, max(10, len(d)//50))

    cols = [
        "supply_C", "return_C", "deltaT_C", "valve_pct", "flow_LPM", "pressure_kPa",
        "inlet_C", "fan_pct", "airflow_CFM", "it_kw",
    ]

    for c in cols:
        d[c + "_mean"] = d[c].rolling(window=window).mean()
        d[c + "_z"] = (d[c] - d[c + "_mean"]) / d[c].rolling(window=window).std()

    for c in cols:
        mu = d[c].rolling(window, min_periods=1).mean()
        sd = d[c].rolling(window, min_periods=1).std().fillna(1e-6)
        d[f"{c}_z"] = (d[c] - mu) / (sd + 1e-6)
        d[f"{c}_roc"] = d[c].diff().fillna(0.0)  # simple rate-of-change

    # Cross-relations (very simple):
    d["flow_per_valve"] = d["flow_LPM"] / (d["valve_pct"] + 1e-3)
    d["deltaT_over_flow"] = d["deltaT_C"] / (d["flow_LPM"] + 1e-3)

    return d.fillna(0.0)

# ------------------------------ main -------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Simple anomaly + inefficiency scoring")
    ap.add_argument("--in", type=str, default="sensor_stream.csv", help="input CSV from simulator")
    ap.add_argument("--out", type=str, default="scored_stream.csv", help="output CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.__dict__["in"])

    df.interpolate(limit_direction="both", inplace=True)

    feats = build_features(df)

    feature_cols = [
        c for c in feats.columns
        if any(tag in c for tag in ("_z","_roc","flow_per_valve","deltaT_over_flow"))
    ]

    # Scale features so distances aren't dominated by any single feature scale.
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[feature_cols])

    # --- (1) Unsupervised anomaly detection --- #
    # Contamination is the expected fraction of anomalies; match sim default (~2%).
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=0)
    iso.fit(X)

    # IsolationForest returns larger (less negative) scores for normal points.
    # We flip/normalize so higher => more anomalous for easier reading.
    df["anomaly_score"] = -iso.decision_function(X)
    threshold = np.quantile(df["anomaly_score"], 0.98)
    df["anomaly_flag"] = (df["anomaly_score"] > threshold).astype(int)

    # --- (2) Inefficiency via ΔT residual --- #
    # Train a small model to predict expected ΔT from a few drivers, 
    # using only windows the detector considers normal.
    ok = df["anomaly_flag"] == 0
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(df.loc[ok, ["it_kw","flow_LPM","supply_C"]], df.loc[ok, "deltaT_C"])

    df["deltaT_pred"] = model.predict(df[["it_kw","flow_LPM","supply_C"]])
    df["deltaT_residual"] = df["deltaT_pred"] - df["deltaT_C"]

    # Flag inefficiency when residual is high (>1.5°C) or ΔT is outright low (<5°C)
    df["inefficiency_flag"] = ((df["deltaT_residual"] > 1.5) | (df["deltaT_C"] < 5)).astype(int)

    # Save
    df.to_csv(args.__dict__["out"], index=False)

    # Tiny summary for the console
    print(
        f"Wrote {args.__dict__['out']} | rows={len(df):,} | "
        f"anomalies={int(df['anomaly_flag'].sum())} | "
        f"inefficiency_flags={int(df['inefficiency_flag'].sum())}"
    )


if __name__ == "__main__":
    main()
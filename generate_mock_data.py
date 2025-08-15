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

def generate_mock_data(n=3600, dt_s=5, anomaly_rate=0.01, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt_s
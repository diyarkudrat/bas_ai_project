#!/usr/bin/env python3
"""
Version 2.0 — train_and_score.py

Refactored to clean, modular design:
  - FeatureBuilder: computes robust, human-intuitive time-series features
  - AnomalyDetector: encapsulates IsolationForest logic and thresholds
  - InefficiencyModel: predicts expected ΔT, emits residuals and flags
  - ScoringPipeline: orchestrates the end-to-end flow

CLI remains compatible with v1 defaults.
"""

import argparse
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ------------------------------ feature builder ---------------------------- #
@dataclass(frozen=True)
class FeatureBuilder:
    base_columns: Sequence[str] = (
        "supply_C",
        "return_C",
        "deltaT_C",
        "valve_pct",
        "flow_LPM",
        "pressure_kPa",
        "inlet_C",
        "fan_pct",
        "airflow_CFM",
        "it_kw",
    )
    min_window: int = 10
    max_window: int = 60

    def build(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Build a copy of the data with rolling statistics and simple interactions.

        The window adapts to dataset length to avoid over/under-smoothing.
        """
        augmented_data_frame = data_frame.copy()
        window = min(self.max_window, max(self.min_window, len(augmented_data_frame) // 50))

        for column in self.base_columns:
            # Rolling stats
            rolling_mean = augmented_data_frame[column].rolling(window=window)
            augmented_data_frame[f"{column}_mean"] = rolling_mean.mean()

        for column in self.base_columns:
            # Compute rolling statistics for local z-score normalization
            mu = augmented_data_frame[column].rolling(window, min_periods=1).mean()
            sd = augmented_data_frame[column].rolling(window, min_periods=1).std().fillna(1e-6)

            # Create features: z-scores (how unusual vs recent history) and rate of change
            augmented_data_frame[f"{column}_z"] = (augmented_data_frame[column] - mu) / (sd + 1e-6)
            augmented_data_frame[f"{column}_roc"] = augmented_data_frame[column].diff().fillna(0.0)

        # Cross-relations
        augmented_data_frame["flow_per_valve"] = augmented_data_frame["flow_LPM"] / (augmented_data_frame["valve_pct"] + 1e-3)
        augmented_data_frame["deltaT_over_flow"] = augmented_data_frame["deltaT_C"] / (augmented_data_frame["flow_LPM"] + 1e-3)

        return augmented_data_frame.fillna(0.0)


# ------------------------------ anomaly detector --------------------------- #
@dataclass
class AnomalyDetector:
    """
    IsolationForest-based anomaly detector with internal scaling and explicit
    quantile thresholding on flipped decision scores (higher = more anomalous).
    """

    contamination: float = 0.02 # expected percentage of outliers in the data.
    n_estimators: int = 200 # number of trees in the IsolationForest (ML model).
    random_state: int = 0 # seed for reproducibility.
    threshold_quantile: float = 0.98 # Scores above this value are considered anomalies.

    def __post_init__(self) -> None:
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    def fit_transform(self, features: pd.DataFrame, feature_columns: List[str]) -> pd.Series:
        """
        1. Scale the selected feature columns and fit IsolationForest.
        
        2. Return anomaly scores as a Series (higher = more anomalous).
        """

        X = self._scaler.fit_transform(features[feature_columns])
        self._model.fit(X)

        # IsolationForest returns negative scores where more negative = more anomalous.
        # We flip the sign so higher scores = more anomalous (easier to interpret).
        scores = -self._model.decision_function(X)
        
        return pd.Series(scores, index=features.index, name="anomaly_score")

    def predict_flags(self, scores: pd.Series) -> pd.Series:
        """
        1. Compute a threshold at `threshold_quantile` over the provided scores.

        2. Return a 0/1 Series where 1 indicates an anomaly (score > threshold).
        """
        
        threshold = float(np.quantile(scores, self.threshold_quantile))
        flags = (scores > threshold).astype(int)
        return flags.rename("anomaly_flag")


# ------------------------------ inefficiency model ------------------------- #
@dataclass
class InefficiencyModel:
    """
    Predicts expected ΔT (temperature difference) based on IT power, flow, and supply temperature.

    Flags inefficiencies where ΔT is too high or too low.
    """
    n_estimators: int = 200 # number of trees in the RandomForestRegressor (ML model).
    random_state: int = 0 # seed for reproducibility.
    residual_threshold_C: float = 1.5 # temperature threshold for flagging high residuals (predicted - actual).
    low_deltaT_threshold_C: float = 5.0 # minimum temperature difference for flagging poor heat rejection.

    def __post_init__(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )

    def fit(self, df: pd.DataFrame, ok_mask: pd.Series) -> None:
        """
        Fit the ML model on the training data.

        Only trains on non-anomalous windows to avoid learning from faulty data.
        """
        
        # Inputs used to predict temperature difference.
        inputs = ["it_kw", "flow_LPM", "supply_C"]

        self._model.fit(df.loc[ok_mask, inputs], df.loc[ok_mask, "deltaT_C"])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict temperature difference using the trained ML model.
        """
        
        inputs = ["it_kw", "flow_LPM", "supply_C"]
        preds = self._model.predict(df[inputs])

        return pd.Series(preds, index=df.index, name="deltaT_pred")

    def flag_inefficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        Flag inefficiencies where the predicted temperature difference is too high or too low.
        """
        
        residual = df["deltaT_pred"] - df["deltaT_C"]
        hi_residual = residual > self.residual_threshold_C
        low_deltaT = df["deltaT_C"] < self.low_deltaT_threshold_C

        return (hi_residual | low_deltaT).astype(int).rename("inefficiency_flag")

# ------------------------------ pipeline ----------------------------------- #
@dataclass
class ScoringPipeline:
    """
    Orchestrates the end-to-end scoring reliably.
    """
    
    feature_builder: FeatureBuilder
    anomaly_detector: AnomalyDetector
    inefficiency_model: InefficiencyModel

    def run(self, raw: pd.DataFrame) -> pd.DataFrame:
        # 1) Clean small gaps
        df = raw.copy()
        df.interpolate(limit_direction="both", inplace=True)

        # 2) Features
        feats = self.feature_builder.build(df)
        feature_cols = [
            c
            for c in feats.columns
            if any(tag in c for tag in ("_z", "_roc", "flow_per_valve", "deltaT_over_flow"))
        ]

        # 3) Anomaly detection
        scores = self.anomaly_detector.fit_transform(feats, feature_cols)
        flags = self.anomaly_detector.predict_flags(scores)

        # 4) Inefficiency residual model trained only on normal windows
        ok = flags == 0
        self.inefficiency_model.fit(df, ok)
        deltaT_pred = self.inefficiency_model.predict(df)
        df_out = df.copy()
        df_out["anomaly_score"] = scores
        df_out["anomaly_flag"] = flags
        df_out["deltaT_pred"] = deltaT_pred
        df_out["deltaT_residual"] = df_out["deltaT_pred"] - df_out["deltaT_C"]
        df_out["inefficiency_flag"] = self.inefficiency_model.flag_inefficiency(df_out)

        return df_out


# ------------------------------ cli / main --------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Simple anomaly + inefficiency scoring")
    ap.add_argument("--in", type=str, default="sensor_stream.csv", help="input CSV from simulator")
    ap.add_argument("--out", type=str, default="scored_stream.csv", help="output CSV")
    args = ap.parse_args()

    raw_df = pd.read_csv(args.__dict__["in"])

    pipeline = ScoringPipeline(
        feature_builder=FeatureBuilder(),
        anomaly_detector=AnomalyDetector(),
        inefficiency_model=InefficiencyModel(),
    )

    result = pipeline.run(raw_df)
    result.to_csv(args.__dict__["out"], index=False)

    print(
        f"Wrote {args.__dict__['out']} | rows={len(result):,} | "
        f"anomalies={int(result['anomaly_flag'].sum())} | "
        f"inefficiency_flags={int(result['inefficiency_flag'].sum())}"
    )


if __name__ == "__main__":
    main()
"""
Module: anomaly_detector.py
Detects anomalies using three methods:
  1. Z-Score (statistical)
  2. IQR (statistical)
  3. Isolation Forest (ML-based, unsupervised)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Detects and optionally removes/flags anomalous rows."""

    def __init__(self, df: pd.DataFrame, contamination: float = 0.05):
        """
        Args:
            df: Input DataFrame (should be post-cleaning).
            contamination: Estimated fraction of anomalies (for Isolation Forest).
        """
        self.df = df.copy()
        self.contamination = contamination
        self.anomaly_report = {}
        # Reduce estimators for large datasets to avoid timeout
        # Large = more than 50k rows
        self.n_estimators = 50 if len(df) < 50000 else 20

    def run(self, strategy: str = "flag") -> pd.DataFrame:
        """
        Detect anomalies and either 'flag' them or 'remove' them.

        Args:
            strategy: 'flag' adds an 'is_anomaly' column; 'remove' drops anomalous rows.

        Returns:
            Cleaned/flagged DataFrame.
        """
        print("\n[Anomaly Detector] Running anomaly detection...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            print("  → No numeric columns found. Skipping anomaly detection.")
            return self.df

        # Run all three methods
        zscore_anomalies = self._zscore_detection(numeric_cols)
        iqr_anomalies = self._iqr_detection(numeric_cols)
        iforest_anomalies = self._isolation_forest(numeric_cols)

        # Combine: a row is anomalous if flagged by at least 2 out of 3 methods
        combined = zscore_anomalies | iqr_anomalies | iforest_anomalies
        consensus = (
            zscore_anomalies.astype(int)
            + iqr_anomalies.astype(int)
            + iforest_anomalies.astype(int)
        ) >= 2

        self.anomaly_report = {
            "zscore_count": int(zscore_anomalies.sum()),
            "iqr_count": int(iqr_anomalies.sum()),
            "isolation_forest_count": int(iforest_anomalies.sum()),
            "consensus_count": int(consensus.sum()),
            "anomalous_indices": self.df[consensus].index.tolist(),
        }

        print(f"  → Z-Score anomalies      : {self.anomaly_report['zscore_count']}")
        print(f"  → IQR anomalies          : {self.anomaly_report['iqr_count']}")
        print(f"  → Isolation Forest       : {self.anomaly_report['isolation_forest_count']}")
        print(f"  → Consensus anomalies    : {self.anomaly_report['consensus_count']} (flagged by ≥2 methods)")

        if strategy == "flag":
            self.df["is_anomaly"] = consensus.astype(int)
            print(f"  → Strategy: FLAGGED anomalies in 'is_anomaly' column")
        elif strategy == "remove":
            before = len(self.df)
            self.df = self.df[~consensus].reset_index(drop=True)
            print(f"  → Strategy: REMOVED {before - len(self.df)} anomalous row(s)")

        return self.df

    def get_report(self) -> dict:
        return self.anomaly_report

    # ------------------------------------------------------------------ #
    #  Method 1: Z-Score  (|z| > 3 considered anomaly)                   #
    # ------------------------------------------------------------------ #
    def _zscore_detection(self, cols: list) -> pd.Series:
        anomaly_mask = pd.Series(False, index=self.df.index)
        for col in cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std == 0:
                continue
            z_scores = (self.df[col] - mean) / std
            anomaly_mask |= z_scores.abs() > 3
        return anomaly_mask

    # ------------------------------------------------------------------ #
    #  Method 2: IQR  (value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR])     #
    # ------------------------------------------------------------------ #
    def _iqr_detection(self, cols: list) -> pd.Series:
        anomaly_mask = pd.Series(False, index=self.df.index)
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            anomaly_mask |= (self.df[col] < lower) | (self.df[col] > upper)
        return anomaly_mask

    # ------------------------------------------------------------------ #
    #  Method 3: Isolation Forest (ML-based, unsupervised)               #
    # ------------------------------------------------------------------ #
    def _isolation_forest(self, cols: list) -> pd.Series:
        X = self.df[cols].copy()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=self.n_estimators
        )
        preds = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal

        return pd.Series(preds == -1, index=self.df.index)

"""
Module: ml_imputer.py
Predicts missing values using ML models (RandomForest) with:
  - Confidence scores for every imputed value
  - Range clipping: predictions are clipped to the min-max range
    learned from the actual training data, so ML can never predict
    an impossible value like age=118 or marks=999
  - Placeholder imputation for identity columns (names, emails etc.)
  - Statistical fallback (median/mode) when ML is not viable
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class MLImputer:
    MIN_ROWS_FOR_ML = 20

    def _get_n_estimators(self) -> int:
        """Use fewer trees for large datasets to prevent timeout."""
        return 20 if len(self.df) > 50000 else 50

    def _detect_column_type(self, col: str) -> str:
        """
        Detect the semantic type of a numeric column so predictions
        are formatted correctly. Returns one of:
          - 'boolean'    : only 0 and 1 (e.g. yes/no encoded as 0/1)
          - 'rating'     : integers on a fixed small scale (e.g. 1-5, 1-10)
          - 'integer'    : whole numbers with no fixed scale (e.g. age, marks)
          - 'percentage' : values between 0 and 100 (e.g. attendance_pct)
          - 'decimal'    : general decimal numbers (e.g. salary, price)
        """
        series = self.df[col].dropna()
        if len(series) == 0:
            return 'decimal'

        unique_vals = sorted(series.unique())
        n_unique = len(unique_vals)
        col_min = float(series.min())
        col_max = float(series.max())

        # Check if 95%+ values are whole numbers
        whole_ratio = (series == series.round(0)).sum() / len(series)
        is_whole = whole_ratio >= 0.95

        # Boolean: only contains 0 and 1
        if is_whole and set(unique_vals).issubset({0.0, 1.0}):
            return 'boolean'

        # Rating scale: whole numbers, small range (<=10 unique values),
        # min is 0 or 1, max is between 2 and 10
        if is_whole and n_unique <= 10 and col_min >= 0 and col_max <= 10:
            return 'rating'

        # Percentage: values between 0 and 100 (allow small decimals like 92.5)
        if col_min >= 0 and col_max <= 100:
            # Check if it looks like a percentage (has values spread across range)
            range_span = col_max - col_min
            if range_span > 10:  # Not a tiny scale — likely a percentage
                return 'percentage'

        # Integer: 95%+ whole numbers, not a small scale
        if is_whole:
            return 'integer'

        return 'decimal'

    def _format_prediction(self, value: float, col_type: str) -> float:
        """
        Format a predicted value based on the column type.
        Ensures predictions match the format of the original data.
        """
        if col_type == 'boolean':
            # Round to nearest 0 or 1
            return float(round(value))

        elif col_type == 'rating':
            # Round to nearest whole number
            return float(round(value))

        elif col_type == 'integer':
            # Round to nearest whole number
            return float(round(value))

        elif col_type == 'percentage':
            # Keep 1 decimal place, clip to 0-100
            return round(float(np.clip(value, 0, 100)), 1)

        else:  # decimal
            # Keep 2 decimal places
            return round(float(value), 2)

    def _format_display(self, value: float, col_type: str) -> str:
        """Format a value for display in the log."""
        if col_type in ('boolean', 'rating', 'integer'):
            return str(int(value))
        elif col_type == 'percentage':
            return f"{value:.1f}%"
        else:
            return f"{value:.2f}"

    NON_PREDICTABLE_KEYWORDS = [
        "name", "email", "mail", "phone", "mobile", "contact",
        "address", "url", "link", "uuid", "guid", "username",
        "user_name", "first_name", "last_name", "full_name"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.log = []
        self._label_encoders = {}

    def run(self) -> pd.DataFrame:
        print("\n[Imputer] Running ML-based imputation...")

        missing_cols = [c for c in self.df.columns if self.df[c].isnull().any()]

        if not missing_cols:
            print("  -> No missing values found. Skipping.")
            return self.df

        for col in missing_cols:
            missing_count = self.df[col].isnull().sum()
            print(f"  Imputing '{col}' ({missing_count} missing value(s))...")
            if self._is_non_predictable(col):
                self._placeholder_impute(col)
            else:
                self._impute_column(col)

        return self.df

    def get_log(self) -> list:
        return self.log

    def _is_non_predictable(self, col: str) -> bool:
        """
        Returns True if a column should not be imputed with ML.
        Detects three cases:
          1. Name/email/phone/URL columns by keyword matching
          2. High-cardinality text columns (likely identity fields)
          3. High-cardinality numeric columns — these are timestamps,
             transaction IDs, order numbers etc. ML cannot meaningfully
             predict these since every value is essentially unique.
             Detection: if >50% of numeric values are unique AND the
             values are large numbers (>100000), treat as ID/timestamp.
        """
        col_lower = col.lower()

        # Check 1: known non-predictable keywords
        if any(keyword in col_lower for keyword in self.NON_PREDICTABLE_KEYWORDS):
            return True

        # Check 2: high cardinality text column
        if self.df[col].dtype == object:
            notna_count = self.df[col].notna().sum()
            if notna_count > 0 and self.df[col].nunique() / notna_count > 0.7:
                return True

        # Check 3: high cardinality numeric column (timestamp/ID/transaction number)
        if pd.api.types.is_numeric_dtype(self.df[col]):
            notna_vals = self.df[col].dropna()
            if len(notna_vals) > 0:
                unique_ratio = notna_vals.nunique() / len(notna_vals)
                median_val = notna_vals.median()
                # Large numbers with high uniqueness = timestamp or ID
                if unique_ratio > 0.5 and abs(median_val) > 100000:
                    return True

        return False

    def _impute_column(self, target_col: str):
        col_data = self.df[target_col]
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        feature_cols = self._get_usable_features(target_col)
        train_mask = col_data.notna()
        pred_mask = col_data.isna()

        if train_mask.sum() < self.MIN_ROWS_FOR_ML or len(feature_cols) == 0:
            self._statistical_impute(target_col, is_numeric)
            return

        try:
            X_train, X_pred = self._prepare_features(feature_cols, train_mask, pred_mask)
            y_train = col_data[train_mask]

            if is_numeric:
                # --- Learn the valid range from ACTUAL observed data ---
                # Exclude anomaly-flagged rows from range calculation so
                # outliers like 999 do not pollute the valid range.
                if "is_anomaly" in self.df.columns:
                    clean_mask = train_mask & (self.df["is_anomaly"] == 0)
                    y_clean = col_data[clean_mask]
                    valid_min = float(y_clean.min()) if len(y_clean) > 0 else float(y_train.min())
                    valid_max = float(y_clean.max()) if len(y_clean) > 0 else float(y_train.max())
                else:
                    valid_min = float(y_train.min())
                    valid_max = float(y_train.max())

                model = RandomForestRegressor(n_estimators=self._get_n_estimators(), random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_pred)

                # Clip predictions to the valid range
                predictions_clipped = np.clip(predictions, valid_min, valid_max)

                # Flag any predictions that were clipped
                clipped_mask = predictions != predictions_clipped
                if clipped_mask.any():
                    clipped_info = [
                        f"row {idx}: {orig:.1f} -> clipped to {clipped:.1f}"
                        for idx, orig, clipped, was_clipped
                        in zip(self.df[pred_mask].index, predictions, predictions_clipped, clipped_mask)
                        if was_clipped
                    ]
                    print(f"    [clip] Range clipping applied to '{target_col}' "
                          f"(valid range: {valid_min}-{valid_max}): {', '.join(clipped_info)}")

                predictions = predictions_clipped

                # Confidence: based on std deviation across trees
                all_tree_preds = np.array([tree.predict(X_pred) for tree in model.estimators_])
                std_devs = all_tree_preds.std(axis=0)
                value_range = valid_max - valid_min
                if value_range > 0:
                    confidence_pct = np.clip(100 - (std_devs / value_range * 100), 0, 100)
                else:
                    confidence_pct = np.full(len(predictions), 100.0)

                # Detect column type and format predictions accordingly
                col_type = self._detect_column_type(target_col)
                formatted = np.array([self._format_prediction(v, col_type) for v in predictions])
                self.df.loc[pred_mask, target_col] = formatted

                indices = self.df[pred_mask].index.tolist()
                range_label = (f"{int(valid_min)}-{int(valid_max)}"
                               if col_type in ('boolean','rating','integer')
                               else f"{valid_min}-{valid_max}")
                conf_summary = ", ".join(
                    f"row {idx}: {self._format_display(val, col_type)} ({conf:.0f}% conf)"
                    for idx, val, conf in zip(indices, formatted, confidence_pct)
                )
                self._log(target_col,
                    f"ML (RandomForestRegressor) imputed {pred_mask.sum()} value(s) "
                    f"[type: {col_type}, range: {range_label}] -> {conf_summary}")

            else:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_train.astype(str))
                model = RandomForestClassifier(n_estimators=self._get_n_estimators(), random_state=42)
                model.fit(X_train, y_encoded)
                pred_encoded = model.predict(X_pred)
                pred_proba = model.predict_proba(X_pred)

                confidence_pct = pred_proba.max(axis=1) * 100
                predictions = le.inverse_transform(pred_encoded)
                self.df.loc[pred_mask, target_col] = predictions

                indices = self.df[pred_mask].index.tolist()
                conf_summary = ", ".join(
                    f"row {idx}: '{val}' ({conf:.0f}% conf)"
                    for idx, val, conf in zip(indices, predictions, confidence_pct)
                )
                self._log(target_col,
                    f"ML (RandomForestClassifier) imputed {pred_mask.sum()} value(s) -> {conf_summary}")

        except Exception as e:
            print(f"    ! ML imputation failed for '{target_col}': {e}. Using statistical fallback.")
            self._statistical_impute(target_col, is_numeric)

    def _get_usable_features(self, exclude_col: str) -> list:
        return [
            c for c in self.df.columns
            if c != exclude_col
            and self.df[c].notna().all()
            and not str(self.df[c].dtype).startswith("datetime")
        ]

    def _prepare_features(self, feature_cols, train_mask, pred_mask):
        X = self.df[feature_cols].copy()
        for col in X.select_dtypes(include="object").columns:
            if col not in self._label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders[col]
                X[col] = X[col].astype(str).map(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )
        return X[train_mask].values, X[pred_mask].values

    def _placeholder_impute(self, col: str):
        count = self.df[col].isnull().sum()
        col_lower = col.lower()

        # Determine the reason and best placeholder
        if pd.api.types.is_numeric_dtype(self.df[col]):
            # Numeric identity column (timestamp, transaction ID etc.)
            # Use NaN — do not invent fake IDs or timestamps
            reason = "high-cardinality numeric (timestamp/ID) - leaving as NaN"
            # Leave as NaN, do not fill
            self._log(col, f"Skipped imputation for {count} value(s) [{reason}]")
            return

        def make_placeholder(idx):
            if "email" in col_lower or "mail" in col_lower:
                return f"unknown_{idx}@unknown.com"
            elif "name" in col_lower:
                return f"Unknown_{idx}"
            elif "phone" in col_lower or "mobile" in col_lower:
                return "N/A"
            else:
                return f"Unknown_{idx}"

        for idx in self.df[self.df[col].isnull()].index:
            self.df.at[idx, col] = make_placeholder(idx)

        self._log(col, f"Placeholder imputed {count} value(s) "
                       f"[identity column - ML prediction not appropriate]")

    def _statistical_impute(self, col: str, is_numeric: bool):
        if is_numeric:
            fill_value = self.df[col].median()
            strategy = "median"
        else:
            mode = self.df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            strategy = "mode"

        count = self.df[col].isnull().sum()
        self.df[col] = self.df[col].fillna(fill_value)
        self._log(col, f"Statistical ({strategy}) imputed {count} value(s) -> fill: {fill_value}")

    def _log(self, column: str, message: str):
        self.log.append({"column": column, "action": message})
        print(f"  [OK] {column}: {message}")

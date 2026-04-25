"""
Module: data_profiler.py
Profiles the dataset to identify quality issues before cleaning.
Now includes:
  - Column type inference report (numeric, categorical, date, id)
  - High-missing column warnings (>60% missing = suggest dropping)
"""

import pandas as pd
import numpy as np


class DataProfiler:
    """Analyzes a DataFrame and returns a comprehensive quality profile."""

    HIGH_MISSING_THRESHOLD = 0.60  # Warn if >60% of a column is missing

    # Same ID keywords as cleaning engine for consistency
    ID_KEYWORDS = ["id", "_id", "uuid", "guid", "key", "ref", "serial"]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.profile = {}
        self.inferred_types = {}       # col -> 'numeric' | 'categorical' | 'date' | 'id'
        self.high_missing_cols = []    # cols where >60% values are missing

    def run(self) -> dict:
        print("\n[Profiler] Running dataset profiling...")

        self._infer_column_types()

        self.profile = {
            "shape": self._get_shape(),
            "dtypes": self._get_dtypes(),
            "missing": self._get_missing(),
            "duplicates": self._get_duplicates(),
            "numeric_stats": self._get_numeric_stats(),
            "categorical_stats": self._get_categorical_stats(),
            "inferred_types": self.inferred_types,
            "high_missing_cols": self.high_missing_cols,
        }

        self._print_summary()
        self._print_type_inference()
        self._print_high_missing_warnings()
        return self.profile

    # ------------------------------------------------------------------ #
    #  Column Type Inference                                               #
    # ------------------------------------------------------------------ #
    def _infer_column_types(self):
        """
        Infer the semantic type of each column:
          - id         : identifier column (should be skipped in processing)
          - numeric    : numbers (int or float)
          - date       : datetime-like values
          - categorical: text with limited unique values
        """
        for col in self.df.columns:
            col_lower = col.lower()

            # Check ID first
            is_id = (col_lower == "id") or any(
                col_lower.endswith("_" + kw) or col_lower.startswith(kw + "_")
                for kw in self.ID_KEYWORDS
            ) or (self.df[col].nunique() == len(self.df))

            if is_id:
                self.inferred_types[col] = "id"
                continue

            # Check numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if it is a high-cardinality numeric (timestamp/ID)
                notna_vals = self.df[col].dropna()
                if len(notna_vals) > 0:
                    unique_ratio = notna_vals.nunique() / len(notna_vals)
                    median_val = notna_vals.median()
                    if unique_ratio > 0.5 and abs(median_val) > 100000:
                        self.inferred_types[col] = "id"
                        continue
                self.inferred_types[col] = "numeric"
                continue

            # Check date (by name or by trying to parse)
            is_date_name = any(k in col_lower for k in ["date", "time", "dob", "birth", "joined", "created"])
            if is_date_name:
                self.inferred_types[col] = "date"
                continue

            # Try parsing a sample as dates
            sample = self.df[col].dropna().head(10)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().sum() >= len(sample) * 0.7:
                    self.inferred_types[col] = "date"
                    continue

            # Everything else is categorical
            self.inferred_types[col] = "categorical"

    # ------------------------------------------------------------------ #
    #  High-Missing Warning                                                #
    # ------------------------------------------------------------------ #
    def _get_missing(self) -> dict:
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)

        result = {}
        for col in self.df.columns:
            if missing_counts[col] > 0:
                pct = float(missing_pct[col])
                result[col] = {"count": int(missing_counts[col]), "percent": pct}
                if pct / 100 > self.HIGH_MISSING_THRESHOLD:
                    self.high_missing_cols.append(col)
        return result

    # ------------------------------------------------------------------ #
    #  Standard profile methods                                            #
    # ------------------------------------------------------------------ #
    def _get_shape(self) -> dict:
        rows, cols = self.df.shape
        return {"rows": rows, "columns": cols}

    def _get_dtypes(self) -> dict:
        return self.df.dtypes.astype(str).to_dict()

    def _get_duplicates(self) -> dict:
        n_dupes = int(self.df.duplicated().sum())
        dupe_rows = self.df[self.df.duplicated(keep=False)].index.tolist()
        return {"count": n_dupes, "row_indices": dupe_rows}

    def _get_numeric_stats(self) -> dict:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        stats = {}
        for col in numeric_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            stats[col] = {
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "q1": round(float(s.quantile(0.25)), 2),
                "q3": round(float(s.quantile(0.75)), 2),
            }
        return stats

    def _get_categorical_stats(self) -> dict:
        cat_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        stats = {}
        for col in cat_cols:
            vc = self.df[col].value_counts()
            stats[col] = {
                "unique_values": int(self.df[col].nunique()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
            }
        return stats

    # ------------------------------------------------------------------ #
    #  Print helpers                                                        #
    # ------------------------------------------------------------------ #
    def _print_summary(self):
        p = self.profile
        print(f"  -> Shape       : {p['shape']['rows']} rows x {p['shape']['columns']} columns")
        print(f"  -> Missing cols: {len(p['missing'])} column(s) have missing values")
        print(f"  -> Duplicates  : {p['duplicates']['count']} duplicate row(s) found")
        print(f"  -> Numeric cols: {len(p['numeric_stats'])}")
        print(f"  -> Text cols   : {len(p['categorical_stats'])}")

    def _print_type_inference(self):
        """Print a clean table showing the inferred type of each column."""
        print("\n  [Profiler] Column Type Inference:")
        print("  " + "-" * 45)
        print(f"  {'COLUMN':<20} {'INFERRED TYPE':<15} {'MISSING'}")
        print("  " + "-" * 45)
        missing = self.profile.get("missing", {})
        for col, ctype in self.inferred_types.items():
            miss_info = f"{missing[col]['percent']}%" if col in missing else "None"
            type_label = {
                "id": "ID (skipped)",
                "numeric": "Numeric",
                "date": "Date/Time",
                "categorical": "Categorical",
            }.get(ctype, ctype)
            print(f"  {col:<20} {type_label:<15} {miss_info}")
        print("  " + "-" * 45)

    def _print_high_missing_warnings(self):
        """Warn about columns with too many missing values."""
        if not self.high_missing_cols:
            return
        print("\n  [WARNING] The following columns have >60% missing values.")
        print("  ML imputation is unreliable for these. Consider dropping them:")
        for col in self.high_missing_cols:
            pct = self.profile["missing"][col]["percent"]
            print(f"    ! '{col}' is {pct}% empty — consider dropping this column")

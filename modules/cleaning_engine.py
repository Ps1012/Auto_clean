"""
Module: cleaning_engine.py
Applies rule-based cleaning strategies:
  - ID column detection and skipping
  - Numeric type coercion
  - Whitespace stripping
  - Text standardization
  - Date/time validation
  - Negative value detection for logically positive columns
  - Duplicate removal
"""

import pandas as pd
import numpy as np


class CleaningEngine:
    ID_KEYWORDS = ["id", "_id", "uuid", "guid", "key", "ref", "serial"]
    NON_NEGATIVE_KEYWORDS = [
        "age", "salary", "price", "cost", "amount", "quantity",
        "count", "score", "weight", "height", "income", "revenue",
        "population", "distance", "duration", "marks", "grade"
    ]

    def __init__(self, df):
        self.df = df.copy()
        self.log = []
        self.id_columns = []
        self.date_columns = []
        self.column_summary = {}

    def run(self):
        print("\n[Cleaner] Running rule-based cleaning...")
        self._detect_id_columns()
        self._coerce_numeric_columns()
        self._strip_whitespace()
        self._standardize_text_columns()
        self._validate_dates()
        self._check_negative_values()
        self._remove_duplicates()
        self._build_column_summary()
        print(f"  -> {len(self.log)} cleaning action(s) applied.")
        return self.df

    def get_log(self):
        return self.log

    def get_column_summary(self):
        return self.column_summary

    def get_id_columns(self):
        return self.id_columns

    def _detect_id_columns(self):
        for col in self.df.columns:
            col_lower = col.lower()
            name_match = (col_lower == "id") or any(
                col_lower.endswith("_" + kw) or col_lower.startswith(kw + "_")
                for kw in self.ID_KEYWORDS
            )
            all_unique = self.df[col].nunique() == len(self.df)
            if name_match or all_unique:
                self.id_columns.append(col)
        if self.id_columns:
            print(f"  [i] ID columns detected (will be skipped): {self.id_columns}")

    def _is_id_column(self, col):
        return col in self.id_columns

    def _coerce_numeric_columns(self):
        for col in self.df.columns:
            if self._is_id_column(col):
                continue
            if self.df[col].dtype == object:
                converted = pd.to_numeric(self.df[col], errors="coerce")
                success_rate = converted.notna().sum() / len(converted)
                if success_rate > 0.6 and converted.notna().sum() > self.df[col].notna().sum() * 0.5:
                    bad_vals = self.df.loc[self.df[col].notna() & converted.isna(), col].tolist()
                    if bad_vals:
                        self._log(col, f"Coerced to numeric; set {bad_vals} -> NaN")
                        self.df[col] = converted

    def _strip_whitespace(self):
        for col in self.df.select_dtypes(include="object").columns:
            if self._is_id_column(col):
                continue
            before = self.df[col].copy()
            self.df[col] = self.df[col].str.strip()
            self.df[col] = self.df[col].replace("", np.nan)
            changed = (before != self.df[col]).sum()
            if changed:
                self._log(col, f"Stripped whitespace / empty strings -> NaN ({changed} cell(s))")

    def _standardize_text_columns(self):
        for col in self.df.select_dtypes(include="object").columns:
            if self._is_id_column(col):
                continue
            col_lower = col.lower()
            if any(k in col_lower for k in ["name", "department", "category", "city"]):
                self.df[col] = self.df[col].str.title()
                self._log(col, "Applied title-case standardization")
            elif "email" in col_lower or "mail" in col_lower:
                self.df[col] = self.df[col].str.lower()
                self._log(col, "Applied lowercase standardization for email")

    def _validate_dates(self):
        for col in self.df.select_dtypes(include="object").columns:
            if self._is_id_column(col):
                continue
            col_lower = col.lower()
            is_date_col = any(k in col_lower for k in ["date", "time", "dob", "birth", "joined", "created"])
            if not is_date_col:
                sample = self.df[col].dropna().head(10)
                parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
                if len(sample) > 0 and parsed_sample.notna().sum() >= len(sample) * 0.7:
                    is_date_col = True
            if is_date_col:
                self.date_columns.append(col)
                original_notna = int(self.df[col].notna().sum())
                parsed = pd.to_datetime(self.df[col], errors="coerce", format="mixed")
                invalid_count = original_notna - int(parsed.notna().sum())
                if invalid_count > 0:
                    self._log(col, f"Date validation: {invalid_count} invalid date(s) -> NaN")
                self.df[col] = parsed
                future_mask = parsed > pd.Timestamp("2100-01-01")
                future_count = int(future_mask.sum())
                if future_count > 0:
                    self.df.loc[future_mask, col] = pd.NaT
                    self._log(col, f"Removed {future_count} unrealistic future date(s) -> NaT")

    def _check_negative_values(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self._is_id_column(col):
                continue
            col_lower = col.lower()
            if any(k in col_lower for k in self.NON_NEGATIVE_KEYWORDS):
                neg_mask = self.df[col] < 0
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    neg_vals = self.df.loc[neg_mask, col].tolist()
                    self.df.loc[neg_mask, col] = np.nan
                    self._log(col, f"Removed {neg_count} impossible negative value(s) {neg_vals} -> NaN")

    def _remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            self._log("ALL", f"Removed {removed} exact duplicate row(s)")

    def _build_column_summary(self):
        for col in self.df.columns:
            actions = [e["action"] for e in self.log if e["column"] == col]
            if col in self.id_columns:
                self.column_summary[col] = "Skipped (ID column)"
            elif actions:
                self.column_summary[col] = " | ".join(actions)
            else:
                self.column_summary[col] = "No issues found"

    def _log(self, column, message):
        self.log.append({"column": column, "action": message})
        print(f"  [OK] {column}: {message}")

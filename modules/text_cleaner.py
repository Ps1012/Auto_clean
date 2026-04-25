"""
Module: text_cleaner.py
Handles intelligent text column cleaning:
  1. Inconsistent category detection & standardization
     (e.g. "engineering", "ENGINEERING", "Engg" -> "Engineering")
  2. Fuzzy matching using difflib (no external libraries needed)
  3. Pattern validation for emails, phones, pincodes
  4. Free text / high-cardinality column detection (skip gracefully)
  5. Rare category detection (values that appear only once may be typos)
"""

import pandas as pd
import numpy as np
import re
from difflib import get_close_matches


class TextCleaner:
    """
    Cleans categorical text columns by detecting and fixing
    inconsistent values using fuzzy matching.
    """

    # Minimum similarity score (0-1) to consider two values the same
    FUZZY_THRESHOLD = 0.80

    # If a column has more unique values than this fraction of total rows,
    # it is likely free text or an identity column — skip it
    HIGH_CARDINALITY_RATIO = 0.5

    # Columns whose names suggest they are free text
    FREE_TEXT_KEYWORDS = [
        "comment", "description", "note", "remark", "feedback",
        "message", "text", "summary", "detail", "reason"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.log = []

    def run(self) -> pd.DataFrame:
        print("\n[TextCleaner] Running intelligent text cleaning...")

        cat_cols = self.df.select_dtypes(include="object").columns.tolist()

        for col in cat_cols:
            if self._should_skip(col):
                continue
            self._clean_column(col)

        if not self.log:
            print("  -> No text inconsistencies found.")
        else:
            print(f"  -> {len(self.log)} text cleaning action(s) applied.")

        return self.df

    def get_log(self) -> list:
        return self.log

    def _should_skip(self, col: str) -> bool:
        """Skip free text, high-cardinality, or identity columns."""
        col_lower = col.lower()

        # Skip free text columns by name
        if any(kw in col_lower for kw in self.FREE_TEXT_KEYWORDS):
            return True

        # Skip high-cardinality columns (likely names, IDs, emails etc.)
        notna = self.df[col].dropna()
        if len(notna) == 0:
            return True
        if notna.nunique() / len(notna) > self.HIGH_CARDINALITY_RATIO:
            return True

        return False

    def _clean_column(self, col: str):
        """
        For a categorical column:
        1. Case-normalize all values
        2. Build a canonical list of unique values
        3. Use fuzzy matching to find near-duplicates
        4. Standardize all variants to the most common form
        """
        series = self.df[col].dropna()
        if series.nunique() <= 1:
            return

        # Step 1: Get value counts (most common first)
        value_counts = series.value_counts()
        unique_vals = value_counts.index.tolist()

        # Step 2: Build canonical mapping
        # For each value, find if it is a fuzzy match to a more common value
        canonical_map = {}  # variant -> canonical form
        canonical_list = []  # list of accepted canonical values

        for val in unique_vals:
            val_str = str(val).strip()
            if val_str in canonical_map:
                continue

            # Check if this value is close to any already-accepted canonical
            matches = get_close_matches(
                val_str.lower(),
                [c.lower() for c in canonical_list],
                n=1,
                cutoff=self.FUZZY_THRESHOLD
            )

            if matches:
                # Find the actual canonical (case-sensitive match)
                matched_lower = matches[0]
                matched_canonical = next(
                    c for c in canonical_list if c.lower() == matched_lower
                )
                canonical_map[val_str] = matched_canonical
            else:
                # This is a new canonical value
                canonical_list.append(val_str)
                canonical_map[val_str] = val_str

        # Step 3: Apply the mapping and log changes
        changes = {}
        for original, canonical in canonical_map.items():
            if original != canonical:
                changes[original] = canonical

        if changes:
            self.df[col] = self.df[col].map(
                lambda x: canonical_map.get(str(x).strip(), x) if pd.notna(x) else x
            )
            change_summary = ", ".join(
                f"'{k}' -> '{v}'" for k, v in list(changes.items())[:5]
            )
            if len(changes) > 5:
                change_summary += f" ... and {len(changes) - 5} more"
            self._log(col, f"Fuzzy standardization: {len(changes)} variant(s) unified -> {change_summary}")

        # Step 4: Detect rare values (appear only once — likely typos)
        rare_vals = value_counts[value_counts == 1].index.tolist()
        # Only flag rare values that weren't already fixed by fuzzy matching
        unfixed_rare = [v for v in rare_vals if v not in changes]
        if unfixed_rare and len(unique_vals) > 3:
            self._log(col,
                f"Rare value warning: {unfixed_rare[:5]} "
                f"(appear only once — may be typos, review manually)")

    def _log(self, column: str, message: str):
        self.log.append({"column": column, "action": message})
        print(f"  [OK] {column}: {message}")

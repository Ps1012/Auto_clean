"""
AutoClean AI -- Main Pipeline
Usage:
    python main.py --input sample_data/dirty_data.csv
    python main.py --input your_file.csv --anomaly-strategy remove
    python main.py --input your_file.csv --output cleaned_output.csv
    python main.py --input your_file.csv --dry-run
"""

import argparse
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from modules.data_profiler import DataProfiler
from modules.cleaning_engine import CleaningEngine
from modules.anomaly_detector import AnomalyDetector
from modules.ml_imputer import MLImputer
from modules.report_generator import ReportGenerator


BANNER = """
+======================================================+
|          AUTO CLEAN AI  --  Intelligent Data         |
|          Cleaning & Anomaly Detection System         |
+======================================================+
"""


def _print_column_summary(column_summary: dict):
    print("\n" + "-" * 65)
    print(f"  {'COLUMN':<20} {'ACTION TAKEN':<43}")
    print("-" * 65)
    for col, action in column_summary.items():
        display = action if len(action) <= 43 else action[:40] + "..."
        print(f"  {col:<20} {display:<43}")
    print("-" * 65)


def _compute_quality_score(df: pd.DataFrame) -> float:
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 100.0
    missing = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())
    return round(max(0.0, 100.0 - ((missing + dupes) / total_cells * 100)), 1)


def _run_dry_run(input_path: str):
    print(BANNER)
    print("  ** DRY-RUN MODE -- No files will be modified **\n")
    print(f"[Dry-Run] Loading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"  ERROR: File not found: {input_path}")
        sys.exit(1)

    print(f"  -> Loaded {df.shape[0]} rows x {df.shape[1]} columns\n")

    profiler = DataProfiler(df)
    profile = profiler.run()

    cleaner = CleaningEngine(df)
    df_cleaned = cleaner.run()

    missing_after_clean = df_cleaned.isnull().sum()
    imputable = {c: int(missing_after_clean[c]) for c in df_cleaned.columns if missing_after_clean[c] > 0}
    score_before = _compute_quality_score(df)

    print("\n" + "=" * 55)
    print("  DRY-RUN SUMMARY")
    print("=" * 55)
    print(f"  Rows              : {df.shape[0]}")
    print(f"  Columns           : {df.shape[1]}")
    print(f"  Missing values    : {int(df.isnull().sum().sum())} would be filled")
    print(f"  Duplicates        : {int(df.duplicated().sum())} would be removed")
    print(f"  Anomaly detection : runs BEFORE imputation (safe order)")
    print(f"  Range clipping    : ML predictions clipped to observed data range")
    print(f"  Current Quality   : {score_before}%")

    if imputable:
        print(f"\n  Columns needing imputation after cleaning:")
        for col, count in imputable.items():
            print(f"    - '{col}': {count} missing value(s)")

    if profile.get("high_missing_cols"):
        print(f"\n  Columns with >60% missing (risky to impute):")
        for col in profile["high_missing_cols"]:
            pct = profile["missing"][col]["percent"]
            print(f"    ! '{col}': {pct}% empty")

    print("\n  No files were modified. Run without --dry-run to apply changes.")
    print("=" * 55)


def run_pipeline(input_path: str,
                 output_path: str = None,
                 anomaly_strategy: str = "flag",
                 report_dir: str = "reports"):

    print(BANNER)

    # 1. Load Data
    print(f"[Pipeline] Loading data from: {input_path}")
    try:
        df_original = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"  ERROR: File not found: {input_path}")
        sys.exit(1)

    print(f"  -> Loaded {df_original.shape[0]} rows x {df_original.shape[1]} columns\n")
    df = df_original.copy()
    score_before = _compute_quality_score(df_original)

    # 2. Profile
    profiler = DataProfiler(df)
    profile = profiler.run()

    # 3. Rule-Based Cleaning
    cleaner = CleaningEngine(df)
    df = cleaner.run()
    cleaning_log = cleaner.get_log()
    column_summary = cleaner.get_column_summary()

    # 4. Anomaly Detection BEFORE imputation
    #    Reason: detect outliers on real observed data first so the
    #    ML imputer learns the valid range from clean rows only.
    print("\n[Pipeline] Running anomaly detection BEFORE imputation...")
    print("  (This ensures ML imputer learns valid ranges from clean data only)")
    detector_pre = AnomalyDetector(df, contamination=0.05)
    df = detector_pre.run(strategy="flag")  # always flag at this stage
    anomaly_report = detector_pre.get_report()

    # 6. ML Imputation (now trains on clean, non-anomalous rows)
    #    The range clipping inside MLImputer uses y_train min/max,
    #    which now excludes flagged anomaly rows if desired.
    imputer = MLImputer(df)
    df = imputer.run()
    cleaning_log += imputer.get_log()

    for entry in imputer.get_log():
        col = entry["column"]
        if col in column_summary:
            column_summary[col] += " | " + entry["action"]
        else:
            column_summary[col] = entry["action"]

    # 7. Remove the temporary is_anomaly column added in step 4
    #    then apply final anomaly strategy on the fully cleaned data
    if "is_anomaly" in df.columns:
        df = df.drop(columns=["is_anomaly"])

    print("\n[Pipeline] Re-running anomaly detection on fully cleaned data...")
    detector_final = AnomalyDetector(df, contamination=0.05)
    df = detector_final.run(strategy=anomaly_strategy)
    anomaly_report = detector_final.get_report()

    score_after = _compute_quality_score(df)
    improvement = round(score_after - score_before, 1)

    # 8. Save Cleaned Data
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base}_cleaned.csv")

    df.to_csv(output_path, index=False)
    print(f"\n[Pipeline] Cleaned data saved -> {output_path}")

    # 9. Generate Report
    reporter = ReportGenerator(
        df_before=df_original,
        df_after=df,
        anomaly_report=anomaly_report,
        cleaning_log=cleaning_log,
        score_before=score_before,
        score_after=score_after,
        output_dir=report_dir
    )
    chart_path, summary_path = reporter.generate()

    # 10. Column-Level Summary
    print("\n[Pipeline] Column-Level Summary:")
    _print_column_summary(column_summary)

    # 11. Final Summary
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Original rows         : {df_original.shape[0]}")
    print(f"  Cleaned rows          : {df.shape[0]}")
    print(f"  Missing (before)      : {int(df_original.isnull().sum().sum())}")
    print(f"  Missing (after)       : {int(df.isnull().sum().sum())}")
    print(f"  Duplicates fixed      : {int(df_original.duplicated().sum())}")
    print(f"  Anomalies found       : {anomaly_report.get('consensus_count', 0)}")
    print(f"  Data Quality Score    : {score_before}% -> {score_after}%  (+{improvement}% improvement)")
    print(f"  Output CSV            : {output_path}")
    print(f"  Visual Report         : {chart_path}")
    print(f"  Text Summary          : {summary_path}")
    print("=" * 55)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="AutoClean AI -- Intelligent Data Cleaning & Anomaly Detection"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", default=None, help="Path for cleaned CSV output")
    parser.add_argument("--anomaly-strategy", "-a", choices=["flag", "remove"],
                        default="flag", help="'flag' or 'remove' anomalies. Default: flag")
    parser.add_argument("--report-dir", "-r", default="reports",
                        help="Directory to save reports (default: reports/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be cleaned without modifying any files")

    args = parser.parse_args()

    if args.dry_run:
        _run_dry_run(args.input)
    else:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            anomaly_strategy=args.anomaly_strategy,
            report_dir=args.report_dir
        )


if __name__ == "__main__":
    main()

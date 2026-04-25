"""
AutoClean AI — Flask Web Server
Connects the Python pipeline to the browser UI.
Run with: python app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import queue
import threading
import io
import base64
from datetime import datetime

from flask import Flask, request, jsonify, send_file, Response
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from modules.data_profiler import DataProfiler
from modules.cleaning_engine import CleaningEngine
from modules.anomaly_detector import AnomalyDetector
from modules.ml_imputer import MLImputer
from modules.report_generator import ReportGenerator

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Global store for latest results
latest_result = {}


def compute_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 100.0
    missing = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())
    return round(max(0.0, 100.0 - ((missing + dupes) / total_cells * 100)), 1)


def run_pipeline_stream(filepath, anomaly_strategy, result_queue):
    """Run the full pipeline and push progress events into result_queue."""
    def emit(event, data):
        result_queue.put({"event": event, "data": data})

    try:
        emit("progress", {"step": "loading", "message": "Loading dataset...", "pct": 5})
        df_original = pd.read_csv(filepath)
        rows, cols = df_original.shape
        emit("progress", {"step": "loaded", "message": f"Loaded {rows} rows x {cols} columns", "pct": 10})

        score_before = compute_quality_score(df_original)
        df = df_original.copy()

        # Profiling
        emit("progress", {"step": "profiling", "message": "Profiling dataset...", "pct": 20})
        profiler = DataProfiler(df)
        profile = profiler.run()

        # Cleaning
        emit("progress", {"step": "cleaning", "message": "Running rule-based cleaning...", "pct": 35})
        cleaner = CleaningEngine(df)
        df = cleaner.run()
        cleaning_log = cleaner.get_log()
        column_summary = cleaner.get_column_summary()

        # Anomaly Detection BEFORE imputation
        emit("progress", {"step": "anomaly", "message": "Detecting anomalies (pre-imputation)...", "pct": 50})
        detector_pre = AnomalyDetector(df, contamination=0.05)
        df = detector_pre.run(strategy="flag")

        # Imputing (trains only on clean range)
        emit("progress", {"step": "imputing", "message": "Running ML imputation with range clipping...", "pct": 65})
        imputer = MLImputer(df)
        df = imputer.run()
        cleaning_log += imputer.get_log()
        for entry in imputer.get_log():
            col = entry["column"]
            column_summary[col] = column_summary.get(col, "") + " | " + entry["action"]

        # Remove temp anomaly col, re-run final anomaly detection
        if "is_anomaly" in df.columns:
            df = df.drop(columns=["is_anomaly"])
        emit("progress", {"step": "anomaly2", "message": "Final anomaly detection on clean data...", "pct": 78})
        detector = AnomalyDetector(df, contamination=0.05)
        df = detector.run(strategy=anomaly_strategy)
        anomaly_report = detector.get_report()

        # Save cleaned CSV
        emit("progress", {"step": "saving", "message": "Saving cleaned data...", "pct": 85})
        base = os.path.splitext(os.path.basename(filepath))[0]
        cleaned_path = os.path.join(UPLOAD_FOLDER, f"{base}_cleaned.csv")
        df.to_csv(cleaned_path, index=False)

        # Generate report
        emit("progress", {"step": "report", "message": "Generating visual report...", "pct": 92})
        score_after = compute_quality_score(df)
        reporter = ReportGenerator(
            df_before=df_original,
            df_after=df,
            anomaly_report=anomaly_report,
            cleaning_log=cleaning_log,
            score_before=score_before,
            score_after=score_after,
            output_dir=REPORTS_FOLDER
        )
        chart_path, summary_path = reporter.generate()

        # Read chart as base64 for sending to browser
        with open(chart_path, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Build result payload
        result = {
            "rows_before": rows,
            "rows_after": int(df.shape[0]),
            "cols": cols,
            "missing_before": int(df_original.isnull().sum().sum()),
            "missing_after": int(df.isnull().sum().sum()),
            "duplicates_fixed": int(df_original.duplicated().sum()),
            "anomalies_found": anomaly_report.get("consensus_count", 0),
            "score_before": score_before,
            "score_after": score_after,
            "improvement": round(score_after - score_before, 1),
            "column_summary": column_summary,
            "cleaning_log": [e["action"] for e in cleaning_log],
            "inferred_types": profile.get("inferred_types", {}),
            "high_missing_cols": profile.get("high_missing_cols", []),
            "anomaly_report": {
                "zscore": anomaly_report.get("zscore_count", 0),
                "iqr": anomaly_report.get("iqr_count", 0),
                "isolation_forest": anomaly_report.get("isolation_forest_count", 0),
                "consensus": anomaly_report.get("consensus_count", 0),
            },
            "chart_b64": chart_b64,
            "cleaned_filename": os.path.basename(cleaned_path),
        }

        # Store globally for download
        global latest_result
        latest_result = {"cleaned_path": cleaned_path, "chart_path": chart_path}

        emit("progress", {"step": "done", "message": "Pipeline complete!", "pct": 100})
        emit("result", result)

    except Exception as e:
        import traceback
        emit("error", {"message": str(e), "traceback": traceback.format_exc()})


@app.route("/")
def index():
    return send_file("templates/index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    anomaly_strategy = request.form.get("anomaly_strategy", "flag")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result_queue = queue.Queue()

    def generate():
        thread = threading.Thread(
            target=run_pipeline_stream,
            args=(filepath, anomaly_strategy, result_queue)
        )
        thread.start()

        while True:
            try:
                item = result_queue.get(timeout=600)
                yield f"data: {json.dumps(item)}\n\n"
                if item["event"] in ("result", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Pipeline timed out. Try a smaller dataset or use --remove for anomalies.'}})}\n\n"
                break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    print("\n  Auto Clean AI — Web UI")
    print("  Open your browser at: http://localhost:5000\n")
    app.run(debug=False, port=5000)

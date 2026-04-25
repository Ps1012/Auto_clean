"""
Module: report_generator.py
Generates visual charts and a text summary comparing before/after cleaning.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime


class ReportGenerator:
    """Creates visual reports showing before/after data quality improvements."""

    def __init__(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                 anomaly_report: dict, cleaning_log: list,
                 score_before: float = 0.0, score_after: float = 0.0,
                 output_dir: str = "reports"):
        self.before = df_before
        self.after = df_after
        self.anomaly_report = anomaly_report
        self.cleaning_log = cleaning_log
        self.score_before = score_before
        self.score_after = score_after
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self):
        """Generate all report components."""
        print("\n[Reporter] Generating visual report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        chart_path = os.path.join(self.output_dir, f"quality_report_{timestamp}.png")
        text_path = os.path.join(self.output_dir, f"cleaning_summary_{timestamp}.txt")

        self._create_visual_report(chart_path)
        self._create_text_summary(text_path)

        print(f"  → Chart saved : {chart_path}")
        print(f"  → Summary     : {text_path}")
        return chart_path, text_path

    # ------------------------------------------------------------------ #
    #  Visual Report                                                       #
    # ------------------------------------------------------------------ #
    def _create_visual_report(self, path: str):
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor("#0f1117")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

        title_color = "#e2e8f0"
        accent = "#38bdf8"
        good_color = "#4ade80"
        bad_color = "#f87171"

        fig.suptitle("Auto Clean AI — Data Quality Report",
                     fontsize=18, fontweight="bold",
                     color=title_color, y=0.97)

        # 1. Missing Values Before vs After
        ax1 = fig.add_subplot(gs[0, 0])
        missing_before = self.before.isnull().sum()
        missing_before = missing_before[missing_before > 0]
        missing_after = self.after.isnull().sum().reindex(missing_before.index, fill_value=0)
        x = np.arange(len(missing_before))
        w = 0.35
        ax1.bar(x - w/2, missing_before.values, w, label="Before", color=bad_color, alpha=0.85)
        ax1.bar(x + w/2, missing_after.values, w, label="After", color=good_color, alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(missing_before.index, rotation=30, ha="right", fontsize=8, color=title_color)
        ax1.set_title("Missing Values: Before vs After", color=accent, fontweight="bold")
        ax1.legend(fontsize=8)
        self._style_ax(ax1)

        # 2. Duplicate Rows
        ax2 = fig.add_subplot(gs[0, 1])
        dupes_before = int(self.before.duplicated().sum())
        dupes_after = int(self.after.duplicated().sum())
        bars = ax2.bar(["Before", "After"], [dupes_before, dupes_after],
                       color=[bad_color, good_color], width=0.4, alpha=0.9)
        for bar, val in zip(bars, [dupes_before, dupes_after]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(val), ha="center", va="bottom", color=title_color, fontsize=11, fontweight="bold")
        ax2.set_title("Duplicate Rows", color=accent, fontweight="bold")
        ax2.set_ylim(0, max(dupes_before + 2, 3))
        self._style_ax(ax2)

        # 3. Anomaly Detection Summary
        ax3 = fig.add_subplot(gs[0, 2])
        methods = ["Z-Score", "IQR", "Isolation\nForest", "Consensus"]
        counts = [
            self.anomaly_report.get("zscore_count", 0),
            self.anomaly_report.get("iqr_count", 0),
            self.anomaly_report.get("isolation_forest_count", 0),
            self.anomaly_report.get("consensus_count", 0),
        ]
        colors = ["#fb923c", "#facc15", "#c084fc", "#38bdf8"]
        ax3.barh(methods, counts, color=colors, alpha=0.85)
        for i, v in enumerate(counts):
            ax3.text(v + 0.05, i, str(v), va="center", color=title_color, fontsize=9, fontweight="bold")
        ax3.set_title("Anomalies Detected", color=accent, fontweight="bold")
        self._style_ax(ax3)

        # 4. Numeric Distribution: Age (before vs after, if exists)
        ax4 = fig.add_subplot(gs[1, 0])
        num_col = None
        for c in self.before.select_dtypes(include=np.number).columns:
            if c != "id" and "anomaly" not in c.lower():
                num_col = c
                break
        if num_col:
            self.before[num_col].dropna().plot.hist(ax=ax4, bins=15, color=bad_color,
                                                     alpha=0.6, label="Before", density=True)
            self.after[num_col].dropna().plot.hist(ax=ax4, bins=15, color=good_color,
                                                    alpha=0.6, label="After", density=True)
            ax4.set_title(f"Distribution: '{num_col}'", color=accent, fontweight="bold")
            ax4.legend(fontsize=8)
        self._style_ax(ax4)

        # 5. Overall Quality Score
        ax5 = fig.add_subplot(gs[1, 1])
        total_cells_before = self.before.shape[0] * self.before.shape[1]
        missing_before_total = int(self.before.isnull().sum().sum())
        dupes_before = int(self.before.duplicated().sum())
        issues_before = missing_before_total + dupes_before

        missing_after_total = int(self.after.isnull().sum().sum())
        issues_after = missing_after_total

        score_before = max(0, 100 - (issues_before / total_cells_before * 100))
        score_after = min(100, max(0, 100 - (issues_after / total_cells_before * 100)))

        categories = ["Before", "After"]
        scores = [round(score_before, 1), round(score_after, 1)]
        bars = ax5.bar(categories, scores, color=[bad_color, good_color], width=0.4, alpha=0.9)
        for bar, val in zip(bars, scores):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                     f"{val}%", ha="center", va="top", color="white", fontsize=13, fontweight="bold")
        ax5.set_ylim(0, 110)
        ax5.set_title("Data Quality Score", color=accent, fontweight="bold")
        self._style_ax(ax5)

        # 6. Cleaning Actions Log
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        log_lines = ["Cleaning Actions Log\n" + "─" * 30]
        for entry in self.cleaning_log[:10]:
            col = entry["column"]
            action = entry["action"]
            line = f"• [{col}] {action[:45]}{'...' if len(action) > 45 else ''}"
            log_lines.append(line)
        if len(self.cleaning_log) > 10:
            log_lines.append(f"  ... and {len(self.cleaning_log) - 10} more")
        ax6.text(0.02, 0.95, "\n".join(log_lines),
                 transform=ax6.transAxes, fontsize=7.5, color=title_color,
                 va="top", ha="left", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=accent, alpha=0.9))

        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

    def _style_ax(self, ax):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.yaxis.label.set_color("#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")

    # ------------------------------------------------------------------ #
    #  Text Summary                                                        #
    # ------------------------------------------------------------------ #
    def _create_text_summary(self, path: str):
        lines = [
            "=" * 60,
            "       AUTO CLEAN AI — CLEANING SUMMARY REPORT",
            f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "DATASET OVERVIEW",
            f"  Before : {self.before.shape[0]} rows × {self.before.shape[1]} columns",
            f"  After  : {self.after.shape[0]} rows × {self.after.shape[1]} columns",
            "",
            "ISSUES RESOLVED",
            f"  Missing values  (before): {int(self.before.isnull().sum().sum())}",
            f"  Missing values  (after) : {int(self.after.isnull().sum().sum())}",
            f"  Duplicate rows  (before): {int(self.before.duplicated().sum())}",
            f"  Duplicate rows  (after) : {int(self.after.duplicated().sum())}",
            "",
            "ANOMALY DETECTION",
            f"  Z-Score anomalies      : {self.anomaly_report.get('zscore_count', 0)}",
            f"  IQR anomalies          : {self.anomaly_report.get('iqr_count', 0)}",
            f"  Isolation Forest       : {self.anomaly_report.get('isolation_forest_count', 0)}",
            f"  Consensus anomalies    : {self.anomaly_report.get('consensus_count', 0)}",
            "",
            "CLEANING LOG",
        ]
        for i, entry in enumerate(self.cleaning_log, 1):
            lines.append(f"  {i:2}. [{entry['column']}] {entry['action']}")

        lines += ["", "=" * 60, "END OF REPORT", "=" * 60]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

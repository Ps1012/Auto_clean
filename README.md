# Auto Clean AI
### Intelligent Data Cleaning & Anomaly Detection System

A Python-based automated data preprocessing system that detects, analyzes,
and resolves data quality issues in structured CSV datasets using
statistical analysis and machine learning techniques.

---

## Features

- **ID Column Detection** — Automatically skips identifier columns (id, uuid etc.)
- **Missing Value Imputation** — Uses RandomForest ML to predict missing numeric/categorical values
- **Smart Placeholder Imputation** — Uses placeholder values for identity columns (names, emails) where ML prediction is not appropriate
- **Negative Value Detection** — Flags and removes impossible negative values in columns like age, salary, price
- **Date Validation** — Detects and fixes invalid or unrealistic date values
- **Duplicate Removal** — Detects and removes exact duplicate rows
- **Anomaly Detection** — Three methods: Z-Score, IQR, and Isolation Forest with consensus voting
- **Data Quality Score** — Computes a before/after quality score showing improvement
- **Visual Reports** — Generates before/after comparison charts
- **Column-Level Summary** — Prints a detailed table of actions taken per column

---

## Project Structure

```
autoclean_ai/
├── main.py                    <- Entry point, runs the full pipeline
├── requirements.txt           <- Python dependencies
├── README.md                  <- This file
├── sample_data/
│   ├── dirty_data.csv         <- Sample dataset with intentional issues
│   └── dirty_data_cleaned.csv <- Output after cleaning (auto-generated)
├── modules/
│   ├── data_profiler.py       <- Profiles dataset: missing values, duplicates, stats
│   ├── cleaning_engine.py     <- Rule-based cleaning: types, whitespace, dates, negatives
│   ├── ml_imputer.py          <- ML-based missing value imputation (RandomForest)
│   └── anomaly_detector.py    <- Anomaly detection: Z-Score + IQR + Isolation Forest
│   └── report_generator.py    <- Generates visual charts and text summary
└── reports/                   <- Output reports saved here (auto-created)
```

---

## Installation

### Step 1: Make sure Python is installed
Download from https://python.org (Python 3.10 or higher recommended)

### Step 2: Open a terminal in the project folder
In VS Code: press Ctrl+` to open the terminal

### Step 3: Install dependencies
```
pip install -r requirements.txt
```

---

## Usage

### Run on the sample dataset
```
python main.py --input sample_data/dirty_data.csv
```

### Run on your own CSV file
```
python main.py --input your_file.csv
```

### Remove anomalies instead of flagging them
```
python main.py --input your_file.csv --anomaly-strategy remove
```

### Specify a custom output path
```
python main.py --input your_file.csv --output cleaned.csv --report-dir my_reports
```

---

## Output Files

| File | Description |
|------|-------------|
| `sample_data/*_cleaned.csv` | Cleaned dataset with `is_anomaly` column added |
| `reports/quality_report_*.png` | Visual before/after comparison chart |
| `reports/cleaning_summary_*.txt` | Full text log of all cleaning actions |

---

## How Anomaly Detection Works

The system uses three independent methods and flags a row as anomalous
only if at least 2 out of 3 methods agree (consensus voting).
This reduces false positives.

| Method | How it works |
|--------|-------------|
| Z-Score | Flags values more than 3 standard deviations from the mean |
| IQR | Flags values outside 1.5x the interquartile range |
| Isolation Forest | ML-based unsupervised detection of outlier patterns |

---

## Technologies Used

- Python 3.10+
- pandas — data manipulation
- numpy — numerical operations
- scikit-learn — RandomForest imputation, Isolation Forest anomaly detection
- matplotlib — report visualization

---

## Author
Prabhjot Singh — 6EA

---

## Web UI (Frontend)

You can also run Auto Clean AI as a web app in your browser!

### Step 1: Install Flask
```
pip install flask
```

### Step 2: Start the web server
```
python app.py
```

### Step 3: Open your browser
Go to: **http://localhost:5000**

You'll see a drag-and-drop interface where you can:
- Upload any CSV file
- Watch the pipeline run live with a progress bar
- See results: quality score, column types, anomaly detection, cleaning summary
- Download the cleaned CSV directly from the browser

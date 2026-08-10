"""
AI Workforce Analytics — PostgreSQL Data Loader (Simple Version)

Uses psycopg2 directly (no SQLAlchemy). Three steps:
  1. Connect to the database
  2. Create tables from schema.sql (if they don't exist)
  3. Load the cleaned + feature-engineered CSVs into the tables

Easy to explain in a demo:
  "We connect with psycopg2, run our schema.sql to build the tables,
   then read the CSVs with pandas and bulk-insert the rows."
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

CLEANED_PATH = BASE_DIR / "data" / "processed" / "employee_attrition_cleaned_dataset.csv"
FEATURES_PATH = BASE_DIR / "Machine Learning" / "dataset1_feature_engineered_final.csv"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

# ============================================================
# 2. TABLE COLUMNS (must match schema.sql, minus auto/serial fields)
# ============================================================

EMPLOYEES_COLS = [
    "employee_id", "age", "gender", "marital_status", "education_level",
    "department", "employment_type", "job_level", "monthly_income",
    "years_at_company", "years_in_current_role",
]

METRICS_COLS = [
    "employee_id", "performance_rating", "overall_satisfaction_index",
    "burnout_risk_score", "absence_rate_per_year", "hr_red_flag_count",
    "career_stagnation_flag", "overtime_and_low_satisfaction_flag",
    "long_commute_flag", "high_performer_flag", "training_hours_last_year",
    "attrition",
]

# ============================================================
# 3. CONNECT TO POSTGRES
# ============================================================

def get_connection():
    required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"Missing from .env: {', '.join(missing)}")

    use_ssl = os.environ.get("DB_SSL", "false").lower() == "true"

    conn = psycopg2.connect(
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        sslmode="require" if use_ssl else "prefer",
    )
    return conn

# ============================================================
# 4. APPLY SCHEMA
# ============================================================

def apply_schema(conn):
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.sql not found at:\n{SCHEMA_PATH}")

    # Check if tables already exist (e.g. you ran schema.sql manually in
    # the Supabase SQL Editor) — if so, skip re-applying it through the
    # pooler, which can hit a statement timeout on some connections.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'employees'
            );
        """)
        tables_exist = cur.fetchone()[0]

    if tables_exist:
        print("Tables already exist — skipping schema creation.")
        return

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '120000';")  # 120 seconds
        cur.execute(schema_sql)
    conn.commit()
    print("Schema applied (tables created if they didn't already exist).")

# ============================================================
# 5. BUILD DATAFRAMES FROM CSVs
# ============================================================

def build_tables():
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at:\n{CLEANED_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature-engineered dataset not found at:\n{FEATURES_PATH}")

    cleaned = pd.read_csv(CLEANED_PATH)
    features = pd.read_csv(FEATURES_PATH)

    cleaned.columns = [c.lower().strip() for c in cleaned.columns]
    features.columns = [c.lower().strip() for c in features.columns]

    if "employee_id" not in features.columns:
        features = features.copy()
        features.insert(0, "employee_id", cleaned["employee_id"].values)

    missing_emp = [c for c in EMPLOYEES_COLS if c not in cleaned.columns]
    if missing_emp:
        raise ValueError(f"Missing columns in cleaned dataset: {missing_emp}")

    missing_metrics = [c for c in METRICS_COLS if c not in features.columns]
    if missing_metrics:
        raise ValueError(f"Missing columns in feature dataset: {missing_metrics}")

    employees_df = cleaned[EMPLOYEES_COLS].copy()
    metrics_df = features[METRICS_COLS].copy()

    # Schema defines these as BOOLEAN, but the CSV stores them as 0/1 ints.
    # Cast explicitly so psycopg2 sends real booleans, not integers.
    boolean_cols = [
        "career_stagnation_flag",
        "overtime_and_low_satisfaction_flag",
        "long_commute_flag",
        "high_performer_flag",
        "attrition",
    ]
    for col in boolean_cols:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].astype(bool)

    print(f"Employees rows: {len(employees_df)} | Metrics rows: {len(metrics_df)}")
    return employees_df, metrics_df

# ============================================================
# 6. LOAD DATA (bulk insert with execute_values)
# ============================================================

def load(conn, employees_df, metrics_df):
    with conn.cursor() as cur:
        # Clear old data but keep the table structure
        print("Truncating existing data...")
        cur.execute("TRUNCATE TABLE employee_metrics, employees RESTART IDENTITY CASCADE;")
        conn.commit()
        print("Truncated.\n")

        employees_rows = list(employees_df.itertuples(index=False, name=None))
        print(f"Inserting {len(employees_rows)} rows into employees (in batches of 500)...")
        execute_values(
            cur,
            f"INSERT INTO employees ({', '.join(EMPLOYEES_COLS)}) VALUES %s",
            employees_rows,
            page_size=500,
        )
        conn.commit()
        print(f"Loaded {len(employees_rows)} rows into employees\n")

        metrics_rows = list(metrics_df.itertuples(index=False, name=None))
        print(f"Inserting {len(metrics_rows)} rows into employee_metrics (in batches of 500)...")
        execute_values(
            cur,
            f"INSERT INTO employee_metrics ({', '.join(METRICS_COLS)}) VALUES %s",
            metrics_rows,
            page_size=500,
        )
        conn.commit()
        print(f"Loaded {len(metrics_rows)} rows into employee_metrics")

    print("Data committed successfully.")

# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("Connecting to PostgreSQL...")
    conn = get_connection()
    print("Connected!\n")

    apply_schema(conn)
    employees_df, metrics_df = build_tables()
    load(conn, employees_df, metrics_df)

    conn.close()
    print("\nDone. Connection closed.")


if __name__ == "__main__":
    main()
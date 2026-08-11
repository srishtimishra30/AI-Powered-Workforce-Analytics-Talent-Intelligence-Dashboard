CREATE TABLE IF NOT EXISTS employees (
    employee_id            INTEGER PRIMARY KEY,
    age                     SMALLINT,
    gender                  VARCHAR(20),
    marital_status          VARCHAR(20),
    education_level         VARCHAR(30),
    department              VARCHAR(50),
    employment_type         VARCHAR(20),
    job_level               SMALLINT,
    monthly_income           NUMERIC(12,2),
    years_at_company        SMALLINT,
    years_in_current_role   SMALLINT,
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS employee_metrics (
    metric_id                          SERIAL PRIMARY KEY,
    employee_id                        INTEGER REFERENCES employees(employee_id) ON DELETE CASCADE,
    performance_rating                 SMALLINT,
    overall_satisfaction_index         NUMERIC(6,3),
    burnout_risk_score                 NUMERIC(6,3),
    absence_rate_per_year              NUMERIC(6,3),
    hr_red_flag_count                  SMALLINT,
    is_new_hire                        BOOLEAN,
    career_stagnation_flag             BOOLEAN,
    overtime_and_low_satisfaction_flag BOOLEAN,
    long_commute_flag                  BOOLEAN,
    high_performer_flag                BOOLEAN,
    training_hours_last_year           SMALLINT,
    attrition                          BOOLEAN,
    recorded_at                        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id           SERIAL PRIMARY KEY,
    employee_id              INTEGER REFERENCES employees(employee_id) ON DELETE CASCADE,
    model_name               VARCHAR(50),      
    model_version             VARCHAR(20),
    attrition_probability     NUMERIC(5,4),        
    predicted_label           BOOLEAN,
    predicted_at              TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department);
CREATE INDEX IF NOT EXISTS idx_metrics_attrition ON employee_metrics(attrition);
CREATE INDEX IF NOT EXISTS idx_metrics_employee_id ON employee_metrics(employee_id);
CREATE INDEX IF NOT EXISTS idx_predictions_employee_id ON predictions(employee_id);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON predictions(attrition_probability DESC);

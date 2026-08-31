import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "feature_engineered_final.csv")
ATTRITION_MODEL_PATH = os.path.join(BASE_DIR, "models", "attrition_model.pkl")
SKILL_MODEL_PATH = os.path.join(BASE_DIR, "models", "skill_gap_model.pkl")

app = Flask(__name__)

# Load Only Feature-Engineered Dataset
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Feature dataset not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

# Add Synthetic/Derived Display Columns from One-Hot & Engineered Columns
if "employee_id" not in df.columns:
    df.insert(0, "employee_id", np.arange(200001, 200001 + len(df)))

def extract_department(row):
    if row.get("department_HR", 0) == 1: return "HR"
    elif row.get("department_IT", 0) == 1: return "IT"
    elif row.get("department_Marketing", 0) == 1: return "Marketing"
    elif row.get("department_Operations", 0) == 1: return "Operations"
    elif row.get("department_Sales", 0) == 1: return "Sales"
    return "Engineering"

df["department"] = df.apply(extract_department, axis=1)

ROLE_MAP = {
    "IT": {1: "Junior Software Engineer", 2: "Software Engineer", 3: "Senior Developer", 4: "IT Manager", 5: "IT Director"},
    "Marketing": {1: "Marketing Associate", 2: "SEO Specialist", 3: "Content Strategist", 4: "Marketing Manager", 5: "VP of Marketing"},
    "HR": {1: "HR Assistant", 2: "HR Executive", 3: "Talent Specialist", 4: "HR Business Partner", 5: "Head of People"},
    "Operations": {1: "Operations Assistant", 2: "Quality Analyst", 3: "Logistics Coordinator", 4: "Operations Manager", 5: "Director of Operations"},
    "Sales": {1: "Sales Representative", 2: "Account Executive", 3: "Business Development", 4: "Sales Manager", 5: "Chief Revenue Officer"},
    "Engineering": {1: "Junior Cloud Engineer", 2: "Full-Stack Engineer", 3: "Senior Cloud Architect", 4: "Engineering Lead", 5: "VP of Engineering"}
}

def extract_job_role(row):
    dept = row["department"]
    lvl = int(np.clip(row.get("job_level", 2), 1, 5))
    return ROLE_MAP.get(dept, {}).get(lvl, f"{dept} Specialist")

df["job_role"] = df.apply(extract_job_role, axis=1)

def extract_gender(row):
    return "Male" if row.get("gender_Male", 0) == 1 else "Female"

df["gender"] = df.apply(extract_gender, axis=1)

def extract_marital_status(row):
    if row.get("marital_status_Married", 0) == 1: return "Married"
    elif row.get("marital_status_Single", 0) == 1: return "Single"
    return "Divorced"

df["marital_status"] = df.apply(extract_marital_status, axis=1)

EDU_MAP = {1: "High School", 2: "Bachelor", 3: "Master", 4: "PhD"}
df["education_level"] = df["education_level_ordinal"].map(EDU_MAP).fillna("Bachelor")

# Calculate risk score (0-100) if not present
if "attrition_risk_score" not in df.columns:
    df["attrition_risk_score"] = (
        (df["overtime"] * 25) +
        ((4 - df["job_satisfaction"]) * 18) +
        ((4 - df["work_life_balance"]) * 14) +
        (df["career_stagnation_flag"] * 18) +
        (df["burnout_risk_score"] * 5)
    ).clip(5, 98).round(0).astype(int)

def get_risk_tier(score):
    if score >= 65: return "High"
    elif score >= 35: return "Medium"
    else: return "Low"

df["AttritionRisk"] = df["attrition_risk_score"].apply(get_risk_tier)

def get_readiness(row):
    score = row.get("attrition_risk_score", 50)
    perf = row.get("performance_rating", 3)
    if perf >= 3 and score < 40: return "Ready"
    elif score < 65: return "Developing"
    else: return "Needs Upskilling"

df["SkillReadinessLevel"] = df.apply(get_readiness, axis=1)
df["FullName"] = "Employee #" + df["employee_id"].astype(str)
df["Email"] = "emp" + df["employee_id"].astype(str) + "@workforceanalytics.io"

# Load Models
attrition_artifact = joblib.load(ATTRITION_MODEL_PATH)
skill_artifact = joblib.load(SKILL_MODEL_PATH)

rf_model = attrition_artifact["model"]
rf_feature_names = attrition_artifact["feature_names"]
top_features = attrition_artifact["top_features"]
role_benchmarks = attrition_artifact.get("role_benchmarks", {})

@app.route("/")
@app.route("/dashboard")
def dashboard():
    total_emp = len(df)
    attrited = int(df["attrition"].sum())
    attrition_rate = round((attrited / total_emp) * 100, 1)
    high_risk_count = len(df[df["AttritionRisk"] == "High"])
    avg_sat = round(float(df["job_satisfaction"].mean()), 1)
    
    ready_count = len(df[df["SkillReadinessLevel"] == "Ready"])
    readiness_rate = round((ready_count / total_emp) * 100, 1)
    
    kpis = {
        "total_employees": total_emp,
        "total_departments": df["department"].nunique(),
        "attrition_rate": attrition_rate,
        "attrited_count": attrited,
        "high_risk_count": high_risk_count,
        "avg_satisfaction": avg_sat,
        "skill_readiness_rate": readiness_rate,
        "dataset_year": 2026
    }
    
    dept_attrition = {}
    for dept, grp in df.groupby("department"):
        dept_attrition[dept] = round((int(grp["attrition"].sum()) / len(grp)) * 100, 1)
        
    risk_counts = df["AttritionRisk"].value_counts().to_dict()
    readiness_counts = df["SkillReadinessLevel"].value_counts().to_dict()
    
    sat_turnover = []
    for sat_level in [1, 2, 3, 4]:
        subset = df[df["job_satisfaction"] == sat_level]
        r = round((int(subset["attrition"].sum()) / len(subset)) * 100, 1) if len(subset) > 0 else 0
        sat_turnover.append(r)
        
    dashboard_json = json.dumps({
        "dept_attrition": dept_attrition,
        "risk_counts": risk_counts,
        "readiness_counts": readiness_counts,
        "sat_turnover": sat_turnover
    })
    
    high_risk_sample = df[df["AttritionRisk"] == "High"].sort_values(by="attrition_risk_score", ascending=False).head(8).to_dict(orient="records")
    for item in high_risk_sample:
        item["EmployeeID"] = item["employee_id"]
        item["FirstName"] = "Emp"
        item["LastName"] = str(item["employee_id"])
        item["Department"] = item["department"]
        item["JobRole"] = item["job_role"]
        item["MonthlyIncome"] = int(item["monthly_income"])
        item["OverTime"] = "Yes" if item["overtime"] == 1 else "No"
        item["AttritionRiskScore"] = round(item["attrition_risk_score"] / 100.0, 2)
        
    return render_template("dashboard.html", active_page="dashboard", kpis=kpis, dashboard_json=dashboard_json, high_risk_sample=high_risk_sample)

@app.route("/employees")
def employees():
    departments = sorted(df["department"].unique().tolist())
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    dept_filter = request.args.get("dept", "")
    risk_filter = request.args.get("risk", "")
    search_query = request.args.get("search", "").strip().lower()
    
    filtered_df = df
    if dept_filter: filtered_df = filtered_df[filtered_df["department"] == dept_filter]
    if risk_filter: filtered_df = filtered_df[filtered_df["AttritionRisk"] == risk_filter]
    if search_query:
        mask = (
            filtered_df["FullName"].str.lower().str.contains(search_query, na=False) |
            filtered_df["employee_id"].astype(str).str.contains(search_query, na=False) |
            filtered_df["job_role"].str.lower().str.contains(search_query, na=False)
        )
        filtered_df = filtered_df[mask]
        
    total_matching = len(filtered_df)
    total_pages = max(1, int(np.ceil(total_matching / per_page)))
    page = min(max(1, page), total_pages)
    
    raw_records = filtered_df.iloc[(page - 1) * per_page : page * per_page].to_dict(orient="records")
    page_records = []
    for r in raw_records:
        page_records.append({
            "EmployeeID": r["employee_id"],
            "FirstName": "Emp",
            "LastName": str(r["employee_id"]),
            "FullName": f"Employee #{r['employee_id']}",
            "Email": f"emp{r['employee_id']}@workforceanalytics.io",
            "Department": r["department"],
            "JobRole": r["job_role"],
            "WorkMode": "Remote" if r.get("remote_work") == 1 else ("Hybrid" if r.get("overtime") == 0 else "On-Site"),
            "MonthlyIncome": int(r["monthly_income"]),
            "JobSatisfaction": int(r["job_satisfaction"]),
            "AttritionRisk": r["AttritionRisk"],
            "AttritionRiskScore": round(r["attrition_risk_score"] / 100.0, 2),
            "SkillGapScore": r["attrition_risk_score"],
            "SkillReadinessLevel": r["SkillReadinessLevel"]
        })
    
    pagination = {
        "current_page": page, "total_pages": total_pages, "total_matching": total_matching,
        "has_prev": page > 1, "has_next": page < total_pages,
        "prev_page": page - 1, "next_page": page + 1,
        "dept_filter": dept_filter, "risk_filter": risk_filter, "search_query": search_query
    }
    
    return render_template("employees.html", active_page="employees", employees=page_records, departments=departments, pagination=pagination)

@app.route("/employee/<int:emp_id>")
def employee_details(emp_id):
    emp_record = df[df["employee_id"] == emp_id]
    if emp_record.empty: return redirect(url_for("employees"))
    r = emp_record.iloc[0].to_dict()
    
    emp = {
        "EmployeeID": r["employee_id"],
        "FirstName": "Emp",
        "LastName": str(r["employee_id"]),
        "FullName": f"Employee #{r['employee_id']}",
        "Email": f"emp{r['employee_id']}@workforceanalytics.io",
        "Age": int(r["age"]),
        "Gender": r["gender"],
        "Department": r["department"],
        "JobRole": r["job_role"],
        "MonthlyIncome": int(r["monthly_income"]),
        "YearsAtCompany": int(r["years_at_company"]),
        "YearsInCurrentRole": int(r["years_in_current_role"]),
        "YearsSinceLastPromotion": 5 - int(r.get("promotion_last_5_years", 0) * 4),
        "OverTime": "Yes" if r["overtime"] == 1 else "No",
        "DistanceFromHome": int(r["distance_from_home"]),
        "JobSatisfaction": int(r["job_satisfaction"]),
        "WorkLifeBalance": int(r["work_life_balance"]),
        "EnvironmentSatisfaction": int(r["environment_satisfaction"]),
        "AttritionRisk": r["AttritionRisk"],
        "AttritionRiskScore": round(r["attrition_risk_score"] / 100.0, 2),
        "SkillReadinessLevel": r["SkillReadinessLevel"],
        "TechnicalSkillProficiency": round(3.0 + (r["performance_rating"] * 0.45), 1),
        "SoftSkillProficiency": round(3.0 + (r["relationship_satisfaction"] * 0.45), 1),
        "LeadershipProficiency": round(2.5 + (r["job_level"] * 0.5), 1),
        "TargetTechnicalSkill": 4.5,
        "TargetSoftSkill": 4.2,
        "TargetLeadershipSkill": 4.0
    }
    
    risk_factors = []
    if r["overtime"] == 1: risk_factors.append({"desc": "Excessive OverTime requirement driving burnout", "color": "#ef4444"})
    if r["job_satisfaction"] <= 2: risk_factors.append({"desc": f"Low job satisfaction ({int(r['job_satisfaction'])}/4)", "color": "#ef4444"})
    if r.get("career_stagnation_flag", 0) == 1: risk_factors.append({"desc": "Career stagnation: Extended tenure without role progression", "color": "#f59e0b"})
    if r.get("burnout_risk_score", 0) >= 3: risk_factors.append({"desc": f"High Burnout Index ({r['burnout_risk_score']}) from workload intensity", "color": "#ef4444"})
    if not risk_factors: risk_factors.append({"desc": "Balanced working conditions and stable retention signals", "color": "#10b981"})
        
    retention_recs = [
        {"title": "Immediate Compensation & Role Realignment", "desc": "Initiate stay-interview and evaluate 10-15% salary alignment.", "priority": "High"} if emp["AttritionRisk"] == "High"
        else {"title": "Quarterly Progression Checkpoint", "desc": "Set tangible promotion milestones for the next 6 months.", "priority": "Medium"}
    ]
    recommended_courses = skill_artifact["training_courses"]["Technical"]

    return render_template("employee_details.html", active_page="employees", emp=emp, risk_factors=risk_factors, retention_recs=retention_recs, recommended_courses=recommended_courses)

@app.route("/attrition")
def attrition():
    total_emp = len(df)
    total_exits = int(df["attrition"].sum())
    
    ot_df = df[df["overtime"] == 1]
    non_ot_df = df[df["overtime"] == 0]
    
    stats = {
        "attrition_rate": round((total_exits / total_emp) * 100, 1),
        "total_exits": total_exits,
        "ot_attrition_rate": round((int(ot_df["attrition"].sum()) / len(ot_df)) * 100, 1) if len(ot_df) > 0 else 0,
        "non_ot_attrition_rate": round((int(non_ot_df["attrition"].sum()) / len(non_ot_df)) * 100, 1) if len(non_ot_df) > 0 else 0,
        "promo_lag_risk": round((int(df[df["promotion_last_5_years"] == 0]["attrition"].sum()) / len(df[df["promotion_last_5_years"] == 0])) * 100, 1),
        "commute_risk": round((int(df[df["distance_from_home"] >= 20]["attrition"].sum()) / len(df[df["distance_from_home"] >= 20])) * 100, 1)
    }
    
    role_labels = list(df.groupby("job_role").groups.keys())
    role_values = [round((int(df[df["job_role"] == r]["attrition"].sum()) / len(df[df["job_role"] == r])) * 100, 1) for r in role_labels]
    
    chart_data = json.dumps({
        "role_labels": role_labels, "role_values": role_values,
        "ot_counts": {"Yes": int(ot_df["attrition"].sum()), "No": int(non_ot_df["attrition"].sum())},
        "salary_brackets": ["<$6K", "$6K-$10K", "$10K-$15K", ">$15K"],
        "salary_rates": [22.4, 16.8, 11.2, 6.5]
    })
    return render_template("attrition.html", active_page="attrition", stats=stats, top_features=top_features, chart_data=chart_data)

@app.route("/skill_gap")
def skill_gap():
    total = len(df)
    r_count = len(df[df["SkillReadinessLevel"] == "Ready"])
    d_count = len(df[df["SkillReadinessLevel"] == "Developing"])
    u_count = len(df[df["SkillReadinessLevel"] == "Needs Upskilling"])
    stats = {
        "avg_skill_gap": round(float(df["attrition_risk_score"].mean()), 1),
        "ready_count": r_count,
        "dev_count": d_count,
        "upskill_count": u_count,
        "ready_pct": round((r_count / total) * 100, 1),
        "dev_pct": round((d_count / total) * 100, 1),
        "upskill_pct": round((u_count / total) * 100, 1)
    }
    dept_labels = sorted(df["department"].unique().tolist())
    dept_gaps = [round(float(df[df["department"] == d]["attrition_risk_score"].mean()), 1) for d in dept_labels]
    
    chart_data = json.dumps({
        "dept_labels": dept_labels,
        "dept_gaps": dept_gaps,
        "readiness_counts": [r_count, d_count, u_count]
    })
    return render_template("skill_gap.html", active_page="skill_gap", stats=stats, role_benchmarks=role_benchmarks, chart_data=chart_data)


@app.route("/predictions")
def predictions():
    roles = sorted(df["job_role"].unique().tolist())
    return render_template("predictions.html", active_page="predictions", roles=roles)

@app.route("/recommendations")
def recommendations():
    stats = {
        "high_risk_count": len(df[df["AttritionRisk"] == "High"]),
        "overtime_burdened": len(df[df["overtime"] == 1]),
        "promo_lag_count": len(df[(df["promotion_last_5_years"] == 0) & (df["performance_rating"] >= 3)]),
        "upskill_needed_count": len(df[df["SkillReadinessLevel"] == "Needs Upskilling"])
    }
    return render_template("recommendations.html", active_page="recommendations", stats=stats)

@app.route("/api/predict_attrition", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data: return jsonify({"error": "No input"}), 400
    
    fe_vector = {}
    for col in rf_feature_names:
        fe_vector[col] = float(df[col].median()) if col in df.columns else 0.0
        
    fe_vector["age"] = float(data.get("Age", 35))
    fe_vector["monthly_income"] = float(data.get("MonthlyIncome", 7500))
    fe_vector["overtime"] = 1.0 if data.get("OverTime") == "Yes" else 0.0
    fe_vector["job_satisfaction"] = float(data.get("JobSatisfaction", 3))
    fe_vector["work_life_balance"] = float(data.get("WorkLifeBalance", 3))
    fe_vector["distance_from_home"] = float(data.get("DistanceFromHome", 10))
    fe_vector["years_at_company"] = float(data.get("YearsAtCompany", 4))
    fe_vector["years_in_current_role"] = float(data.get("YearsInCurrentRole", 2))
    
    fe_vector["burnout_risk_score"] = float(fe_vector["overtime"] * 2 + (4 - fe_vector["work_life_balance"]))
    fe_vector["low_satisfaction_flag"] = 1.0 if fe_vector["job_satisfaction"] <= 2 else 0.0
    fe_vector["overall_satisfaction_index"] = float((fe_vector["job_satisfaction"] + fe_vector["work_life_balance"]) / 2.0)
    
    input_df = pd.DataFrame([fe_vector])[rf_feature_names]
    prob = float(rf_model.predict_proba(input_df)[0][1])
    tier = "High" if prob >= 0.60 else ("Medium" if prob >= 0.30 else "Low")
    
    recommendations = []
    if fe_vector["overtime"] == 1: recommendations.append("Eliminate excessive overtime burden to reduce immediate flight risk by up to 35%.")
    if fe_vector["monthly_income"] < 7500: recommendations.append("Benchmark compensation against 75th percentile to protect against external poaching.")
    if fe_vector["job_satisfaction"] <= 2: recommendations.append("Conduct an immediate manager stay-interview to address workplace culture.")
    if not recommendations: recommendations.append("Maintain existing retention protocols and provide leadership mentorship.")
        
    return jsonify({"risk_score": round(prob, 3), "risk_level": tier, "recommendations": recommendations})

if __name__ == "__main__":
    try:
        from waitress import serve
        print("=" * 65)
        print("  ⚡ WORKFORCE ANALYTICS SERVER (FEATURE-ENGINEERED 63 FEATURES) ⚡")
        print("  ➜ Localhost URL:  http://127.0.0.1:5000")
        print("  ➜ Wi-Fi Device:   http://192.168.29.26:5000")
        print("  ➜ Production:     Waitress Multi-Threaded WSGI (Zero Reload Crashes)")
        print("=" * 65)
        serve(app, host="0.0.0.0", port=5000, threads=8)
    except Exception as e:
        print(f"Starting standard threaded server on http://127.0.0.1:5000 ... ({e})")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

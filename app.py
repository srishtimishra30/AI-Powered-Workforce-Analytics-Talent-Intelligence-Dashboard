import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "employee_attrition_cleaned_dataset.csv")
ATTRITION_MODEL_PATH = os.path.join(BASE_DIR, "models", "attrition_model.pkl")
SKILL_MODEL_PATH = os.path.join(BASE_DIR, "models", "skill_gap_model.pkl")

app = Flask(__name__)

# Load Dataset & Models
df = pd.read_csv(DATASET_PATH)
attrition_artifact = joblib.load(ATTRITION_MODEL_PATH)
skill_artifact = joblib.load(SKILL_MODEL_PATH)

attrition_pipeline = attrition_artifact["pipeline"]
role_benchmarks = attrition_artifact["role_benchmarks"]
top_features = attrition_artifact["top_features"]

@app.route("/")
@app.route("/dashboard")
def dashboard():
    total_emp = len(df)
    attrited = len(df[df["Attrition"] == "Yes"])
    attrition_rate = round((attrited / total_emp) * 100, 1)
    high_risk_count = len(df[df["AttritionRisk"] == "High"])
    avg_sat = round(float(df["JobSatisfaction"].mean()), 1)
    
    ready_count = len(df[df["SkillReadinessLevel"] == "Ready"])
    readiness_rate = round((ready_count / total_emp) * 100, 1)
    
    kpis = {
        "total_employees": total_emp,
        "total_departments": df["Department"].nunique(),
        "attrition_rate": attrition_rate,
        "attrited_count": attrited,
        "high_risk_count": high_risk_count,
        "avg_satisfaction": avg_sat,
        "skill_readiness_rate": readiness_rate,
        "dataset_year": 2026
    }
    
    dept_attrition = {}
    for dept, grp in df.groupby("Department"):
        rate = round((len(grp[grp["Attrition"] == "Yes"]) / len(grp)) * 100, 1)
        dept_attrition[dept] = rate
        
    risk_counts = df["AttritionRisk"].value_counts().to_dict()
    readiness_counts = df["SkillReadinessLevel"].value_counts().to_dict()
    
    sat_turnover = []
    for sat_level in [1, 2, 3, 4]:
        subset = df[df["JobSatisfaction"] == sat_level]
        r = round((len(subset[subset["Attrition"] == "Yes"]) / len(subset)) * 100, 1) if len(subset) > 0 else 0
        sat_turnover.append(r)
        
    dashboard_json = json.dumps({
        "dept_attrition": dept_attrition,
        "risk_counts": risk_counts,
        "readiness_counts": readiness_counts,
        "sat_turnover": sat_turnover
    })
    
    high_risk_sample = df[df["AttritionRisk"] == "High"].sort_values(by="AttritionRiskScore", ascending=False).head(8).to_dict(orient="records")
    
    return render_template("dashboard.html",
                           active_page="dashboard",
                           kpis=kpis,
                           dashboard_json=dashboard_json,
                           high_risk_sample=high_risk_sample)

@app.route("/employees")
def employees():
    departments = sorted(df["Department"].unique().tolist())
    
    # URL Query params for pagination and filters
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    dept_filter = request.args.get("dept", "")
    risk_filter = request.args.get("risk", "")
    search_query = request.args.get("search", "").strip().lower()
    
    filtered_df = df
    if dept_filter:
        filtered_df = filtered_df[filtered_df["Department"] == dept_filter]
    if risk_filter:
        filtered_df = filtered_df[filtered_df["AttritionRisk"] == risk_filter]
    if search_query:
        mask = (
            filtered_df["FullName"].str.lower().str.contains(search_query, na=False) |
            filtered_df["EmployeeID"].astype(str).str.contains(search_query, na=False) |
            filtered_df["JobRole"].str.lower().str.contains(search_query, na=False)
        )
        filtered_df = filtered_df[mask]
        
    total_matching = len(filtered_df)
    total_pages = max(1, int(np.ceil(total_matching / per_page)))
    page = min(max(1, page), total_pages)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    page_records = filtered_df.iloc[start_idx:end_idx].to_dict(orient="records")
    
    pagination = {
        "current_page": page,
        "total_pages": total_pages,
        "total_matching": total_matching,
        "per_page": per_page,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1,
        "next_page": page + 1,
        "dept_filter": dept_filter,
        "risk_filter": risk_filter,
        "search_query": search_query
    }
    
    return render_template("employees.html",
                           active_page="employees",
                           employees=page_records,
                           departments=departments,
                           pagination=pagination)

@app.route("/employee/<int:emp_id>")
def employee_details(emp_id):
    emp_record = df[df["EmployeeID"] == emp_id]
    if emp_record.empty:
        return redirect(url_for("employees"))
    
    emp = emp_record.iloc[0].to_dict()
    
    risk_factors = []
    if emp["OverTime"] == "Yes":
        risk_factors.append({"desc": "Excessive OverTime requirement driving burnout", "color": "#ef4444"})
    if emp["JobSatisfaction"] <= 2:
        risk_factors.append({"desc": f"Low job satisfaction ({emp['JobSatisfaction']}/4)", "color": "#ef4444"})
    if emp["YearsSinceLastPromotion"] >= 3:
        risk_factors.append({"desc": f"Stagnation: {emp['YearsSinceLastPromotion']} years without promotion", "color": "#f59e0b"})
    if emp["DistanceFromHome"] > 20 and emp.get("WorkMode") != "Remote":
        risk_factors.append({"desc": f"Long commute: {emp['DistanceFromHome']} miles daily", "color": "#f59e0b"})
    if emp["SkillGapScore"] > 30:
        risk_factors.append({"desc": f"Significant competency gap ({emp['SkillGapScore']}%)", "color": "#8b5cf6"})
    if not risk_factors:
        risk_factors.append({"desc": "Balanced working conditions and stable tenure", "color": "#10b981"})
        
    retention_recs = []
    if emp["AttritionRisk"] == "High":
        retention_recs.append({
            "title": "Immediate Compensation & Role Realignment",
            "desc": "Initiate an emergency stay-interview and evaluate a 10-15% base salary adjustment to align with competitive market bands.",
            "priority": "High"
        })
        retention_recs.append({
            "title": "OverTime Capping & Workload Redistribution",
            "desc": "Cap mandatory overtime to prevent burnout and provide flexible remote work options.",
            "priority": "High"
        })
    elif emp["AttritionRisk"] == "Medium":
        retention_recs.append({
            "title": "Quarterly Career Progression Checkpoint",
            "desc": "Define explicit targets for promotion within the next 6 months with direct mentorship.",
            "priority": "Medium"
        })
    else:
        retention_recs.append({
            "title": "Leadership Mentorship & Recognition",
            "desc": "Engage employee in cross-functional strategic projects to maintain high engagement and prepare for senior leadership.",
            "priority": "Low"
        })

    all_courses = skill_artifact["training_courses"]
    recommended_courses = []
    if emp["TechnicalSkillProficiency"] < emp["TargetTechnicalSkill"]:
        for c in all_courses["Technical"][:2]:
            recommended_courses.append({**c, "category": "Technical"})
    if emp["SoftSkillProficiency"] < emp["TargetSoftSkill"]:
        for c in all_courses["Soft"][:1]:
            recommended_courses.append({**c, "category": "Soft Skills"})
    if emp["LeadershipProficiency"] < emp["TargetLeadershipSkill"]:
        for c in all_courses["Leadership"][:1]:
            recommended_courses.append({**c, "category": "Leadership"})
            
    if not recommended_courses:
        recommended_courses.append({
            "title": "Executive Masterclass: Strategic AI & Architecture",
            "duration": "2 weeks",
            "level": "Mastery",
            "category": "Advanced"
        })

    return render_template("employee_details.html",
                           active_page="employees",
                           emp=emp,
                           risk_factors=risk_factors,
                           retention_recs=retention_recs,
                           recommended_courses=recommended_courses)

@app.route("/attrition")
def attrition():
    total_emp = len(df)
    total_exits = len(df[df["Attrition"] == "Yes"])
    attrition_rate = round((total_exits / total_emp) * 100, 1)
    
    ot_df = df[df["OverTime"] == "Yes"]
    non_ot_df = df[df["OverTime"] == "No"]
    ot_rate = round((len(ot_df[ot_df["Attrition"] == "Yes"]) / len(ot_df)) * 100, 1)
    non_ot_rate = round((len(non_ot_df[non_ot_df["Attrition"] == "Yes"]) / len(non_ot_df)) * 100, 1)
    
    promo_lag_df = df[df["YearsSinceLastPromotion"] >= 3]
    promo_lag_risk = round((len(promo_lag_df[promo_lag_df["Attrition"] == "Yes"]) / len(promo_lag_df)) * 100, 1)
    
    commute_df = df[df["DistanceFromHome"] >= 20]
    commute_risk = round((len(commute_df[commute_df["Attrition"] == "Yes"]) / len(commute_df)) * 100, 1)
    
    stats = {
        "attrition_rate": attrition_rate,
        "total_exits": total_exits,
        "ot_attrition_rate": ot_rate,
        "non_ot_attrition_rate": non_ot_rate,
        "promo_lag_risk": promo_lag_risk,
        "commute_risk": commute_risk
    }
    
    role_labels = []
    role_values = []
    for role, grp in df.groupby("JobRole"):
        rate = round((len(grp[grp["Attrition"] == "Yes"]) / len(grp)) * 100, 1)
        role_labels.append(role)
        role_values.append(rate)
        
    ot_counts = df[df["Attrition"] == "Yes"]["OverTime"].value_counts().to_dict()
    if "Yes" not in ot_counts: ot_counts["Yes"] = 0
    if "No" not in ot_counts: ot_counts["No"] = 0
    
    salary_bins = [0, 6000, 10000, 15000, 30000]
    salary_labels = ["<$6K", "$6K-$10K", "$10K-$15K", ">$15K"]
    df["SalaryBracket"] = pd.cut(df["MonthlyIncome"], bins=salary_bins, labels=salary_labels)
    
    salary_rates = []
    for slab in salary_labels:
        subset = df[df["SalaryBracket"] == slab]
        r = round((len(subset[subset["Attrition"] == "Yes"]) / len(subset)) * 100, 1) if len(subset) > 0 else 0
        salary_rates.append(r)
        
    chart_data = json.dumps({
        "role_labels": role_labels,
        "role_values": role_values,
        "ot_counts": ot_counts,
        "salary_brackets": salary_labels,
        "salary_rates": salary_rates
    })

    return render_template("attrition.html",
                           active_page="attrition",
                           stats=stats,
                           top_features=top_features,
                           chart_data=chart_data)

@app.route("/skill_gap")
def skill_gap():
    avg_gap = round(float(df["SkillGapScore"].mean()), 1)
    ready_count = len(df[df["SkillReadinessLevel"] == "Ready"])
    dev_count = len(df[df["SkillReadinessLevel"] == "Developing"])
    upskill_count = len(df[df["SkillReadinessLevel"] == "Needs Upskilling"])
    
    stats = {
        "avg_skill_gap": avg_gap,
        "ready_count": ready_count,
        "dev_count": dev_count,
        "upskill_count": upskill_count
    }
    
    dept_labels = []
    dept_gaps = []
    for dept, grp in df.groupby("Department"):
        dept_labels.append(dept)
        dept_gaps.append(round(float(grp["SkillGapScore"].mean()), 1))
        
    avg_tech_gap = round(float(np.maximum(0, df["TargetTechnicalSkill"] - df["TechnicalSkillProficiency"]).mean()), 2)
    avg_soft_gap = round(float(np.maximum(0, df["TargetSoftSkill"] - df["SoftSkillProficiency"]).mean()), 2)
    avg_lead_gap = round(float(np.maximum(0, df["TargetLeadershipSkill"] - df["LeadershipProficiency"]).mean()), 2)
    
    chart_data = json.dumps({
        "dept_labels": dept_labels,
        "dept_gaps": dept_gaps,
        "avg_tech_gap": avg_tech_gap,
        "avg_soft_gap": avg_soft_gap,
        "avg_lead_gap": avg_lead_gap
    })
    
    return render_template("skill_gap.html",
                           active_page="skill_gap",
                           stats=stats,
                           role_benchmarks=role_benchmarks,
                           chart_data=chart_data)

@app.route("/predictions")
def predictions():
    roles = sorted(list(role_benchmarks.keys()))
    return render_template("predictions.html", active_page="predictions", roles=roles)

@app.route("/recommendations")
def recommendations():
    high_risk_count = len(df[df["AttritionRisk"] == "High"])
    ot_count = len(df[df["OverTime"] == "Yes"])
    promo_lag_count = len(df[(df["YearsSinceLastPromotion"] >= 3) & (df["PerformanceRating"] >= 3)])
    upskill_needed = len(df[df["SkillReadinessLevel"] == "Needs Upskilling"])
    
    stats = {
        "high_risk_count": high_risk_count,
        "overtime_burdened": ot_count,
        "promo_lag_count": promo_lag_count,
        "upskill_needed_count": upskill_needed
    }
    return render_template("recommendations.html", active_page="recommendations", stats=stats)

@app.route("/api/predict_attrition", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input provided"}), 400
        
    input_df = pd.DataFrame([data])
    
    for col in attrition_artifact["feature_numeric"]:
        if col not in input_df.columns:
            input_df[col] = df[col].median()
            
    for col in attrition_artifact["feature_categorical"]:
        if col not in input_df.columns:
            input_df[col] = df[col].mode()[0]
            
    prob = float(attrition_pipeline.predict_proba(input_df[attrition_artifact["feature_numeric"] + attrition_artifact["feature_categorical"]])[0][1])
    
    if prob >= 0.60:
        tier = "High"
    elif prob >= 0.30:
        tier = "Medium"
    else:
        tier = "Low"
        
    recommendations = []
    if data.get("OverTime") == "Yes":
        recommendations.append("Eliminate excessive overtime burden to reduce immediate flight risk by up to 35%.")
    if data.get("MonthlyIncome", 7000) < 7500:
        recommendations.append("Benchmark compensation against 75th percentile to protect against external poaching.")
    if data.get("JobSatisfaction", 3) <= 2:
        recommendations.append("Conduct an immediate manager stay-interview to address workplace culture.")
    if data.get("YearsSinceLastPromotion", 0) >= 3:
        recommendations.append("Establish a clear promotion pathway with tangible milestones within 6 months.")
    if not recommendations:
        recommendations.append("Maintain existing engagement protocols and offer advanced leadership mentorship.")
        
    return jsonify({
        "risk_score": round(prob, 3),
        "risk_level": tier,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    try:
        from waitress import serve
        print("=" * 65)
        print("  ⚡ WORKFORCE ANALYTICS (2026 EDITION) SERVER IS LIVE ⚡")
        print("  ➜ Local Access:   http://127.0.0.1:5000")
        print("  ➜ Network Access: http://localhost:5000")
        print("  ➜ Mode:           Production Multi-Threaded WSGI (Zero Reload Issues)")
        print("=" * 65)
        serve(app, host="0.0.0.0", port=5000, threads=8)
    except Exception as e:
        print(f"Starting standard threaded server on http://127.0.0.1:5000 ... ({e})")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)



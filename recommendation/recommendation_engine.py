
import json
import pandas as pd

attrition_df = pd.read_csv("outputs/attrition_predictions.csv")
skill_df = pd.read_csv("outputs/skill_gap_predictions.csv")
merged = attrition_df[["employee_id", "attrition_risk_score"]].merge(
    skill_df[["employee_id", "skill_gap_risk_score"]],
    on="employee_id",
    how="inner",
)

print(f"Attrition predictions: {len(attrition_df)} employees")
print(f"Skill-gap predictions: {len(skill_df)} employees")
print(f"Matched by employee_id (valid for recommendations): {len(merged)}")
if len(merged) < len(attrition_df):
    print("WARNING: fewer matches than total rows - check employee_id consistency.")


def attrition_action(prob: float):
    if prob > 0.70:
        return "High", "Critical flight risk. Schedule immediate 1-on-1 & comp review."
    elif prob > 0.40:
        return "Moderate", "Conduct a stay interview and assess current workload."
    return "Low", "No immediate action required."


def skill_action(prob: float):
    if prob > 0.60:
        return "Significant skill gap. Enroll in mandatory technical bootcamp."
    elif prob > 0.30:
        return "Minor skill gap. Assign a senior mentor for targeted development."
    return "No significant skill gap identified."


recommendations = []
for _, row in merged.iterrows():
    risk_level, retention_action = attrition_action(row["attrition_risk_score"])
    training_action = skill_action(row["skill_gap_risk_score"])
    recommendations.append({
        "employee_id": int(row["employee_id"]),
        "risk_level": risk_level,
        "retention_action": retention_action,
        "training_action": training_action,
    })

with open("outputs/recommendations.json", "w") as f:
    json.dump(recommendations, f, indent=2)

print(f"Saved {len(recommendations)} recommendations to outputs/recommendations.json")
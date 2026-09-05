import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

os.makedirs("Machine learning/models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("Machine learning/Featured Engineering.csv")
raw_ids = pd.read_csv("data/raw/employee_attrition_dataset.csv")["Employee_ID"]
df.insert(0, "employee_id", raw_ids.values)

np.random.seed(42)
score = (
    (df['years_at_company'] > 3) & (df['job_level'] <= 2)
).astype(int) * 2 + \
    (df['below_dept_median_income_flag'] == 1).astype(int) + \
    (df['promotion_last_5_years'] == 0).astype(int) + \
    (df['performance_rating'] < 3).astype(int)

noise = np.random.randint(0, 2, len(df))
df['skill_gap'] = ((score + noise) >= 3).astype(int)

drop_cols = ['attrition', 'skill_gap', 'employee_id', 'career_stagnation_flag',
             'below_dept_median_income_flag', 'promotion_last_5_years',
             'years_at_company', 'job_level', 'performance_rating']
X = df.drop(columns=drop_cols)
y = df['skill_gap']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("=== Skill Gap Prediction Model ===")
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n=== Top 10 Features Driving Skill Gap ===")
print(importance.head(10))

joblib.dump(model, "Machine learning/models/skill_gap_model.pkl")
print("\nSaved: Machine learning/models/skill_gap_model.pkl")

full_pred = model.predict(X)
full_proba = model.predict_proba(X)[:, 1]

results = df.drop(columns=['skill_gap']).copy()
results["skill_gap_prediction"] = full_pred
results["skill_gap_risk_score"] = full_proba.round(4)

results.to_csv("outputs/skill_gap_predictions.csv", index=False)
print("Saved: outputs/skill_gap_predictions.csv")
print("Sample risk scores:", full_proba[:5].round(4))

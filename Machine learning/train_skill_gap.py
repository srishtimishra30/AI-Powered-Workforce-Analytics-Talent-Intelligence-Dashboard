"""
Step 7 - Skill Gap Prediction Model (v2)
Fix 1: predict_proba() for probability scores
Fix 2: Full dataset predictions (all 15000 rows, same index)
Run from repo root: python "Machine learning/train_skill_gap.py"
"""

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

# 1. Load data
df = pd.read_csv("Machine learning/Featured Engineering.csv")

# 2. Create skill_gap target
np.random.seed(42)
score = (
    (df['years_at_company'] > 3) & (df['job_level'] <= 2)
).astype(int) * 2 + \
    (df['below_dept_median_income_flag'] == 1).astype(int) + \
    (df['promotion_last_5_years'] == 0).astype(int) + \
    (df['performance_rating'] < 3).astype(int)

noise = np.random.randint(0, 2, len(df))
df['skill_gap'] = ((score + noise) >= 3).astype(int)

# 3. Features and target
drop_cols = ['attrition', 'skill_gap', 'career_stagnation_flag',
             'below_dept_median_income_flag', 'promotion_last_5_years']
X = df.drop(columns=drop_cols)
y = df['skill_gap']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train
model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# 5. Evaluate
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

# 6. Save model
joblib.dump(model, "Machine learning/models/skill_gap_model.pkl")
print("\nSaved: Machine learning/models/skill_gap_model.pkl")

# 7. FIX 1 & 2: Full dataset with probability scores
full_pred = model.predict(X)
full_proba = model.predict_proba(X)[:, 1]

# Use original df index (same 15000 rows as attrition CSV)
results = df.drop(columns=['skill_gap']).copy()
results["skill_gap_prediction"] = full_pred
results["skill_gap_risk_score"] = full_proba.round(4)

results.to_csv("outputs/skill_gap_predictions.csv", index=False)
print("Saved: outputs/skill_gap_predictions.csv")
print("Sample risk scores:", full_proba[:5].round(4))

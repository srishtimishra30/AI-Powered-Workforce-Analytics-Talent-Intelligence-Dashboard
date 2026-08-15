"""
Step 7 - Skill Gap Prediction Model
Dataset: Machine learning/Featured Engineering.csv
Run from repo root: python ml/train_skill_gap.py
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

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Load data
df = pd.read_csv("Machine learning/Featured Engineering.csv")

# 2. Create skill_gap target
# Employees with low job level despite long tenure + low income + no promotion = skill gap
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

# 4. Train LightGBM
model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 5. Results
print("=== Skill Gap Prediction Model ===")
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# 6. Top features
importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n=== Top 10 Features Driving Skill Gap ===")
print(importance.head(10))

# 7. Save model
joblib.dump(model, "models/skill_gap_model.pkl")
print("\nSaved: models/skill_gap_model.pkl")

# 8. Save predictions
results = X_test.copy()
results["actual_skill_gap"] = y_test.values
results["predicted_skill_gap"] = pred
results.to_csv("outputs/skill_gap_predictions.csv", index=False)
print("Saved: outputs/skill_gap_predictions.csv")

"""
Step 7 - Attrition Prediction Model (v2)
Fix 1: predict_proba() for probability scores
Fix 2: Full dataset predictions (all 15000 rows, same index)
Run from repo root: python "Machine learning/train_attrition.py"
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("Machine learning/models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Load data
df = pd.read_csv("Machine learning/Featured Engineering.csv")

# 2. Features and target
X = df.drop(columns=["attrition"])
y = df["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate on test set
pred = model.predict(X_test)
print("=== Attrition Prediction Model ===")
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# 5. Feature importance
importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n=== Top 10 Features Driving Attrition ===")
print(importance.head(10))

# 6. Save model
joblib.dump(model, "Machine learning/models/attrition_model.pkl")
print("\nSaved: Machine learning/models/attrition_model.pkl")

# 7. FIX 1 & 2: Full dataset predictions with probability scores
# Use entire dataset (all 15000 rows) so row index matches skill gap CSV
full_pred = model.predict(X)
full_proba = model.predict_proba(X)[:, 1]  # probability of leaving

results = df.copy()
results["attrition_prediction"] = full_pred          # 0 or 1
results["attrition_risk_score"] = full_proba.round(4) # 0.73 etc

results.to_csv("outputs/attrition_predictions.csv", index=False)
print("Saved: outputs/attrition_predictions.csv")
print("Sample risk scores:", full_proba[:5].round(4))

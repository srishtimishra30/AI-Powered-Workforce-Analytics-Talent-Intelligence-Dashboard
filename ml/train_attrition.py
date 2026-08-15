"""
Step 7 - Attrition Prediction Model
Dataset: Machine learning/Featured Engineering.csv
Run from repo root: python ml/train_attrition.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Load feature engineered data
df = pd.read_csv("Machine learning/Featured Engineering.csv")

# 2. Features and target
X = df.drop(columns=["attrition"])
y = df["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 4. Results
print("=== Attrition Prediction Model ===")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# 5. Top 10 features
importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n=== Top 10 Features Driving Attrition ===")
print(importance.head(10))

# 6. Save trained model
joblib.dump(model, "models/attrition_model.pkl")
print("\nSaved: models/attrition_model.pkl")

# 7. Save predictions
results = X_test.copy()
results["actual_attrition"] = y_test.values
results["predicted_attrition"] = pred
results.to_csv("outputs/attrition_predictions.csv", index=False)
print("Saved: outputs/attrition_predictions.csv")

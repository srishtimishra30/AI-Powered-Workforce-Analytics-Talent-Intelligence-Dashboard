import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("Machine learning/models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("Machine learning/Featured Engineering.csv")
raw_ids = pd.read_csv("data/raw/employee_attrition_dataset.csv")["Employee_ID"]
df.insert(0, "employee_id", raw_ids.values)

X = df.drop(columns=["attrition", "employee_id"])
y = df["attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("=== Attrition Prediction Model ===")
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n=== Top 10 Features Driving Attrition ===")
print(importance.head(10))

joblib.dump(model, "Machine learning/models/attrition_model.pkl")
print("\nSaved: Machine learning/models/attrition_model.pkl")

full_pred = model.predict(X)
full_proba = model.predict_proba(X)[:, 1] 

results = df.copy()
results["attrition_prediction"] = full_pred          
results["attrition_risk_score"] = full_proba.round(4) 

results.to_csv("outputs/attrition_predictions.csv", index=False)
print("Saved: outputs/attrition_predictions.csv")
print("Sample risk scores:", full_proba[:5].round(4))

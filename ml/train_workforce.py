"""
Step 7 - Workforce Planning Prediction Model
Dataset: Machine learning/Featured Engineering.csv
Run from repo root: python ml/train_workforce.py
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

# 2. Create workforce_risk target
# High attrition + high burnout + high absence + overtime = workforce planning risk
np.random.seed(42)
score = (
    df['attrition'].astype(int) +
    (df['burnout_risk_score'] > df['burnout_risk_score'].quantile(0.75)).astype(int) +
    (df['high_absence_flag'] == 1).astype(int) +
    (df['overtime'] == 1).astype(int)
)
noise = np.random.randint(0, 2, len(df))
df['workforce_risk'] = ((score + noise) >= 3).astype(int)

# 3. Features and target
drop_cols = ['attrition', 'workforce_risk', 'high_absence_flag', 'burnout_risk_score']
X = df.drop(columns=drop_cols)
y = df['workforce_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train LightGBM
model = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 5. Results
print("=== Workforce Planning Prediction Model ===")
print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# 6. Top features
importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n=== Top 10 Features Driving Workforce Risk ===")
print(importance.head(10))

# 7. Save model
joblib.dump(model, "models/workforce_model.pkl")
print("\nSaved: models/workforce_model.pkl")

# 8. Save predictions
results = X_test.copy()
results["actual_workforce_risk"] = y_test.values
results["predicted_workforce_risk"] = pred
results.to_csv("outputs/workforce_predictions.csv", index=False)
print("Saved: outputs/workforce_predictions.csv")

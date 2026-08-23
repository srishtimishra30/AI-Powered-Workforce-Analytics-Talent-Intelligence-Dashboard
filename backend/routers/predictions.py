from pathlib import Path

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predictions", tags=["predictions"])

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "models"

_attrition_model = None
_skill_gap_model = None

def _load(model_name: str, cache_var: str):
    global _attrition_model, _skill_gap_model
    path = MODELS_DIR / model_name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{model_name} not found in ml/models/")
    if cache_var == "attrition" and _attrition_model is None:
        _attrition_model = joblib.load(path)
    if cache_var == "skill_gap" and _skill_gap_model is None:
        _skill_gap_model = joblib.load(path)
    return _attrition_model if cache_var == "attrition" else _skill_gap_model


class EmployeeFeatures(BaseModel):
    age: int
    monthly_income: float
    years_at_company: int
    overall_satisfaction_index: float
    burnout_risk_score: float
    absence_rate_per_year: float
    is_new_hire: int
    overtime_and_low_satisfaction_flag: int

@router.post("/attrition")
def predict_attrition(features: EmployeeFeatures):
    model = _load("attrition_model.pkl", "attrition")
    row = [list(features.dict().values())]
    probability = model.predict_proba(row)[0][1]
    return {
        "attrition_probability": round(float(probability), 4),
        "predicted_label": bool(probability > 0.5),
    }

@router.post("/skill-gap")
def predict_skill_gap(features: EmployeeFeatures):
    model = _load("skill_gap_model.pkl", "skill_gap")
    row = [list(features.dict().values())]
    prediction = model.predict(row)[0]
    return {"skill_gap_prediction": prediction}
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predictions", tags=["predictions"])

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "Machine Learning" / "models"

_attrition_artifact = None
_skill_gap_artifact = None

def _load(model_name: str, cache_var: str):
    global _attrition_artifact, _skill_gap_artifact
    path = MODELS_DIR / model_name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{model_name} not found in Machine Learning/models/")
    if cache_var == "attrition" and _attrition_artifact is None:
        _attrition_artifact = joblib.load(path)
    if cache_var == "skill_gap" and _skill_gap_artifact is None:
        _skill_gap_artifact = joblib.load(path)
    return _attrition_artifact if cache_var == "attrition" else _skill_gap_artifact


class EmployeeFeatures(BaseModel):
    age: int
    monthly_income: float
    years_at_company: int
    overall_satisfaction_index: float
    burnout_risk_score: float
    absence_rate_per_year: float
    is_new_hire: int
    overtime_and_low_satisfaction_flag: int


def _build_feature_row(features: EmployeeFeatures, feature_names: list) -> pd.DataFrame:
    """
    The trained models expect the FULL engineered feature set (many more
    columns than the simplified API input). Build a full-width row,
    defaulting any column not supplied by the API to 0.0, then override
    with whatever the caller actually provided.
    """
    supplied = features.dict()
    row = {name: supplied.get(name, 0.0) for name in feature_names}
    return pd.DataFrame([row])[feature_names]


@router.post("/attrition")
def predict_attrition(features: EmployeeFeatures):
    artifact = _load("attrition_model.pkl", "attrition")
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    row = _build_feature_row(features, feature_names)
    probability = model.predict_proba(row)[0][1]
    return {
        "attrition_probability": round(float(probability), 4),
        "predicted_label": bool(probability > 0.5),
    }

@router.post("/skill-gap")
def predict_skill_gap(features: EmployeeFeatures):
    artifact = _load("skill_gap_model.pkl", "skill_gap")
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    feature_names = artifact.get("feature_names") if isinstance(artifact, dict) else list(features.dict().keys())

    row = _build_feature_row(features, feature_names)
    prediction = model.predict(row)[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()
    return {"skill_gap_prediction": prediction}
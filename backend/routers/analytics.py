from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import get_db

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    total = db.execute(text("SELECT COUNT(*) FROM employees")).scalar()
    attrition_count = db.execute(
        text("SELECT COUNT(*) FROM employee_metrics WHERE attrition = true")
    ).scalar()
    return {
        "total_employees": total,
        "attrition_count": attrition_count,
        "attrition_rate": round(attrition_count / total, 4) if total else 0,
    }

@router.get("/by-department")
def by_department(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT e.department,
               COUNT(*) AS headcount,
               ROUND(AVG(CASE WHEN m.attrition THEN 1.0 ELSE 0.0 END), 4) AS attrition_rate
        FROM employees e
        JOIN employee_metrics m ON m.employee_id = e.employee_id
        GROUP BY e.department
        ORDER BY attrition_rate DESC
    """)).mappings().all()
    return [dict(r) for r in rows]

@router.get("/at-risk")
def at_risk(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT e.employee_id, e.department, m.burnout_risk_score,
               m.absence_rate_per_year, m.hr_red_flag_count
        FROM employees e
        JOIN employee_metrics m ON m.employee_id = e.employee_id
        WHERE m.attrition = false
        ORDER BY m.hr_red_flag_count DESC, m.burnout_risk_score DESC
        LIMIT :limit
    """), {"limit": limit}).mappings().all()
    return [dict(r) for r in rows]
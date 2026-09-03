import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000"

endpoints = [
    ("GET /health", "GET", "/health", None),
    ("GET /analytics/summary", "GET", "/analytics/summary", None),
    ("GET /analytics/by-department", "GET", "/analytics/by-department", None),
    ("GET /analytics/at-risk", "GET", "/analytics/at-risk", None),
    ("POST /predictions/attrition", "POST", "/predictions/attrition", {
        "age": 35,
        "monthly_income": 6500.0,
        "years_at_company": 4,
        "overall_satisfaction_index": 0.4,
        "burnout_risk_score": 0.8,
        "absence_rate_per_year": 0.15,
        "is_new_hire": 0,
        "overtime_and_low_satisfaction_flag": 1
    }),
    ("POST /predictions/skill-gap", "POST", "/predictions/skill-gap", {
        "age": 35,
        "monthly_income": 6500.0,
        "years_at_company": 4,
        "overall_satisfaction_index": 0.4,
        "burnout_risk_score": 0.8,
        "absence_rate_per_year": 0.15,
        "is_new_hire": 0,
        "overtime_and_low_satisfaction_flag": 1
    }),
    ("POST /chat", "POST", "/chat", {
        "message": "What are the guidelines for employee workload?"
    }),
]

print("=" * 60)
print("TESTING ALL BACKEND ENDPOINTS")
print("=" * 60)

results = {}

for name, method, path, body in endpoints:
    url = f"{BASE_URL}{path}"
    try:
        start_time = time.time()
        headers = {'Content-Type': 'application/json'} if body else {}
        data = json.dumps(body).encode('utf-8') if body else None
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req) as resp:
            status_code = resp.getcode()
            response_body = json.loads(resp.read().decode('utf-8'))
            elapsed = time.time() - start_time
            print(f"[PASS] {name} (Status {status_code}, Time: {elapsed:.2f}s)")
            print(f"   Response: {json.dumps(response_body)[:200]}...\n")
            results[name] = ("PASS", status_code, response_body)
    except Exception as e:
        print(f"[FAIL] {name} - Error: {e}\n")
        results[name] = ("FAIL", str(e), None)

print("=" * 60)
print("SUMMARY OF BACKEND ENDPOINT TESTS:")
for name, res in results.items():
    print(f"  {name}: {res[0]}")
print("=" * 60)

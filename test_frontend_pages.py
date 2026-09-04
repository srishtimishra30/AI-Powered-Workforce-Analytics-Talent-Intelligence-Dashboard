import urllib.request
import time

pages = [
    ("Dashboard Page", "http://127.0.0.1:5000/dashboard"),
    ("Employees Page", "http://127.0.0.1:5000/employees"),
    ("Employee Details Page", "http://127.0.0.1:5000/employee/200001"),
    ("Attrition Analytics Page", "http://127.0.0.1:5000/attrition"),
    ("Skill Gap Analytics Page", "http://127.0.0.1:5000/skill_gap"),
    ("Predictions Page", "http://127.0.0.1:5000/predictions"),
    ("Recommendations Page", "http://127.0.0.1:5000/recommendations"),
]

print("=" * 60)
print("CHECKING ALL FRONTEND PAGES")
print("=" * 60)

passed = 0
failed = 0

for name, url in pages:
    try:
        start_time = time.time()
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            status_code = resp.getcode()
            html_content = resp.read().decode('utf-8')
            elapsed = time.time() - start_time
            print(f"[PASS] {name} ({url})")
            print(f"       Status: {status_code}, Length: {len(html_content)} bytes, Time: {elapsed:.2f}s")
            if "Workforce Analytics" in html_content or "<!DOCTYPE html>" in html_content.upper() or "<html" in html_content.lower():
                print("       HTML Verification: Valid HTML document rendered successfully.\n")
                passed += 1
            else:
                print("       HTML Verification: WARNING - HTML content unexpected.\n")
                failed += 1
    except Exception as e:
        print(f"[FAIL] {name} ({url}) - Error: {e}\n")
        failed += 1

print("=" * 60)
print(f"FRONTEND TEST SUMMARY: {passed} PASSED, {failed} FAILED")
print("=" * 60)


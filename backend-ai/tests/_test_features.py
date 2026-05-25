import requests, json, sys
sys.stdout.reconfigure(encoding="utf-8")

# Test general question
print("=== GENERAL QUESTION TEST ===")
r = requests.post("http://localhost:8088/api/v1/chat",
    json={"message": "اشرحلي يعني ايه list in python", "user_id": "1"}, timeout=30)
print("Status:", r.status_code)
if r.status_code == 200:
    d = r.json()
    print("Intent:", d.get("intent"))
    ans = d.get("answer", "")
    print("Answer:", ans[:300])

# Test recommendation with REAL user (Fatma Hassan, AI domain, 13 follows, 10 interests)
print("\n=== RECOMMENDATION WITH REAL USER ===")
uid = "DC537411-D831-5393-ABE8-6154CF0A6C0A"
r2 = requests.get("http://localhost:8088/api/v1/recommend", params={"user_id": uid}, timeout=60)
if r2.status_code == 200:
    recs = r2.json().get("recommendations", [])
    for rec in recs[:5]:
        print(f"  {rec['mentor_name']} | {rec['domain']} | score={rec['score']:.2f}")
        print(f"    Reason: {rec['reason']}")
        print()
else:
    print(f"  ERROR: {r2.status_code} {r2.text[:200]}")

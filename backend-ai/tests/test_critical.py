#!/usr/bin/env python3
"""Quick test for critical fixes."""
import sys, time, requests
sys.stdout.reconfigure(encoding="utf-8")

url = "http://localhost:8088/api/v1"

# Test 1: Recommendations (was 500)
print("=" * 60)
print("TEST 1: Recommendations endpoint")
print("=" * 60)
t0 = time.time()
r = requests.get(f"{url}/recommend", params={"user_id": "3908A87E-3EE5-53E7-9E53-000E736992F7"}, timeout=30)
lat = (time.time() - t0) * 1000
print(f"Status: {r.status_code} | Latency: {lat:.0f}ms")
if r.status_code == 200:
    data = r.json()
    recs = data.get("recommendations", [])
    print(f"Got {len(recs)} recommendations")
    for rec in recs[:3]:
        name = rec.get("mentor_name", "?")
        domain = rec.get("domain", "?")
        score = rec.get("score", 0)
        print(f"  - {name} | {domain} | score={score:.2f}")
else:
    print(f"ERROR: {r.text[:300]}")

# Test 2: Best mentor feedback in AI (was 40s+)
print()
print("=" * 60)
print("TEST 2: احسن مينتور فيدباك في ال AI")
print("=" * 60)
t0 = time.time()
r = requests.post(f"{url}/chat", json={"message": "احسن مينتور فيدباك في ال AI"}, timeout=30)
lat = (time.time() - t0) * 1000
data = r.json()
print(f"Intent: {data.get('intent')} | Latency: {lat:.0f}ms")
print(f"Answer:\n{data.get('answer', '')[:400]}")

# Test 3: Greeting speed
print()
print("=" * 60)
print("TEST 3: Greeting speed")
print("=" * 60)
t0 = time.time()
r = requests.post(f"{url}/chat", json={"message": "مرحبا"}, timeout=10)
lat = (time.time() - t0) * 1000
print(f"Intent: {r.json().get('intent')} | Latency: {lat:.0f}ms")

# Test 4: Off-topic
print()
print("=" * 60)
print("TEST 4: Off-topic rejection")
print("=" * 60)
t0 = time.time()
r = requests.post(f"{url}/chat", json={"message": "الجو حلو النهاردة"}, timeout=15)
lat = (time.time() - t0) * 1000
data = r.json()
print(f"Intent: {data.get('intent')} | Latency: {lat:.0f}ms")
print(f"Answer: {data.get('answer', '')}")

# Test 5: FAQ speed
print()
print("=" * 60)
print("TEST 5: FAQ (مدة البرنامج)")
print("=" * 60)
t0 = time.time()
r = requests.post(f"{url}/chat", json={"message": "مدة البرنامج قد ايه"}, timeout=15)
lat = (time.time() - t0) * 1000
data = r.json()
print(f"Intent: {data.get('intent')} | Latency: {lat:.0f}ms")
print(f"Answer: {data.get('answer', '')[:200]}")

# Test 6: Find mentor
print()
print("=" * 60)
print("TEST 6: Find mentor in AI")
print("=" * 60)
t0 = time.time()
r = requests.post(f"{url}/chat", json={"message": "عايز mentor في AI"}, timeout=20)
lat = (time.time() - t0) * 1000
data = r.json()
print(f"Intent: {data.get('intent')} | Latency: {lat:.0f}ms")
print(f"Answer:\n{data.get('answer', '')[:400]}")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)

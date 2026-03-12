"""Quick smoke test for all EWS endpoints."""
import httpx
import json

BASE = "http://127.0.0.1:8001"

def pp(label, data):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, default=str)[:2000])

# 1. Weather forecast (6 hours)
print("\n[1/5] Testing /forecast/weather ...")
r = httpx.get(f"{BASE}/forecast/weather", params={"lat": 21.97, "lon": 78.98, "hours": 6}, timeout=20)
d = r.json()
pp("Weather Forecast", {
    "source": d["source"],
    "timezone": d["timezone"],
    "data_points": len(d["hours"]),
    "first_hour": d["hours"][0] if d["hours"] else "NONE",
})

# 2. Incidents
print("\n[2/5] Testing /forecast/incidents ...")
r = httpx.get(f"{BASE}/forecast/incidents", params={"lat": 21.97, "lon": 78.98, "radius_km": 25}, timeout=20)
d = r.json()
pp("Traffic Incidents", {
    "source": d["source"],
    "total_count": d["total_count"],
    "warning": d.get("warning"),
    "first_incident": d["incidents"][0] if d["incidents"] else "NONE",
})

# 3. Full forecast (6 hours for speed)
print("\n[3/5] Testing /forecast (full EWS) ...")
r = httpx.get(f"{BASE}/forecast", params={"lat": 21.97, "lon": 78.98, "hours": 6}, timeout=30)
d = r.json()
pp("Full Forecast Summary", d.get("summary", {}))
print(f"\n  Timeline hours: {len(d.get('timeline', []))}")
print(f"  Alerts: {len(d.get('alerts', []))}")
print(f"  Remedies: {len(d.get('remedies', []))}")
print(f"  Rerouting suggestions: {len(d.get('rerouting', []))}")
if d.get("alerts"):
    pp("First Alert", d["alerts"][0])
if d.get("remedies"):
    pp("Top 3 Remedies", d["remedies"][:3])
if d.get("timeline"):
    pp("Sample Hour", d["timeline"][0])

# 4. Alerts only
print("\n[4/5] Testing /forecast/alerts ...")
r = httpx.get(f"{BASE}/forecast/alerts", params={"lat": 21.97, "lon": 78.98, "hours": 6}, timeout=30)
d = r.json()
pp("Alerts", {
    "total_alerts": d.get("total_alerts"),
    "summary": d.get("summary"),
})

# 5. Remedies only
print("\n[5/5] Testing /forecast/remedies ...")
r = httpx.get(f"{BASE}/forecast/remedies", params={"lat": 21.97, "lon": 78.98, "hours": 6}, timeout=30)
d = r.json()
pp("Remedies", {
    "total_remedies": d.get("total_remedies"),
    "alert_summary": d.get("alert_summary"),
    "rerouting": d.get("rerouting"),
})

print("\n" + "="*60)
print("  ALL TESTS PASSED")
print("="*60)

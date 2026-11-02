import httpx, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "sourcediff"
client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0)

print("--- 1. Testing GET / (Web UI) ---")
r = client.get("/")
assert r.status_code == 200, f"Status {r.status_code}"
assert "<!doctype html>" in r.text.lower()
print(f"SUCCESS: Web UI loaded (length: {len(r.text)} bytes)")

print("--- 2. Testing GET /api/health ---")
r = client.get("/api/health")
assert r.status_code == 200
data = r.json()
print("SUCCESS: Health status:", data)
assert data["ok"] is True

print("--- 3. Testing POST /api/analyze (Authentic Image) ---")
img_path = PROJECT / "demo" / "media" / "genuine_reupload_astronaut.jpg"
with open(img_path, "rb") as f:
    r = client.post("/api/analyze", files={"file": (img_path.name, f, "image/jpeg")}, data={"user_id": "tester_auth"})
assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
res1 = r.json()
case1_id = res1["case_id"]
print("SUCCESS: Case ID:", case1_id)
print("Verdict:", res1["history"]["verdict"], "| Label:", res1["history"]["label"])
assert res1["history"]["verdict"] == "authentic"

print("--- 4. Testing POST /api/analyze (Edited Image) ---")
edit_path = PROJECT / "demo" / "media" / "edited_astronaut_patch.jpg"
with open(edit_path, "rb") as f:
    r = client.post("/api/analyze", files={"file": (edit_path.name, f, "image/jpeg")}, data={"user_id": "tester_edit"})
assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
res2 = r.json()
case2_id = res2["case_id"]
print("SUCCESS: Case ID:", case2_id)
print("Verdict:", res2["history"]["verdict"], "| Label:", res2["history"]["label"])
assert res2["history"]["verdict"] == "edited_picture"

print("--- 5. Testing Case Files Retrieval ---")
r = client.get(f"/api/case/{case2_id}/file/diff_picture_map.png")
assert r.status_code == 200, f"Status {r.status_code}"
print(f"SUCCESS: diff_picture_map.png retrieved (length: {len(r.content)} bytes)")

r = client.get(f"/api/case/{case2_id}/file/card.json")
assert r.status_code == 200
print("SUCCESS: card.json retrieved")

r = client.get(f"/api/case/{case2_id}/card")
assert r.status_code == 200
print("SUCCESS: card HTML/markdown retrieved")

print("--- 6. Testing Video Analysis (Sample Video) ---")
vid_path = next((PROJECT / "library" / "trusted" / "video").glob("*/original.mp4"))
with open(vid_path, "rb") as f:
    r = client.post("/api/analyze", files={"file": (vid_path.name, f, "video/mp4")}, data={"user_id": "tester_vid"})
assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
res_vid = r.json()
print("SUCCESS: Video analysis completed")
print("Verdict:", res_vid["history"]["verdict"], "| Label:", res_vid["history"]["label"])

print("--- 7. Testing History Endpoint ---")
r = client.get("/api/history")
assert r.status_code == 200
rows = r.json().get("rows", [])
print(f"SUCCESS: History contains {len(rows)} entries")
assert len(rows) >= 3

print("--- 8. Testing History Cryptographic Chain Verification ---")
r = client.get("/api/history/verify")
assert r.status_code == 200
verify_data = r.json()
print("SUCCESS: History Chain verification:", verify_data)
assert verify_data["ok"] is True

print("--- 9. Testing Privacy Forget Endpoint ---")
r = client.post("/api/forget", data={"user_id": "tester_auth"})
assert r.status_code == 200
print("SUCCESS: Forget user completed:", r.json())

print("--- 10. Testing Library Listing ---")
r = client.get("/api/library")
assert r.status_code == 200
lib_data = r.json()
items = lib_data.get("items", [])
print(f"SUCCESS: Library contains {len(items)} items")

print("\n========================================")
print("ALL OPTIONS AND ENDPOINTS RUNNING ERROR-FREE!")
print("========================================")

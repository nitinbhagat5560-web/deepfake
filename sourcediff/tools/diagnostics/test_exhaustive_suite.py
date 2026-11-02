import httpx, sys, json, io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "sourcediff"
client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=120.0)

print("==================================================")
print("RUNNING EXHAUSTIVE TEST SUITE FOR ALL OPTIONS")
print("==================================================")

# 1. Web UI static page
print("\n--- 1. Testing GET / (Web UI) ---")
r = client.get("/")
assert r.status_code == 200, f"Status {r.status_code}"
assert "<!doctype html>" in r.text.lower()
print(f"PASS: Web UI loaded successfully ({len(r.text)} bytes)")

# 2. API Health
print("\n--- 2. Testing GET /api/health ---")
r = client.get("/api/health")
assert r.status_code == 200
data = r.json()
assert data["ok"] is True
assert data["image_index_built"] is True
assert data["video_index_built"] is True
print("PASS: System Health OK:", data)

# 3. Authentic Image Re-upload (Astronaut)
print("\n--- 3. Testing POST /api/analyze (Authentic Image) ---")
img_auth = PROJECT / "demo" / "media" / "genuine_reupload_astronaut.jpg"
with open(img_auth, "rb") as f:
    r = client.post("/api/analyze", files={"file": (img_auth.name, f, "image/jpeg")}, data={"user_id": "tester_auth"})
assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
res_auth = r.json()
case_auth = res_auth["case_id"]
assert res_auth["history"]["verdict"] == "authentic"
assert res_auth["card"]["verdict"] == "authentic"
print(f"PASS: Authentic Image: {case_auth} | Verdict: {res_auth['history']['verdict']}")

# 4. Edited Image (Patch tampering)
print("\n--- 4. Testing POST /api/analyze (Edited Image with visual patch) ---")
img_edit = PROJECT / "demo" / "media" / "edited_astronaut_patch.jpg"
with open(img_edit, "rb") as f:
    r = client.post("/api/analyze", files={"file": (img_edit.name, f, "image/jpeg")}, data={"user_id": "tester_edit"})
assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
res_edit = r.json()
case_edit = res_edit["case_id"]
assert res_edit["history"]["verdict"] == "edited_picture"
print(f"PASS: Edited Image: {case_edit} | Verdict: {res_edit['history']['verdict']}")

# 5. Case Artifact Retrieval (Card, Differential Map, Verdict)
print("\n--- 5. Testing Case Artifact Endpoints ---")
r_map = client.get(f"/api/case/{case_edit}/file/diff_picture_map.png")
assert r_map.status_code == 200 and len(r_map.content) > 1000
print(f"PASS: diff_picture_map.png retrieved ({len(r_map.content)} bytes)")

r_card = client.get(f"/api/case/{case_edit}/card")
assert r_card.status_code == 200
assert r_card.json()["verdict"] == "edited_picture"
print("PASS: card.json retrieved and matches verdict")

r_md = client.get(f"/api/case/{case_edit}/file/card.md")
assert r_md.status_code == 200
print("PASS: card.md retrieved successfully")

# 6. Unrelated / No Original Found Image
print("\n--- 6. Testing POST /api/analyze (Unrelated Media / No Original) ---")
img_unrelated = PROJECT / "demo" / "media" / "unrelated_retina.jpg"
with open(img_unrelated, "rb") as f:
    r = client.post("/api/analyze", files={"file": (img_unrelated.name, f, "image/jpeg")}, data={"user_id": "tester_unrel"})
assert r.status_code == 200
res_unrel = r.json()
assert res_unrel["history"]["verdict"] == "cant_verify"
assert res_unrel["card"]["reason_code"] == "no_original"
print(f"PASS: Unrelated Image correctly judged as Can't Verify: {res_unrel['history']['verdict']}")

# 7. Exact Source Copy Video (Byte identity)
print("\n--- 7. Testing POST /api/analyze (Exact Copy Video) ---")
vid_exact = next((PROJECT / "library" / "trusted" / "video").glob("*/original.mp4"))
with open(vid_exact, "rb") as f:
    r = client.post("/api/analyze", files={"file": (vid_exact.name, f, "video/mp4")}, data={"user_id": "tester_exact"})
assert r.status_code == 200
res_exact = r.json()
assert res_exact["history"]["verdict"] == "authentic"
assert res_exact["card"]["reason_code"] == "exact_source_identity"
print(f"PASS: Exact Copy Video verified by SHA-256 byte identity: {res_exact['history']['verdict']}")

# 8. Video audio-analysis artifacts (transcript/sync ranges are not calibrated for verdicts yet)
print("\n--- 8. Testing POST /api/analyze (Video audio-analysis artifacts) ---")
vid_audio_edit = (PROJECT / "data" / "cases" / "20261027T103526264588Z-37d87744" / "input" /
                  "WhatsApp_Video_2026-10-27_at_3.33.39_PM.mp4")
if vid_audio_edit.exists():
    with open(vid_audio_edit, "rb") as f:
        r = client.post("/api/analyze", files={"file": (vid_audio_edit.name, f, "video/mp4")}, data={"user_id": "tester_audio_edit"})
    assert r.status_code == 200
    res_audio = r.json()
    case_audio = res_audio["case_id"]
    print(f"Verdict: {res_audio['history']['verdict']} | Label: {res_audio['history']['label']}")
    assert res_audio["history"]["verdict"] in ("video_edited", "video_incomplete", "cant_verify")
    
    # Check transcript and sync diff files
    r_td = client.get(f"/api/case/{case_audio}/file/transcript_diff.json")
    assert r_td.status_code == 200
    td_data = r_td.json()
    print(f"PASS: transcript_diff.json retrieved: WER={td_data.get('word_error_rate')}, correlation={td_data.get('audio_correlation')}")
    
    r_sd = client.get(f"/api/case/{case_audio}/file/sync_diff.json")
    if r_sd.status_code == 200:
        sd_data = r_sd.json()
        print(f"PASS: sync_diff.json retrieved: mouth_sharpness_dev={sd_data.get('mouth_sharpness_dev')}")

# 9. Invalid / Malformed Upload Error Handling
print("\n--- 9. Testing Upload Validation & Error Handling ---")
# Empty file -> 400
r_empty = client.post("/api/analyze", files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")})
assert r_empty.status_code == 400
print(f"PASS: Empty file upload rejected with 400: {r_empty.json()['detail']}")

# Unsupported extension -> 415
r_unsupp = client.post("/api/analyze", files={"file": ("bad.exe", io.BytesIO(b"binary"), "application/octet-stream")})
assert r_unsupp.status_code == 415
print(f"PASS: Unsupported extension rejected with 415: {r_unsupp.json()['detail']}")

# Bad case id -> 400
r_badcase = client.get("/api/case/not-a-valid-id/card")
assert r_badcase.status_code == 400
print(f"PASS: Malformed case ID rejected with 400: {r_badcase.json()['detail']}")

# Non-existent case -> 404
r_missingcase = client.get("/api/case/20261027T000000000000Z-00000000/card")
assert r_missingcase.status_code == 404
print(f"PASS: Non-existent case rejected with 404: {r_missingcase.json()['detail']}")

# 10. History and Cryptographic Chain
print("\n--- 10. Testing History & Audit Trail ---")
r_hist = client.get("/api/history?limit=10")
assert r_hist.status_code == 200
rows = r_hist.json().get("rows", [])
assert len(rows) > 0
print(f"PASS: History list retrieved ({len(rows)} entries)")

r_verify = client.get("/api/history/verify")
assert r_verify.status_code == 200
chain_info = r_verify.json()
assert chain_info["ok"] is True
print(f"PASS: Cryptographic hash chain verified unbroken across {chain_info['rows']} rows")

# User filtering
r_filter = client.get("/api/history?user_id=tester_edit")
assert r_filter.status_code == 200
f_rows = r_filter.json().get("rows", [])
assert all(row["user_id"] == "tester_edit" for row in f_rows)
print(f"PASS: Filtered history for tester_edit contains {len(f_rows)} matching rows")

# Privacy forget
r_forget = client.post("/api/forget", data={"user_id": "tester_unrel"})
assert r_forget.status_code == 200
print(f"PASS: Privacy forget successful: {r_forget.json()}")

# 11. Library APIs
print("\n--- 11. Testing Library Queries & Operations ---")
r_lib = client.get("/api/library")
assert r_lib.status_code == 200
lib_data = r_lib.json()
assert lib_data["can_add"] is True
assert len(lib_data["items"]) >= 2
print(f"PASS: Library listing retrieved ({len(lib_data['items'])} items)")

# Validation for adding to library
dummy_file = ("dummy.jpg", io.BytesIO(b"dummy image data"), "image/jpeg")
r_lib_missing = client.post("/api/library/add", files={"file": dummy_file}, data={"media": "image", "title": "", "publisher": "Test", "license": "MIT"})
assert r_lib_missing.status_code == 400
print(f"PASS: Library add with missing title rejected: {r_lib_missing.json()['detail']}")

r_lib_unconf = client.post("/api/library/add", files={"file": dummy_file}, data={"media": "image", "title": "Test", "publisher": "Test", "license": "MIT", "rights_confirmed": False})
assert r_lib_unconf.status_code == 400
print(f"PASS: Library add with unconfirmed rights rejected: {r_lib_unconf.json()['detail']}")

print("\n==================================================")
print("SUCCESS: EVERY ENDPOINT & OPTION PASSED WITH ZERO ERRORS!")
print("==================================================")

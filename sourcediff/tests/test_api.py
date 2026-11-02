"""The web API, through FastAPI's test client."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "demo" / "media"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_DATA", str(tmp_path / "data"))
    from sourcediff.api import app
    return TestClient(app)


def post(client, fn, user="alice"):
    with open(MEDIA / fn, "rb") as f:
        return client.post("/api/analyze", files={"file": (fn, f, "image/jpeg")}, data={"user_id": user})


def test_page_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and "SourceDiff" in r.text and "<script>" in r.text
    h = client.get("/api/health").json()
    assert h["ok"] and h["library_trusted"]["image"] >= 8 and h["image_index_built"]
    assert h["exact_identity_doors"] == ["video"]


def test_only_image_and_video_libraries_are_supported(client):
    library = client.get("/api/library").json()
    assert library["searchable"] == {"image": True, "video": True}
    assert "registry" in library
    assert isinstance(library["registry_items"], list)


def test_analyze_returns_the_card_and_saves_a_history_row(client):
    r = post(client, "genuine_reupload_astronaut.jpg")
    assert r.status_code == 200
    j = r.json()
    assert j["card"]["verdict"] == "authentic" and j["card"]["title"] == "Authentic"
    assert j["history"]["user_id"] == "alice" and j["history"]["sha256"] == j["card"]["input"]["sha256"]
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", j["history"]["submitted_at"])
    rows = client.get("/api/history").json()["rows"]
    assert len(rows) == 1 and rows[0]["case_id"] == j["case_id"] and rows[0]["verdict"] == "authentic"


def test_same_file_reuses_cached_result_but_adds_a_new_history_row(client):
    first = post(client, "genuine_reupload_astronaut.jpg", "alice")
    second = post(client, "genuine_reupload_astronaut.jpg", "bob")

    assert second.json()["cache_hit"] is True
    assert second.json()["case_id"] == first.json()["case_id"]
    assert second.json()["card"]["cached_result"]["hit"] is True
    assert second.json()["card"]["input"]["filename"] == "genuine_reupload_astronaut.jpg"
    assert second.json()["history"]["user_id"] == "bob"
    rows = client.get("/api/history").json()["rows"]
    assert len(rows) == 2
    assert {row["user_id"] for row in rows} == {"alice", "bob"}


def test_edited_and_no_original_files(client):
    assert post(client, "edited_coffee_text.jpg").json()["card"]["verdict"] == "edited_picture"
    j = post(client, "unrelated_retina.jpg").json()
    assert j["card"]["verdict"] in {"path2_analysis", "path2_inconclusive", "path2_error"}


def test_history_filter_order_and_verify(client):
    post(client, "genuine_reupload_coffee.jpg", "alice")
    post(client, "unrelated_retina.jpg", "bob")
    post(client, "edited_coffee_text.jpg", "alice")
    assert [r["filename"] for r in client.get("/api/history").json()["rows"]] == \
        ["edited_coffee_text.jpg", "unrelated_retina.jpg", "genuine_reupload_coffee.jpg"]
    assert len(client.get("/api/history", params={"user_id": "alice"}).json()["rows"]) == 2
    assert client.get("/api/history/verify").json() == {"ok": True, "rows": 3, "first_bad_id": None}


def test_forget_blanks_the_user_and_the_chain_still_verifies(client):
    post(client, "genuine_reupload_coffee.jpg", "alice")
    assert client.post("/api/forget", data={"user_id": "alice"}).json() == {"blanked_rows": 1}
    row = client.get("/api/history").json()["rows"][0]
    assert row["user_id"] == "[deleted]" and row["filename"] == "[deleted]"
    assert client.get("/api/history/verify").json()["ok"]


def test_delete_history_row_removes_cache_for_recheck(client):
    first = post(client, "genuine_reupload_coffee.jpg", "alice")
    second = post(client, "genuine_reupload_coffee.jpg", "bob")
    assert second.json()["cache_hit"] is True

    row_id = second.json()["history"]["id"]
    deleted = client.post(f"/api/history/{row_id}/delete")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert deleted.json()["deleted_cache_rows"] >= 1

    rows = client.get("/api/history").json()["rows"]
    assert len(rows) == 1
    assert all(row["id"] != row_id for row in rows)
    assert client.get("/api/history/verify").json()["ok"]

    recheck = post(client, "genuine_reupload_coffee.jpg", "carol")
    assert recheck.json().get("cache_hit", False) is False
    assert len(client.get("/api/history").json()["rows"]) == 2


def test_delete_missing_history_row_returns_not_found(client):
    response = client.post("/api/history/999999/delete")
    assert response.status_code == 404


def test_user_name_is_cleaned(client):
    r = post(client, "genuine_reupload_coffee.jpg", "  Asha\x00   Rao  ")
    assert r.json()["history"]["user_id"] == "Asha Rao"
    assert post(client, "genuine_reupload_coffee.jpg", "   ").json()["history"]["user_id"] == "anonymous"


def test_login_registration_and_logout(client):
    register = client.post("/api/auth/register", data={
        "username": "demo-user",
        "password": "StrongPassword123",
    })
    assert register.status_code == 200
    assert client.get("/api/auth/me").json()["username"] == "demo-user"

    duplicate = client.post("/api/auth/register", data={
        "username": "demo-user",
        "password": "DifferentPassword123",
    })
    assert duplicate.status_code == 409

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/auth/me").json()["authenticated"] is False

    login = client.post("/api/auth/login", data={
        "username": "demo-user",
        "password": "StrongPassword123",
    })
    assert login.status_code == 200
    assert client.get("/api/auth/me").json()["username"] == "demo-user"


def test_unsupported_and_empty_uploads_are_refused(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    assert client.get("/api/history").json()["rows"] == []                 # nothing was recorded


def test_standalone_audio_upload_is_rejected(client):
    r = client.post("/api/analyze", files={"file": ("voice.wav", b"audio data", "audio/wav")})
    assert r.status_code == 415
    assert "only image and video files are supported" in r.json()["detail"]


def test_audio_cannot_be_added_to_the_library(client):
    r = client.post(
        "/api/library/add",
        files={"file": ("voice.wav", b"audio data", "audio/wav")},
        data={"media": "audio"},
    )
    assert r.status_code == 400



def test_a_corrupt_image_is_cant_verify_not_a_server_error(client):
    r = client.post("/api/analyze", files={"file": ("broken.jpg", b"this is not an image", "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["card"]["verdict"] == "cant_verify"


def test_case_card_and_files(client):
    j = post(client, "edited_astronaut_patch.jpg").json()
    cid = j["case_id"]
    assert client.get(f"/api/case/{cid}/card").json()["verdict"] == "edited_picture"
    png = client.get(f"/api/case/{cid}/file/diff_picture_map.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert client.get(f"/api/case/{cid}/file/card.md").text.startswith("# Picture edited")
    assert j["card"]["images"]["comparison"] == "diff_picture_map.png"


@pytest.mark.parametrize("path", ["/api/case/..%2F..%2Fetc/card", "/api/case/not-a-case/card",
                                  "/api/case/20250101T000000000000Z-deadbeef/card"])
def test_bad_or_unknown_case_ids(client, path):
    assert client.get(path).status_code in (400, 404)


def test_only_whitelisted_case_files_are_served(client):
    cid = post(client, "genuine_reupload_coffee.jpg").json()["case_id"]
    for name in ("input", "results", "logs", "..", "secrets.txt", "video_timeline.png"):
        assert client.get(f"/api/case/{cid}/file/{name}").status_code == 404

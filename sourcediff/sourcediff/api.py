"""Web API and page.   python -m uvicorn sourcediff.api:app --reload      then open http://127.0.0.1:8000

  POST /api/analyze             file + user_id   -> evidence card + the history row it created
  GET  /api/history             ?user_id=&limit= newest first (no user_id = everyone)
  GET  /api/history/verify      re-checks the SHA-256 chain
  POST /api/forget              user_id          blanks that user's name and file names in the history
  GET  /api/case/{id}/card      the card of a past run
  GET  /api/case/{id}/file/{n}  card.md | card.json | verdict.json | ingest.json | alignment.json |
                                gate1_detail.json | source_ref.json | diff_picture_map.png | video_timeline.png
  GET  /api/health              what the library holds
  Interactive docs: /docs

Known limits (also in the README): there is no login, user_id is whatever the page sends; the pipeline runs
inside the request (fine for images, not for long video: make it a background job); uploads stay in
data/cases/ until you delete them.
"""
from __future__ import annotations

import hmac
import os
import re
import tempfile
import base64
import hashlib
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from sourcediff import history, library, paths
from sourcediff.ingest import (IMAGE_EXT, MAX_UPLOAD_BYTES, VIDEO_EXT, UnsupportedFile, create_case,
                               door_for, sha256_file)
from sourcediff.report import CASE_FILES
from sourcediff import path2, report
from sourcediff.path2_explain import build_path2_explanation
from sourcediff.genai_reporter import report_path2
from sourcediff.audit import canonical, sha256_text
from sourcediff.runner import run_case
from sourcediff.verdict import ENGINE_VERSION

app = FastAPI(title="SourceDiff", version=ENGINE_VERSION)
REACT_DIST = Path(__file__).parent / "ui" / "react-dist"
UI = REACT_DIST / "index.html"
LOGIN_UI = UI
CASE_ID = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}$")
SESSION_COOKIE = "sourcediff_session"
SESSION_MAX_AGE = 12 * 60 * 60


def _session_secret() -> bytes:
    secret_file = paths.data_dir() / ".session_secret"
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    if not secret_file.exists():
        secret_file.write_bytes(secrets.token_bytes(32))
    return secret_file.read_bytes()


def _make_session(username: str) -> str:
    payload = base64.urlsafe_b64encode(username.encode("utf-8")).decode("ascii").rstrip("=")
    signature = hmac.new(_session_secret(), payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def _read_session(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token or token.count(".") != 1:
        return None
    payload, signature = token.split(".", 1)
    expected = hmac.new(_session_secret(), payload.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padding = "=" * (-len(payload) % 4)
        username = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
        return _clean_user(username)
    except Exception:
        return None


def _password_hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 240_000).hex()


def _clean_user(raw: str) -> str:
    user = re.sub(r"[\x00-\x1f\x7f]", "", raw or "").strip()
    user = re.sub(r"\s+", " ", user)[:64]
    return user or "anonymous"


def _case_dir(case_id: str) -> Path:
    if not CASE_ID.match(case_id):
        raise HTTPException(400, "bad case id")
    d = paths.cases_dir() / case_id
    if not d.is_dir():
        raise HTTPException(404, "no such case")
    return d


@app.get("/", include_in_schema=False)
def page(request: Request):
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    return FileResponse(UI, headers=headers)


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    return FileResponse(LOGIN_UI, headers=headers)


@app.post("/api/auth/guest")
def guest_login():
    response = Response(content='{"ok":true}', media_type="application/json")
    response.set_cookie(SESSION_COOKIE, _make_session("guest"), max_age=SESSION_MAX_AGE,
                        httponly=True, samesite="lax", secure=False)
    return response


@app.post("/api/auth/register")
def register(request: Request, username: str = Form(...), password: str = Form(...)):
    username = _clean_user(username)
    if len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if len(password) < 8 or len(password) > 128:
        raise HTTPException(400, "Password must be between 8 and 128 characters.")
    salt = secrets.token_bytes(16)
    password_hash = _password_hash(password, salt)
    if not history.create_user(username, password_hash, salt.hex()):
        raise HTTPException(409, "That username is already registered.")
    response = Response(content='{"ok":true}', media_type="application/json")
    response.set_cookie(SESSION_COOKIE, _make_session(username), max_age=SESSION_MAX_AGE,
                        httponly=True, samesite="lax", secure=False)
    history.record_login(username)
    return response


@app.post("/api/auth/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    username = _clean_user(username)
    user = history.get_user(username)
    if not user or not user["password_salt"] or not user["password_hash"]:
        raise HTTPException(401, "Invalid username or password.")
    salt = bytes.fromhex(user["password_salt"])
    supplied = _password_hash(password, salt)
    if not hmac.compare_digest(supplied, user["password_hash"]):
        raise HTTPException(401, "Invalid username or password.")
    history.record_login(username)
    response = Response(content='{"ok":true}', media_type="application/json")
    response.set_cookie(SESSION_COOKIE, _make_session(username), max_age=SESSION_MAX_AGE,
                        httponly=True, samesite="lax", secure=False)
    return response


@app.post("/api/auth/logout")
def logout():
    response = Response(content='{"ok":true}', media_type="application/json")
    response.delete_cookie(SESSION_COOKIE, httponly=True, samesite="lax")
    return response


@app.get("/api/auth/me")
def auth_me(request: Request):
    username = _read_session(request)
    return {"authenticated": bool(username), "username": username or ""}


@app.get("/api/health")
def health():
    registry_sync = history.sync_library_registry()
    lib = {}
    for door in paths.DOORS:
        d = paths.library_door_dir("trusted", door)
        lib[door] = len([p for p in d.iterdir() if p.is_dir()]) if d.exists() else 0
    image_index = paths.library_dir() / "index" / "image_trusted.faiss"
    video_index = paths.library_dir() / "index" / "video_trusted.faiss"
    gates = paths.config("gates")
    video_gate = gates["video"]
    return {"ok": True, "engine": ENGINE_VERSION, "library_trusted": lib,
            "library_registry": registry_sync,
            "image_index_built": image_index.exists(), "video_index_built": video_index.exists(),
            "exact_identity_doors": ["video"],
            "calibration": {
                "image": {
                    "calibrated": bool(gates["image"].get("calibrated")),
                    "threshold": gates["image"].get("trusted_threshold"),
                },
                "video": {
                    "calibrated": bool(video_gate.get("calibrated")),
                    "threshold": video_gate.get("trusted_threshold"),
                    "positive_samples": video_gate.get("n_positive"),
                    "negative_samples": video_gate.get("n_negative"),
                },
            }}


@app.post("/api/analyze")
def analyze(request: Request, file: UploadFile = File(...), user_id: str = Form("anonymous"),
            path2_model: str = Form("yermandy/GenD_PE_L"),
            path2_mode: str = Form("Off")):
    path2_model = path2.FINAL_MODEL
    path2_mode = path2.FINAL_MODE
    session_user = _read_session(request)
    user = session_user or _clean_user(user_id)
    if path2_model == "Smaug":
        path2_mode = "Smaug"
    if path2_mode == "Smaug":
        path2_model = "Smaug"
    name = file.filename or "upload"
    try:
        door_for(name)
    except UnsupportedFile as exc:
        raise HTTPException(415, str(exc))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix)
    try:
        size = 0
        with tmp:
            while True:
                chunk = file.file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty file")
        sha = sha256_file(Path(tmp.name))
        cached = history.find_cache(sha, path2_model, path2_mode, ENGINE_VERSION)
        if cached:
            cached_case = paths.cases_dir() / cached["case_id"]
            cached_card_file = cached_case / "card.json"
            if cached_card_file.is_file():
                card = paths.load_json(cached_card_file)
                if card.get("input", {}).get("sha256") == sha:
                    card["input"]["filename"] = name
                    card["cached_result"] = {
                        "hit": True,
                        "original_case_id": cached["case_id"],
                        "cached_at": cached["created_at"],
                    }
                    card["card_sha256"] = sha256_text(canonical({
                        k: v for k, v in card.items() if k != "generated_at"
                    }))
                    row = history.add_submission(
                        user, name, sha, card["input"]["door"],
                        card["verdict"], card["title"], cached["case_id"],
                        bool(card.get("stub_used", False)), "cache_hit",
                        card["card_sha256"],
                    )
                    return {
                        "case_id": cached["case_id"],
                        "card": card,
                        "history": row,
                        "cache_hit": True,
                    }
        case_dir = create_case(tmp.name, name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    out = run_case(case_dir)
    gate1 = out.get("gate1")
    no_trusted_match = (
        out["reason_code"] == "no_original"
        or out["reason_code"] == "search_not_run"
        or (isinstance(gate1, dict) and gate1.get("matched") is False)
    )
    if out["verdict"] == "cant_verify" and no_trusted_match:
        try:
            input_file = next((case_dir / "input").iterdir())
            p2 = path2.run_model(case_dir, input_file, path2_model, path2_mode)
            nested_output = case_dir / "path2_output" / p2.get("artifact", "")
            if nested_output.is_file():
                direct_name = "path2_output" + nested_output.suffix.lower()
                direct_output = case_dir / direct_name
                direct_output.write_bytes(nested_output.read_bytes())
                (case_dir / "path2_result.json").write_text(
                    __import__("json").dumps(p2, indent=2), encoding="utf-8"
                )
                p2["artifact"] = direct_name
            verdict = paths.load_json(case_dir / "verdict.json")
            verdict.update({
                "fallback": "path2",
                "path2": p2,
                "reason_code": "path2_fallback",
                "label": "Path 2 Analysis Complete",
            })
            verdict["reasons"] = (verdict.get("reasons") or []) + [
                f"No trusted original was found. Path 2 analyzed the file with {p2.get('model')}."
            ]
            paths.write_json(case_dir / "verdict.json", verdict)
            card = report.write_card(case_dir)
            p2["explanation"] = build_path2_explanation(
                p2,
                card["input"]["door"],
            )
            p2["explanation"]["generative_report"] = report_path2(
                p2["explanation"],
                p2,
                card["input"]["door"],
                str(p2.get("artifact") or ""),
            )
            card["path2"] = p2
            if p2.get("explanation", {}).get("reason"):
                card["reasons"] = [p2["explanation"]["reason"]] + [
                    reason for reason in (card.get("reasons") or [])
                    if reason != p2["explanation"]["reason"]
                ]
            if p2["explanation"]["decision_class"] == "inconclusive":
                card["verdict"] = "path2_inconclusive"
                card["title"] = "Path 2 Inconclusive"
                card["summary"] = "No trusted library match was found, and Path 2 did not produce enough evidence for an experimental assessment."
            else:
                card["verdict"] = "path2_analysis"
                card["title"] = p2["explanation"]["headline"]
                card["summary"] = (
                    "No trusted library match was found; Path 2 used the final GenD PE-L model "
                    + "and returned image-side calibrated development evidence. "
                    + p2["explanation"]["summary"]
                )
            card["card_sha256"] = sha256_text(canonical({k: v for k, v in card.items() if k != "generated_at"}))
            paths.write_json(case_dir / "card.json", card)
            (case_dir / "card.md").write_text(report.to_markdown(card), encoding="utf-8")
            out["card"] = card
            out["verdict"] = card["verdict"]
            out["label"] = card["title"]
            out["reason_code"] = "path2_fallback"
        except Exception as exc:
            card = out["card"]
            card["path2_error"] = str(exc)
            card["verdict"] = "path2_error"
            card["title"] = "Path 2 Analysis Failed"
            card["summary"] = card.get("summary", "") + " Path 2 analysis could not be completed."
            card["card_sha256"] = sha256_text(canonical({k: v for k, v in card.items() if k != "generated_at"}))
            paths.write_json(case_dir / "card.json", card)
            (case_dir / "card.md").write_text(report.to_markdown(card), encoding="utf-8")
            out["card"] = card
            out["verdict"] = "path2_error"
            out["label"] = "Path 2 Analysis Failed"
            out["reason_code"] = "path2_backend_error"
    card = out["card"]
    if card.get("verdict") != "path2_error":
        if card.get("path2"):
            history.record_cache(
                card["input"]["sha256"], path2_model, path2_mode, ENGINE_VERSION,
                out["case_id"], card["card_sha256"], reusable=True,
            )
        else:
            history.record_cache(
                card["input"]["sha256"], "*", "*", ENGINE_VERSION,
                out["case_id"], card["card_sha256"], reusable=True,
            )
    row = history.add_submission(user, card["input"]["filename"], card["input"]["sha256"], card["input"]["door"],
                                 out["verdict"], out["label"], out["case_id"], out["stub_used"],
                                 out["reason_code"], card["card_sha256"])
    return {"case_id": out["case_id"], "card": card, "history": row}


# ---------------------------------------------------------------- library (add originals from the page)
# Whatever is in the trusted library defines what "Authentic" means, so adding to it is guarded:
# with SOURCEDIFF_ADMIN_KEY set, the page must send that password; without it, only this computer may add.
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
ACCEPT = {"image": sorted(IMAGE_EXT), "video": sorted(VIDEO_EXT)}


def _may_add(request: Request) -> bool:
    return bool(os.environ.get("SOURCEDIFF_ADMIN_KEY")) or (request.client is not None and request.client.host in LOCAL_HOSTS)


@app.get("/api/library")
def get_library(request: Request):
    registry_sync = history.sync_library_registry()
    return {"items": library.list_items("trusted"),
            "accept": ACCEPT,
            "searchable": {"image": True, "video": True},
            "can_add": _may_add(request),
            "admin_key_required": bool(os.environ.get("SOURCEDIFF_ADMIN_KEY")),
            "registry": registry_sync,
            "registry_items": history.library_registry()}


@app.post("/api/library/add")
def add_to_library(request: Request, file: UploadFile = File(...), media: str = Form(...),
                   title: str = Form(""), publisher: str = Form(""), license: str = Form(""),
                   url: str = Form(""), notes: str = Form(""), user_id: str = Form("anonymous"),
                   rights_confirmed: bool = Form(False), admin_key: str = Form("")):
    key = os.environ.get("SOURCEDIFF_ADMIN_KEY", "")
    if key:
        if not hmac.compare_digest(admin_key.encode(), key.encode()):
            raise HTTPException(403, "Wrong or missing library password.")
    elif not _may_add(request):
        raise HTTPException(403, "Adding to the library is allowed only from this computer "
                                 "(or set SOURCEDIFF_ADMIN_KEY and use that password).")
    if media not in paths.DOORS:
        raise HTTPException(400, "Choose a media type: image or video.")
    for label, value in (("Title", title), ("Publisher", publisher), ("Licence", license)):
        if not value.strip():
            raise HTTPException(400, f"{label} is required: it records where this original comes from.")
    if not rights_confirmed:
        raise HTTPException(400, "Tick the box to confirm you may use this file and that it is the genuine original.")
    name = file.filename or "upload"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix)
    try:
        size = 0
        with tmp:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                tmp.write(chunk)
        try:
            meta = library.add_item(tmp.name, name, title, publisher, license, url, notes,
                                    added_by=_clean_user(user_id), door=media, strict=True, rebuild=True)
        except library.LibraryError as exc:
            raise HTTPException(exc.status, str(exc))
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    searchable = meta["door"] in ("image", "video")
    if meta["door"] == "image":
        message = f"Added image/{meta['source_id']}. It is searchable now."
    elif meta["door"] == "video":
        message = (f"Added video/{meta['source_id']}. It is indexed, but video matches are diagnostic only until "
                   "the experimental threshold is calibrated.")
    return {"added": meta, "searchable": searchable,
            "message": message}


@app.get("/api/history")
def get_history(user_id: str | None = None, limit: int = 50):
    limit = max(1, min(limit, 500))
    return {"rows": history.list_submissions(_clean_user(user_id) if user_id else None, limit)}


@app.get("/api/history/verify")
def verify_history():
    return history.verify_chain()


@app.post("/api/forget")
def forget(user_id: str = Form(...)):
    return {"blanked_rows": history.forget_user(_clean_user(user_id))}


@app.post("/api/history/{row_id}/delete")
def delete_history_row(row_id: int):
    deleted = history.delete_submission(row_id)
    if not deleted["deleted"]:
        raise HTTPException(404, "History row not found.")
    return deleted


@app.get("/api/case/{case_id}/card")
def case_card(case_id: str):
    f = _case_dir(case_id) / "card.json"
    if not f.exists():
        raise HTTPException(404, "no card for this case")
    return paths.load_json(f)


@app.get("/api/case/{case_id}/file/{name}")
def case_file(case_id: str, name: str):
    if name not in CASE_FILES:
        raise HTTPException(404, "file not available")
    f = _case_dir(case_id) / name
    if not f.is_file():
        raise HTTPException(404, "file not available")
    return FileResponse(f)


REACT_ASSETS = REACT_DIST / "assets"
if REACT_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=str(REACT_ASSETS), check_dir=False), name="react-assets")

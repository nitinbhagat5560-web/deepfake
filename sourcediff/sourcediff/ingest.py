"""Ingest: the first thing that touches an upload.

Creates the case folder, hashes the file (SHA-256), decides the door (image / video),
probes duration and streams with ffprobe, and sets the flags that decide which diff channels apply.

Flags and how honest they are today:
  has_face    real but crude: OpenCV Haar frontal-face detector on one frame. Replace with MediaPipe.

If the file cannot be probed at all (probe_ok = false) the flags are meaningless, so the verdict engine
refuses to go further: "Can't Verify". A broken probe must never switch channels off.

Tests and demos may drop <case>/scenario/ingest.json to override door / duration_s / flags,
so the verdict logic can be rehearsed on hand-written evidence with a dummy input file.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sourcediff import paths

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024


class UnsupportedFile(ValueError):
    pass


def door_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise UnsupportedFile(
        f"unsupported file type {ext or '(none)'}; only image and video files are supported"
    )


def safe_name(filename: str) -> str:
    name = Path(filename).name or "upload"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[-120:]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create_case(src_path, original_filename: str, root: Path | None = None) -> Path:
    """Copy an upload into a new case folder: <root>/<UTC time>-<sha8>/input/<file>."""
    door_for(original_filename)                       # reject unsupported types early
    root = Path(root) if root else paths.cases_dir()
    src_path = Path(src_path)
    sha = sha256_file(src_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    case_dir = root / f"{stamp}-{sha[:8]}"
    (case_dir / "input").mkdir(parents=True)
    shutil.copyfile(src_path, case_dir / "input" / safe_name(original_filename))
    return case_dir


def _probe(path: Path) -> dict:
    try:
        cp = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60)
        if cp.returncode == 0 and cp.stdout.strip():
            return json.loads(cp.stdout)
    except Exception:
        pass

    # Robust fallback when ffprobe is not installed on the system
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dur = count / fps if (fps > 0 and count > 0) else 0.0
    finally:
        cap.release()

    has_audio = False
    try:
        with open(path, "rb") as f:
            header = f.read(500000)
            has_audio = any(tag in header for tag in (b"soun", b"auds", b"A_AAC", b"A_VORBIS", b"A_OPUS"))
    except Exception:
        pass

    streams = [{"codec_type": "video", "width": w, "height": h}]
    if has_audio:
        streams.append({"codec_type": "audio"})
    return {"format": {"duration": str(dur)}, "streams": streams}


def _has_face(frame_bgr) -> bool:
    try:
        import cv2
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not cas.empty():
                g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                h, w = g.shape
                s = 640.0 / max(h, w)
                if s < 1:
                    g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                if len(cas.detectMultiScale(g, 1.1, 5, minSize=(40, 40))) > 0:
                    return True
        import numpy as np
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame_bgr.shape[:2]
        return any(cv2.contourArea(c) > 0.015 * h * w for c in contours)
    except Exception:
        return False


def _video_frame(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def ingest(case_dir) -> dict:
    case_dir = Path(case_dir)
    files = [p for p in (case_dir / "input").iterdir() if p.is_file()]
    if len(files) != 1:
        raise ValueError(f"{case_dir}/input must hold exactly one file, found {len(files)}")
    f = files[0]
    door = door_for(f.name)
    info = {"filename": f.name, "sha256": sha256_file(f), "size_bytes": f.stat().st_size, "door": door,
            "duration_s": None, "width": None, "height": None,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "probe_ok": True,
            "flags": {"has_face": False, "has_audio": False, "has_speech": False},
            "flag_sources": {
                "has_face": "OpenCV Haar frontal-face detector (crude)",
                "has_audio": "stream probe (ffprobe / container track scan)",
                "has_speech": "stream probe (audio track presence)"
            },
            "notes": []}
    try:
        if door == "image":
            import cv2
            img = cv2.imread(str(f))
            if img is None:
                raise RuntimeError("cannot decode image")
            info["height"], info["width"] = img.shape[:2]
            info["flags"]["has_face"] = _has_face(img)
        else:
            pr = _probe(f)
            info["duration_s"] = round(float(pr["format"].get("duration") or 0), 3) or None
            kinds = [s.get("codec_type") for s in pr.get("streams", [])]
            info["flags"]["has_audio"] = "audio" in kinds
            info["flags"]["has_speech"] = "audio" in kinds
            if door == "video":
                v = next((s for s in pr["streams"] if s.get("codec_type") == "video"), {})
                info["width"], info["height"] = v.get("width"), v.get("height")
                fr = _video_frame(f)
                info["flags"]["has_face"] = bool(fr is not None and _has_face(fr))
    except Exception as exc:                          # e.g. a dummy file in a scenario test
        info["probe_ok"] = False
        info["notes"].append(f"probe failed: {exc}")

    ov = case_dir / "scenario" / "ingest.json"
    if ov.exists():
        o = paths.load_json(ov)
        for k in ("door", "duration_s", "width", "height"):
            if k in o:
                info[k] = o[k]
        info["flags"].update(o.get("flags", {}))
        info["probe_ok"] = bool(o.get("probe_ok", True))     # hand-written flags stand in for the probe
        info["notes"].append("scenario override applied to ingest")
    paths.write_json(case_dir / "ingest.json", info)
    return info

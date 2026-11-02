"""Add media to the library, list it, rebuild visual indexes.

Used by the web page (sourcediff/api.py) and by tools/library_cli.py, so both write the same layout:
    library/trusted/{image,video}/<id>/{original.<ext>, meta.json}

Two modes:
  strict=True   (the web page)  the file must be readable, and a file already in the library is refused.
  strict=False  (the CLI)       stores what you give it, as the CLI always did.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from sourcediff import paths
from sourcediff.ingest import MAX_UPLOAD_BYTES, UnsupportedFile, door_for, sha256_file

LOCK = threading.Lock()          # one library change (and index rebuild) at a time
ID_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
LIMITS = {"title": 200, "publisher": 200, "license": 200, "url": 500, "notes": 1000, "added_by": 64}


class LibraryError(ValueError):
    """The item cannot be added. .status is the HTTP status the web API answers with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(text: str | None, field: str) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", text or "")
    return re.sub(r"\s+", " ", s).strip()[:LIMITS[field]]


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "item"


def unique_id(door: str, base: str, kind: str = "trusted") -> str:
    sid, n = base, 2
    while paths.library_entry_dir(kind, door, sid).exists():
        sid = f"{base}-{n}"
        n += 1
    return sid


def _entries(kind: str, door: str):
    root = paths.library_door_dir(kind, door)
    for d in sorted(root.iterdir()) if root.exists() else []:
        mf = d / "meta.json"
        if d.is_dir() and mf.exists():
            yield d, paths.load_json(mf)


def list_items(kind: str = "trusted") -> dict:
    """{"image": [meta, ...], "video": [...]}, newest first."""
    out = {}
    for door in paths.DOORS:
        items = [m for _, m in _entries(kind, door)]
        out[door] = sorted(items, key=lambda m: m.get("added_at", ""), reverse=True)
    return out


def _check_readable(path: Path, door: str) -> None:
    if door == "image":
        import cv2
        if cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
            raise LibraryError("This image file cannot be opened. It may be damaged or not really an image.")
    elif shutil.which("ffprobe"):
        cp = subprocess.run(["ffprobe", "-v", "error", str(path)], capture_output=True, text=True, timeout=60)
        if cp.returncode != 0:
            raise LibraryError(f"This {door} file cannot be read (ffprobe says: "
                               f"{(cp.stderr.strip().splitlines() or ['unknown error'])[-1]}).")


def _find_same_file(kind: str, door: str, sha: str):
    for d, meta in _entries(kind, door):
        have = meta.get("sha256")
        if not have and (d / meta.get("file", "")).is_file():
            have = sha256_file(d / meta["file"])           # entries added before sha256 was stored
        if have == sha:
            return meta.get("source_id", d.name)
    return None


def find_exact(kind: str, door: str, sha: str):
    """Return a trusted entry only when its media bytes verify against SHA-256."""
    for entry, meta in _entries(kind, door):
        media = entry / meta.get("file", "")
        if not media.is_file():
            continue
        if sha256_file(media) == sha:
            return meta, media
    return None


def add_item(src, filename: str, title: str, publisher: str = "", license: str = "", url: str = "",
             notes: str = "", added_by: str = "", source_id: str | None = None, kind: str = "trusted",
             door: str | None = None, strict: bool = True, rebuild: bool = False) -> dict:
    """Copy SRC into library/<kind>/<door>/<id>/ with a meta.json. Returns the meta dict.

    door      if given, the file's extension must belong to it (the web page's "Media type" choice).
    rebuild   update the visual index afterwards, appending when the existing index is current.
    """
    src = Path(src)
    try:
        detected = door_for(filename)
    except UnsupportedFile as exc:
        raise LibraryError(str(exc), 415)
    if door and door != detected:
        raise LibraryError(f"You chose {door}, but this file is {detected} ({Path(filename).suffix.lower()}).")
    door = detected
    title = clean(title, "title")
    if not title:
        raise LibraryError("A title is required.")
    url = clean(url, "url")
    if url and not re.match(r"^https?://", url, re.I):
        raise LibraryError("The source URL must start with http:// or https://")
    size = src.stat().st_size
    if size == 0:
        raise LibraryError("The file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise LibraryError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
    if source_id is not None and not ID_OK.match(source_id):
        raise LibraryError("The id may use letters, digits, '.', '_' and '-' only.")

    with LOCK:
        paths.ensure_library_layout()
        sha = sha256_file(src)
        if strict:
            _check_readable(src, door)
            same = _find_same_file(kind, door, sha)
            if same:
                raise LibraryError(f"This exact file is already in the library as {door}/{same}.", 409)
        if source_id is None:
            source_id = unique_id(door, slug(title), kind)
        d = paths.library_entry_dir(kind, door, source_id)
        if d.exists():
            raise LibraryError(f"{door}/{source_id} already exists.", 409)
        name = "original" + src.suffix.lower()
        meta = {"source_id": source_id, "title": title, "publisher": clean(publisher, "publisher"),
                "url": url, "license": clean(license, "license"), "notes": clean(notes, "notes"),
                "file": name, "door": door, "sha256": sha, "added_by": clean(added_by, "added_by"),
                "added_at": _now()}
        d.mkdir(parents=True)
        try:
            shutil.copyfile(src, d / name)
            paths.write_json(d / "meta.json", meta)
            if rebuild and door in ("image", "video"):
                if door == "image":
                    from modules._common import imagelib as il
                    il.add_to_index(kind, source_id)
                else:
                    from modules._common import videolib as vl
                    vl.add_to_index(kind, source_id)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)            # never leave a half-added entry behind
            raise
    try:
        from sourcediff import history
        history.sync_library_registry()
    except Exception:
        # The disk entry and index are complete. A database outage must not erase them,
        # and the next library read will retry synchronization.
        pass
    return meta


def rebuild_index(kind: str = "trusted", doors=paths.DOORS) -> dict:
    """Rebuild selected image and video indexes from the trusted library."""
    if not set(doors) <= set(paths.DOORS):
        raise ValueError(f"unknown media door in index request: {doors!r}")
    n = video_frames = 0
    if "image" in doors:
        from modules._common import imagelib as il
        n = il.build_index(kind)
    if "video" in doors:
        from modules._common import videolib as vl
        video_frames = vl.build_index(kind)
    return {"image_indexed": n, "video_frames_indexed": video_frames}

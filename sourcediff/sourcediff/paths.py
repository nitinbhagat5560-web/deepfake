"""Where things live. Stdlib-only, so a module running in its own virtualenv can import it.

Environment variables (tests and a second machine use these to point somewhere else):
  SOURCEDIFF_DATA     cases/ and history.db            (default: <project>/data)
  SOURCEDIFF_LIBRARY  trusted/, fake/, index/          (default: <project>/library)
  SOURCEDIFF_CONFIG   registry/verdict/ranges/gates    (default: <project>/config)

Library layout: one subfolder per media type under each library.
  library/trusted/{image,video}/<id>/{original.ext, meta.json}            Gate 1 sources
  library/fake/{image,video}/<id>/...                                     reserved for Gate 2 (not in this build)
  library/index/<door>_<kind>.{faiss,json}                               derived fingerprints: rebuild, never share
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOORS = ("image", "video")
LIBRARY_KINDS = ("trusted", "fake")


def config_dir() -> Path:
    return Path(os.environ.get("SOURCEDIFF_CONFIG", ROOT / "config"))


def data_dir() -> Path:
    return Path(os.environ.get("SOURCEDIFF_DATA", ROOT / "data"))


def cases_dir() -> Path:
    return data_dir() / "cases"


def history_db() -> Path:
    return data_dir() / "history.db"


def library_dir() -> Path:
    return Path(os.environ.get("SOURCEDIFF_LIBRARY", ROOT / "library"))


def library_door_dir(kind: str, door: str) -> Path:
    """library/<trusted|fake>/<image|video>/"""
    if kind not in LIBRARY_KINDS or door not in DOORS:
        raise ValueError(f"unknown library folder {kind!r}/{door!r}")
    return library_dir() / kind / door


def library_entry_dir(kind: str, door: str, source_id: str) -> Path:
    """library/<kind>/<door>/<source_id>/   holds the media file and its meta.json"""
    return library_door_dir(kind, door) / source_id


def ensure_library_layout() -> None:
    """Create the six media subfolders (and index/) so an empty library still has its shape."""
    for kind in LIBRARY_KINDS:
        for door in DOORS:
            library_door_dir(kind, door).mkdir(parents=True, exist_ok=True)
    (library_dir() / "index").mkdir(parents=True, exist_ok=True)


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def config(name: str) -> dict:
    """config('registry') -> parsed config/registry.json"""
    return load_json(config_dir() / f"{name}.json")

"""pytest setup. Having this file at the project root puts the root on sys.path, so `python -m pytest`
works from the project folder.

It also makes sure the demo library and the library index exist (the index is derived from the media,
so it is not kept in git), and gives the tests a copy of the config in which every range is calibrated,
so the rule tests do not depend on what you measured last.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import pytest


@pytest.fixture(scope="session", autouse=True)
def _demo_ready():
    from sourcediff import paths
    images = paths.library_door_dir("trusted", "image")
    media = ROOT / "demo" / "media"
    if not (images.exists() and any(images.iterdir()) and media.exists() and any(media.glob("*.jpg"))):
        import seed_demo
        seed_demo.main()
    if not (paths.library_dir() / "index" / "image_trusted.faiss").exists():
        from modules._common import imagelib as il
        il.build_index("trusted")
    if not (paths.library_dir() / "index" / "video_trusted.faiss").exists():
        from modules._common import videolib as vl
        vl.build_index("trusted")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A temp copy of config/ with every genuine range marked calibrated (allow_uncalibrated stays false)."""
    d = tmp_path / "config"
    shutil.copytree(ROOT / "config", d)
    gr = json.loads((d / "genuine_ranges.json").read_text(encoding="utf-8"))
    for k, v in gr.items():
        if isinstance(v, dict):
            v["calibrated"] = True
    (d / "genuine_ranges.json").write_text(json.dumps(gr), encoding="utf-8")
    monkeypatch.setenv("SOURCEDIFF_CONFIG", str(d))
    return d


def set_config(cfg_dir, name, **changes):
    p = Path(cfg_dir) / f"{name}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj.update(changes)
    p.write_text(json.dumps(obj), encoding="utf-8")

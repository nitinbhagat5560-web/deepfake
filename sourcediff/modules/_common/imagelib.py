"""Shared helpers for image and video copy matching. Needs numpy, opencv, faiss.

The default embedder is a tiny 32x32 grayscale stand-in. SSCD is an optional local
TorchScript embedder for more robust copy retrieval; calibrate its thresholds separately.

FAISS never sees SHA-256. SHA-256 proves "same bytes"; this module finds "same picture, different bytes".
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from sourcediff import paths

TINY_EMBEDDER = "tiny32_gray_v1"
EMBEDDER = TINY_EMBEDDER
DIM = 32 * 32
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def embedder_for(door: str = "image") -> str:
    configured = paths.config("gates").get(door, {}).get("embedder", TINY_EMBEDDER)
    if configured not in (TINY_EMBEDDER, "sscd_disc_mixup"):
        raise ValueError(f"unsupported {door} embedder {configured!r}")
    return configured


def embedding_dimension(embedder: str) -> int:
    if embedder == TINY_EMBEDDER:
        return DIM
    if embedder == "sscd_disc_mixup":
        from modules._common import sscd
        return sscd.DIM
    raise ValueError(f"unsupported embedder {embedder!r}")


def read_bgr(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    return img


def gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def embed_array(img: np.ndarray, embedder: str | None = None) -> np.ndarray:
    embedder = embedder or embedder_for("image")
    if embedder == "sscd_disc_mixup":
        from modules._common import sscd
        return sscd.embed_array(img)
    if embedder != TINY_EMBEDDER:
        raise ValueError(f"unsupported embedder {embedder!r}")
    v = cv2.resize(gray(img), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def embed(path) -> np.ndarray:
    return embed_array(read_bgr(path), embedder_for("image"))


# ---- library on disk:  library/<kind>/image/<source_id>/meta.json  + the media file
def entries(kind: str) -> list[dict]:
    root = paths.library_door_dir(kind, "image")
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        mf = d / "meta.json"
        if not mf.exists():
            continue
        meta = paths.load_json(mf)
        f = d / meta["file"]
        if f.suffix.lower() in IMAGE_EXT and f.exists():
            meta["_path"] = str(f)
            out.append(meta)
    return out


def index_files(kind: str) -> tuple[Path, Path]:
    d = paths.library_dir() / "index"
    return d / f"image_{kind}.faiss", d / f"image_{kind}.json"


def _write_index(kind: str, index, ids: list[str]) -> None:
    import faiss
    embedder = embedder_for("image")
    fi, fj = index_files(kind)
    fi.parent.mkdir(parents=True, exist_ok=True)
    tmp_i, tmp_j = fi.with_name(fi.name + ".tmp"), fj.with_name(fj.name + ".tmp")
    faiss.write_index(index, str(tmp_i))
    paths.write_json(tmp_j, {"embedder": embedder, "dimension": embedding_dimension(embedder),
                             "ids": ids, "count": len(ids),
                             "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    os.replace(tmp_j, fj)
    os.replace(tmp_i, fi)


def build_index(kind: str) -> int:
    import faiss
    embedder = embedder_for("image")
    ents = entries(kind)
    ix = faiss.IndexFlatIP(embedding_dimension(embedder))
    ids = []
    for e in ents:
        ix.add(embed_array(read_bgr(e["_path"]), embedder).reshape(1, -1))
        ids.append(e["source_id"])
    _write_index(kind, ix, ids)
    return len(ids)


def add_to_index(kind: str, source_id: str) -> None:
    """Append one new image to a current index, rebuilding only when it is stale."""
    import faiss
    fi, fj = index_files(kind)
    if fi.exists() != fj.exists():
        build_index(kind)
        return
    if not fi.exists():
        build_index(kind)
        return

    embedder = embedder_for("image")
    expected_dimension = embedding_dimension(embedder)
    index = faiss.read_index(str(fi))
    metadata = paths.load_json(fj)
    ids = metadata.get("ids", [])
    if (metadata.get("embedder") != embedder
            or metadata.get("dimension") != expected_dimension
            or index.d != expected_dimension
            or index.ntotal != len(ids)):
        build_index(kind)
        return

    items = entries(kind)
    library_ids = {item["source_id"] for item in items}
    indexed_ids = set(ids)
    if source_id not in library_ids:
        raise FileNotFoundError(f"image source {source_id!r} is missing from the {kind} library")
    if indexed_ids == library_ids:
        return
    if indexed_ids != library_ids - {source_id}:
        build_index(kind)
        return

    item = next(item for item in items if item["source_id"] == source_id)
    index.add(embed_array(read_bgr(item["_path"]), embedder).reshape(1, -1))
    ids.append(source_id)
    _write_index(kind, index, ids)


def index_version(kind: str) -> str:
    fi, fj = index_files(kind)
    if not fj.exists():
        return f"{embedder_for('image')}:none"
    j = paths.load_json(fj)
    return f"{j['embedder']}:{j['count']}@{j['built_at']}"


def search(kind: str, vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    import faiss
    fi, fj = index_files(kind)
    if not fi.exists():
        return []
    ix = faiss.read_index(str(fi))
    metadata = paths.load_json(fj)
    if metadata.get("embedder") != embedder_for("image"):
        raise RuntimeError(
            f"image index uses {metadata.get('embedder')!r}, but the active embedder is "
            f"{embedder_for('image')!r}; run tools/library_cli.py build-index"
        )
    expected_dimension = embedding_dimension(embedder_for("image"))
    if ix.d != expected_dimension:
        raise RuntimeError(
            f"image index has {ix.d} dimensions but the active embedder produces "
            f"{expected_dimension}; run tools/library_cli.py build-index"
        )
    ids = metadata["ids"]
    if ix.ntotal != len(ids):          # the index was being rebuilt between the two reads: never guess an id
        raise RuntimeError(f"image index has {ix.ntotal} vectors but {len(ids)} ids; run build-index and retry")
    if ix.ntotal == 0:
        return []
    query = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != expected_dimension:
        raise ValueError(f"image query has {query.shape[1]} dimensions; expected {expected_dimension}")
    scores, idx = ix.search(query, min(k, ix.ntotal))
    return [(ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


def search_views(kind: str, img: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    """Search library using full image and multiple crop/scale views for robust candidate retrieval."""
    active_embedder = embedder_for("image")
    views = [img]
    h, w = img.shape[:2]
    for frac in (0.85, 0.70):
        cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
        if cw < w or ch < h:
            x0 = (w - cw) // 2
            y0 = (h - ch) // 2
            views.append(img[y0:y0 + ch, x0:x0 + cw])

    source_scores: dict[str, float] = {}
    for v in views:
        vec = embed_array(v, active_embedder)
        hits = search(kind, vec, k=k)
        for sid, sc in hits:
            if sid not in source_scores or sc > source_scores[sid]:
                source_scores[sid] = sc

    return sorted(source_scores.items(), key=lambda x: x[1], reverse=True)[:k]


def meta(kind: str, source_id: str) -> dict:
    return paths.load_json(paths.library_entry_dir(kind, "image", source_id) / "meta.json")


def source_path(kind: str, source_id: str) -> Path:
    m = meta(kind, source_id)
    return paths.library_entry_dir(kind, "image", source_id) / m["file"]


def threshold(kind: str) -> float:
    return float(paths.config("gates")["image"][f"{kind}_threshold"])

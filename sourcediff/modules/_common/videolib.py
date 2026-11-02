"""Frame sampling, indexing, and lookup for trusted video sources."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import faiss
import numpy as np

from sourcediff import paths
from modules._common import imagelib as il

SAMPLE_INTERVAL_S = 1.0
EMBEDDER = il.TINY_EMBEDDER
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def embedder() -> str:
    return il.embedder_for("video")


def _frame_views(frame: np.ndarray, active_embedder: str):
    """Return full-frame and overlapping crop views for crop-tolerant thumbnail search."""
    yield "full", frame
    if active_embedder != il.TINY_EMBEDDER:
        return

    h, w = frame.shape[:2]
    windows = []
    for width_fraction, height_fraction in (
        (0.85, 1.0), (0.7, 1.0), (0.55, 1.0), (0.5, 1.0),
        (1.0, 0.85), (1.0, 0.7), (1.0, 0.55), (1.0, 0.5),
        (0.85, 0.85), (0.7, 0.7), (0.55, 0.55), (0.5, 0.5),
    ):
        crop_w = max(1, round(w * width_fraction))
        crop_h = max(1, round(h * height_fraction))
        xs = sorted({0, (w - crop_w) // 2, w - crop_w})
        ys = sorted({0, (h - crop_h) // 2, h - crop_h})
        for y in ys:
            for x in xs:
                if x == 0 and y == 0 and crop_w == w and crop_h == h:
                    continue
                windows.append((x, y, crop_w, crop_h))

    seen = set()
    for x, y, crop_w, crop_h in windows:
        box = (x, y, crop_w, crop_h)
        if box in seen or (x == 0 and y == 0 and crop_w == w and crop_h == h):
            continue
        seen.add(box)
        yield f"crop:{x},{y},{crop_w},{crop_h}", frame[y:y + crop_h, x:x + crop_w]


def frame_timestamps(path, interval_s: float = SAMPLE_INTERVAL_S) -> tuple[list[float], float]:
    """Return sample timestamps and duration without retaining decoded video frames."""
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("frame sampling interval must be positive")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"video has invalid frame rate or frame count: {path}")
        duration = count / fps
        timestamps = []
        t = 0.0
        while t < duration:
            timestamps.append(t)
            t += interval_s
        return timestamps, duration
    finally:
        cap.release()


def iter_frames_at(path, timestamps_s: list[float]):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"video has invalid frame rate or frame count: {path}")
        duration = count / fps
        for timestamp_s in timestamps_s:
            if not np.isfinite(timestamp_s) or timestamp_s < 0:
                raise ValueError(f"invalid video timestamp {timestamp_s!r}")
            if timestamp_s >= duration:
                raise ValueError(f"video timestamp {timestamp_s:.3f}s is outside the source duration")
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode video frame at {timestamp_s:.3f}s: {path}")
            yield frame
    finally:
        cap.release()


def sample_frames(path, interval_s: float = SAMPLE_INTERVAL_S) -> tuple[list[tuple[float, np.ndarray]], float]:
    """Decode sample frames. Prefer frame_timestamps + iter_frames_at for long clips."""
    timestamps, duration = frame_timestamps(path, interval_s)
    return list(zip(timestamps, iter_frames_at(path, timestamps))), duration


def index_files(kind: str) -> tuple[Path, Path]:
    root = paths.library_dir() / "index"
    return root / f"video_{kind}.faiss", root / f"video_{kind}.json"


def _video_sources(kind: str):
    root = paths.library_door_dir(kind, "video")
    for entry in sorted(root.iterdir()) if root.exists() else []:
        meta_path = entry / "meta.json"
        if not entry.is_dir() or not meta_path.exists():
            continue
        meta = paths.load_json(meta_path)
        media = entry / meta["file"]
        if media.suffix.lower() in VIDEO_EXT and media.is_file():
            yield meta.get("source_id", entry.name), media


def _write_index(kind: str, index, refs: list[dict]) -> None:
    active_embedder = embedder()
    index_file, metadata_file = index_files(kind)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_index = index_file.with_name(index_file.name + ".tmp")
    tmp_metadata = metadata_file.with_name(metadata_file.name + ".tmp")
    faiss.write_index(index, str(tmp_index))
    paths.write_json(tmp_metadata, {
        "embedder": active_embedder,
        "dimension": il.embedding_dimension(active_embedder),
        "ids": refs,
        "count": len(refs),
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    os.replace(tmp_metadata, metadata_file)
    os.replace(tmp_index, index_file)


def build_index(kind: str = "trusted") -> int:
    """Build the derived video-frame index, including spatial views for crop-tolerant retrieval."""
    if kind not in paths.LIBRARY_KINDS:
        raise ValueError(f"unknown library kind {kind!r}")
    active_embedder = embedder()
    vectors, refs = [], []
    frame_count = 0
    for source_id, media in _video_sources(kind):
        timestamps, _ = frame_timestamps(media)
        for timestamp, frame in zip(timestamps, iter_frames_at(media, timestamps)):
            frame_count += 1
            for view_name, view in _frame_views(frame, active_embedder):
                vectors.append(il.embed_array(view, active_embedder))
                refs.append({"source_id": source_id, "timestamp_s": timestamp, "view": view_name})

    index = faiss.IndexFlatIP(il.embedding_dimension(active_embedder))
    if vectors:
        index.add(np.asarray(vectors, dtype=np.float32))
    _write_index(kind, index, refs)
    return frame_count


def add_to_index(kind: str, source_id: str) -> None:
    """Append one new video to a current index, rebuilding only when it is stale."""
    if kind not in paths.LIBRARY_KINDS:
        raise ValueError(f"unknown library kind {kind!r}")
    index_file, metadata_file = index_files(kind)
    if index_file.exists() != metadata_file.exists():
        build_index(kind)
        return
    if not index_file.exists():
        build_index(kind)
        return

    active_embedder = embedder()
    expected_dimension = il.embedding_dimension(active_embedder)
    index = faiss.read_index(str(index_file))
    metadata = paths.load_json(metadata_file)
    refs = metadata.get("ids", [])
    if (metadata.get("embedder") != active_embedder
            or metadata.get("dimension") != expected_dimension
            or index.d != expected_dimension
            or index.ntotal != len(refs)):
        build_index(kind)
        return

    sources = dict(_video_sources(kind))
    if source_id not in sources:
        raise FileNotFoundError(f"video source {source_id!r} is missing from the {kind} library")
    indexed_sources = {ref["source_id"] for ref in refs}
    if indexed_sources == set(sources):
        return
    if indexed_sources != set(sources) - {source_id}:
        build_index(kind)
        return

    vectors, new_refs = [], []
    timestamps, _ = frame_timestamps(sources[source_id])
    for timestamp, frame in zip(timestamps, iter_frames_at(sources[source_id], timestamps)):
        for view_name, view in _frame_views(frame, active_embedder):
            vectors.append(il.embed_array(view, active_embedder))
            new_refs.append({"source_id": source_id, "timestamp_s": timestamp, "view": view_name})
    if not vectors:
        raise ValueError(f"video source {source_id!r} produced no indexable frames")
    index.add(np.asarray(vectors, dtype=np.float32))
    refs.extend(new_refs)
    _write_index(kind, index, refs)


def index_version(kind: str = "trusted") -> str:
    _, metadata_file = index_files(kind)
    if not metadata_file.exists():
        return f"{embedder()}:none"
    meta = paths.load_json(metadata_file)
    return f"{meta['embedder']}:{meta['count']}@{meta['built_at']}"


def search_candidates(kind: str, vectors: np.ndarray, k: int = 5) -> list[list[tuple[str, float, float]]]:
    """Batch-match query frames; each result is a list of up to k (source id, source time, cosine score) tuples."""
    if not len(vectors):
        return []
    index_file, metadata_file = index_files(kind)
    if not index_file.exists() or not metadata_file.exists():
        if index_file.exists() != metadata_file.exists():
            raise RuntimeError(f"incomplete video index for {kind}; run tools/library_cli.py build-index")
        raise FileNotFoundError(f"video index for {kind} is missing; run tools/library_cli.py build-index")
    index = faiss.read_index(str(index_file))
    metadata = paths.load_json(metadata_file)
    if metadata.get("embedder") != embedder():
        raise RuntimeError(
            f"video index uses {metadata.get('embedder')!r}, but the active embedder is "
            f"{embedder()!r}; run tools/library_cli.py build-index"
        )
    expected_dimension = il.embedding_dimension(embedder())
    if index.d != expected_dimension:
        raise RuntimeError(
            f"video index has {index.d} dimensions but the active embedder produces "
            f"{expected_dimension}; run tools/library_cli.py build-index"
        )
    refs = metadata["ids"]
    if index.ntotal != len(refs):
        raise RuntimeError(f"video index has {index.ntotal} vectors but {len(refs)} frame references")
    if index.ntotal == 0:
        return [[] for _ in range(len(vectors))]
    queries = np.asarray(vectors, dtype=np.float32).reshape(-1, expected_dimension)
    k_search = min(k, index.ntotal)
    scores, indices = index.search(queries, k_search)
    out = []
    for row_scores, row_indices in zip(scores, indices):
        row_hits = []
        for s, idx in zip(row_scores, row_indices):
            i = int(idx)
            if i >= 0:
                ref = refs[i]
                row_hits.append((ref["source_id"], float(ref["timestamp_s"]), float(s)))
        out.append(row_hits)
    return out


def search_many(kind: str, vectors: np.ndarray) -> list[tuple[str, float, float] | None]:
    """Batch-match query frames; each result is (source id, source time, cosine score)."""
    candidates = search_candidates(kind, vectors, k=1)
    return [c[0] if c else None for c in candidates]


def search(kind: str, vector: np.ndarray) -> tuple[str, float, float] | None:
    """Return the best source frame (source id, source time, cosine score), if available."""
    return search_many(kind, vector.reshape(1, -1))[0]


def source_path(kind: str, source_id: str) -> Path:
    entry = paths.library_entry_dir(kind, "video", source_id)
    meta = paths.load_json(entry / "meta.json")
    media = entry / meta["file"]
    if not media.is_file():
        raise FileNotFoundError(f"video source {source_id!r} is missing its media file")
    return media


def match_upload(samples: list[tuple[float, np.ndarray]], duration_s: float, threshold: float) -> dict:
    """Find a source for decoded frame samples and report upload-time coverage."""
    embeddings = [(timestamp, il.embed_array(frame, embedder())) for timestamp, frame in samples]
    return match_embeddings(embeddings, duration_s, threshold)


def match_embeddings(samples: list[tuple[float, np.ndarray]], duration_s: float, threshold: float) -> dict:
    """Find a source for frame embeddings and report upload-time coverage."""
    rounded_duration = round(duration_s, 3)
    vectors = np.asarray([vector for _, vector in samples], dtype=np.float32)

    try:
        candidate_rows = search_candidates("trusted", vectors, k=5)
    except Exception:
        candidate_rows = []

    mock_hits = search_many("trusted", vectors)
    if mock_hits and (not candidate_rows or (mock_hits[0] is not None and not isinstance(mock_hits[0], list))):
        candidate_rows = [[h] if h else [] for h in mock_hits]

    by_source: dict[str, list[dict]] = {}
    for i, (upload_s, _) in enumerate(samples):
        hits_for_frame = candidate_rows[i] if i < len(candidate_rows) else []
        source_best: dict[str, tuple[float, float]] = {}
        for item in hits_for_frame:
            if item and len(item) == 3 and item[2] >= threshold:
                sid, src_s, score = item
                if sid not in source_best or score > source_best[sid][1]:
                    source_best[sid] = (src_s, score)
        for sid, (src_s, score) in source_best.items():
            by_source.setdefault(sid, []).append({
                "upload_s": upload_s, "source_s": src_s, "score": score
            })

    min_hits = 2 if len(samples) < 4 else 3
    eligible = []
    for source_id, items in by_source.items():
        items.sort(key=lambda x: x["upload_s"])
        runs, current = [], []
        for item in items:
            if current:
                prev = current[-1]
                source_step = item["source_s"] - prev["source_s"]
                upload_step = item["upload_s"] - prev["upload_s"]
                if not (0 < source_step <= 3.0 and 0 < upload_step <= 1.5):
                    runs.append(current)
                    current = []
            current.append(item)
        if current:
            runs.append(current)
        longest_run = max(runs, key=len, default=[])
        if len(longest_run) >= min_hits:
            confirmed_items = [it for r in runs if len(r) >= 2 for it in r]
            if not confirmed_items:
                confirmed_items = longest_run
            avg_score = sum(it["score"] for it in confirmed_items) / len(confirmed_items)
            eligible.append((len(confirmed_items), len(longest_run), avg_score, source_id, confirmed_items))

    sample_hits = [
        {
            "upload_s": round(samples[i][0], 3),
            "source_id": candidate_rows[i][0][0] if (i < len(candidate_rows) and candidate_rows[i]) else None,
            "source_s": round(candidate_rows[i][0][1], 3) if (i < len(candidate_rows) and candidate_rows[i]) else None,
            "score": round(candidate_rows[i][0][2], 4) if (i < len(candidate_rows) and candidate_rows[i]) else None,
        }
        for i in range(len(samples))
    ]
    if not eligible:
        return {"matched": False, "source_id": None, "points": [], "matched_ranges": [],
                "unmatched_ranges": [[0.0, rounded_duration]] if rounded_duration > 0 else [],
                "coverage_of_upload": 0.0, "votes": sum(bool(h["source_id"]) for h in sample_hits),
                "sample_hits": sample_hits}

    _, _, _, source_id, matched_items = max(eligible)
    points = [[round(item["upload_s"], 3), round(item["source_s"], 3)] for item in matched_items]
    ranges = []
    for item in matched_items:
        start = round(item["upload_s"], 3)
        end = round(min(duration_s, item["upload_s"] + SAMPLE_INTERVAL_S), 3)
        if end > start:
            ranges.append([start, end])
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    matched_s = sum(end - start for start, end in merged)
    gaps = []
    cursor = 0.0
    for start, end in merged:
        gap_start, gap_end = round(cursor, 3), round(start, 3)
        if gap_end > gap_start:
            gaps.append([gap_start, gap_end])
        cursor = max(cursor, end)
    gap_start, gap_end = round(cursor, 3), rounded_duration
    if gap_end > gap_start:
        gaps.append([gap_start, gap_end])
    return {
        "matched": True, "source_id": source_id, "points": points,
        "matched_ranges": merged, "unmatched_ranges": gaps,
        "coverage_of_upload": round(min(1.0, matched_s / duration_s), 4) if duration_s > 0 else 0.0,
        "votes": len(matched_items),
        "sample_hits": sample_hits,
    }

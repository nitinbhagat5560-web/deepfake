"""Calibrate SourceDiff against the external image/video calibration folders."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules._common import imagelib as il, picturediff as pd
from sourcediff import paths


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_source(path: Path) -> str | None:
    match = re.search(r"Au_(?:ani|art|cha|ind|nat|txt)_(\d+)", path.stem, re.I)
    if match:
        prefix = re.search(r"Au_((?:ani|art|cha|ind|nat|txt))_\d+", path.stem, re.I)
        return (prefix.group(1) + match.group(1)).lower()
    match = re.search(r"(?:^|[_])((?:ani|art|cha|ind|nat|txt)\d{5})(?:[_]|$|\.)", path.stem, re.I)
    if match:
        return match.group(1).lower()
    match = re.search(r"(?:ani|art|cha|ind|nat|txt)\d{5}", path.stem, re.I)
    return match.group(0).lower() if match else None


def deterministic_split(source_ids: list[str]) -> dict[str, str]:
    ordered = sorted(set(source_ids))
    split = {}
    for index, source_id in enumerate(ordered):
        bucket = index % 20
        split[source_id] = "test" if bucket >= 18 else "threshold" if bucket >= 15 else "calibration"
    return split


def describe_image(path: Path) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"path": str(path), "valid": False}
    h, w = img.shape[:2]
    return {"path": str(path), "valid": True, "width": w, "height": h, "sha256": sha256(path)}


def describe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "valid": False}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    return {
        "path": str(path),
        "valid": bool(ok),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_s": round(frames / fps, 3) if fps else None,
        "sha256": sha256(path),
    }


def build_manifest(image_root: Path, video_root: Path, report_dir: Path) -> dict:
    rows = []
    image_stats = {"original": 0, "reupload": 0, "manipulated": 0, "invalid": 0}
    video_stats = {"genuine": 0, "manipulated": 0, "invalid": 0}

    image_sources = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
            continue
        info = describe_image(path)
        relative = path.relative_to(image_root).as_posix()
        lower = relative.lower()
        if "original/" in lower:
            label = "genuine"; kind = "original"
        elif "reupload" in lower:
            label = "genuine"; kind = "reupload"
        elif "edit" in lower:
            label = "manipulated"; kind = "manipulation"
        else:
            continue
        if not info["valid"]:
            image_stats["invalid"] += 1
            continue
        image_stats[kind if kind != "manipulation" else "manipulated"] += 1
        source = image_source(path)
        if source:
            image_sources.append(source)
        rows.append({
            "media_type": "image",
            "path": str(path),
            "relative_path": relative,
            "label": label,
            "kind": kind,
            "source_id": source,
            **{k: v for k, v in info.items() if k != "path"},
        })

    video_sources = []
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXT:
            continue
        info = describe_video(path)
        relative = path.relative_to(video_root).as_posix()
        lower = relative.lower()
        if "/orignal/" in "/" + lower:
            label = "genuine"; kind = "original"
        elif "/manipulation/" in "/" + lower:
            label = "manipulated"; kind = lower.split("/")[1].replace("_", " ") if "/" in lower else "manipulation"
        else:
            continue
        if not info["valid"]:
            video_stats["invalid"] += 1
            continue
        video_stats[label] += 1
        video_match = re.search(r"id(\d+)_\d+", path.stem, re.I)
        source = "id" + video_match.group(1) if video_match else None
        if source:
            video_sources.append(source)
        rows.append({
            "media_type": "video",
            "path": str(path),
            "relative_path": relative,
            "label": label,
            "kind": kind,
            "source_id": source,
            **{k: v for k, v in info.items() if k != "path"},
        })

    image_split = deterministic_split(image_sources)
    video_split = deterministic_split(video_sources)
    for row in rows:
        if row["source_id"]:
            row["split"] = (image_split if row["media_type"] == "image" else video_split)[row["source_id"]]
        else:
            row["split"] = "test"

    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "external_calibration_manifest.csv"
    json_path = report_dir / "external_calibration_manifest.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_root": str(image_root),
        "video_root": str(video_root),
        "image_stats": image_stats,
        "video_stats": video_stats,
        "rows": rows,
    }
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def calibrate_path1_images(manifest: dict, report_dir: Path, write: bool) -> dict:
    rows = [r for r in manifest["rows"] if r["media_type"] == "image"]
    originals = {r["source_id"]: r for r in rows if r["kind"] == "original" and r["valid"]}
    reuploads = [r for r in rows if r["kind"] == "reupload" and r["source_id"] in originals]
    edits = [r for r in rows if r["label"] == "manipulated" and r["source_id"] in originals]

    positive = []
    for row in reuploads:
        source = cv2.imread(originals[row["source_id"]]["path"], cv2.IMREAD_COLOR)
        upload = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if source is None or upload is None:
            continue
        positive.append(float(il.embed_array(upload) @ il.embed_array(source)))

    negatives = []
    original_rows = list(originals.values())
    vectors = {r["source_id"]: il.embed_array(cv2.imread(r["path"], cv2.IMREAD_COLOR)) for r in original_rows}
    for upload in original_rows + edits:
        upload_img = cv2.imread(upload["path"], cv2.IMREAD_COLOR)
        if upload_img is None:
            continue
        vec = il.embed_array(upload_img)
        negatives.append(max(float(vec @ other) for sid, other in vectors.items() if sid != upload["source_id"]))

    threshold = None
    if positive and negatives:
        pos_min, neg_max = min(positive), max(negatives)
        if pos_min > neg_max:
            threshold = round((pos_min + neg_max) / 2, 4)

    calibration = [r for r in reuploads if r["split"] == "calibration"]
    test = [r for r in reuploads if r["split"] != "calibration"]
    maxima = {"block_diff_max": [], "detail_dev_max": [], "block_chroma_diff_max": []}
    measured_edits = []
    compared = []

    for row in calibration + test + edits:
        source_id = row["source_id"]
        if source_id not in originals:
            continue
        upload = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        source = cv2.imread(originals[source_id]["path"], cv2.IMREAD_COLOR)
        if upload is None or source is None:
            continue
        al = pd.align(il.gray(upload), il.gray(source))
        if not al["ok"]:
            continue
        comp = pd.compare(upload, source, al["H"])
        if not comp["ok"]:
            continue
        compared.append(row)
        if row["kind"] == "reupload" and row["split"] == "calibration":
            for metric in maxima:
                maxima[metric].append(float(comp[metric]))
        if row["label"] == "manipulated":
            measured_edits.append({
                "path": row["path"],
                **{metric: float(comp[metric]) for metric in maxima},
            })

    upper = {metric: max(values) if values else 0.0 for metric, values in maxima.items()}
    if upper["block_diff_max"] <= 0:
        upper = None

    test_reuploads = [r for r in reuploads if r["split"] in ("threshold", "test")]
    genuine_flagged = 0
    genuine_compared = 0
    edits_caught = 0
    edits_compared = 0
    if upper:
        for row in test_reuploads + edits:
            source = originals.get(row["source_id"])
            if not source:
                continue
            upload = cv2.imread(row["path"], cv2.IMREAD_COLOR)
            original = cv2.imread(source["path"], cv2.IMREAD_COLOR)
            if upload is None or original is None:
                continue
            al = pd.align(il.gray(upload), il.gray(original))
            if not al["ok"]:
                continue
            comp = pd.compare(upload, original, al["H"])
            if not comp["ok"]:
                continue
            outside = any(comp[m] > upper[m] for m in upper)
            if row["label"] == "genuine":
                genuine_compared += 1
                genuine_flagged += int(outside)
            else:
                edits_compared += 1
                edits_caught += int(outside)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(manifest["image_root"]),
        "source_pairs": len(originals),
        "reuploads": len(reuploads),
        "paired_edits": len(edits),
        "gate": {
            "positive_n": len(positive),
            "positive_min": min(positive) if positive else None,
            "negative_n": len(negatives),
            "negative_max": max(negatives) if negatives else None,
            "threshold": threshold,
        },
        "range": upper,
        "range_calibration_samples": len(calibration),
        "heldout_test": {
            "genuine_compared": genuine_compared,
            "genuine_flagged": genuine_flagged,
            "edits_compared": edits_compared,
            "edits_caught": edits_caught,
            "edits_missed": edits_compared - edits_caught,
        },
        "measured_edits": measured_edits[:100],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "calibration_external_image_path1.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if write and upper and threshold is not None:
        gates = paths.config("gates")
        gates["image"]["trusted_threshold"] = threshold
        gates["image"]["calibrated"] = True
        gates["image"]["n_positive"] = len(positive)
        gates["image"]["n_negative"] = len(negatives)
        paths.write_json(paths.config_dir() / "gates.json", gates)
        ranges = paths.config("genuine_ranges")
        for metric, max_value in upper.items():
            key = f"diff_picture.{metric}"
            previous = ranges.get(key, {})
            ranges[key] = {
                "range": [0.0, round(float(max_value), 4)],
                "unit": previous.get("unit", ""),
                "calibrated": True,
                "n": len(maxima[metric]),
                "note": f"Measured from external image data on {len(originals)} matched sources.",
            }
        paths.write_json(paths.config_dir() / "genuine_ranges.json", ranges)
        report["written_to_config"] = True
    else:
        report["written_to_config"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default=r"F:\image_calibration")
    parser.add_argument("--video-root", default=r"F:\video_calibration")
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(Path(args.image_root), Path(args.video_root), Path(args.report_dir))
    image_report = calibrate_path1_images(manifest, Path(args.report_dir), args.write)
    print(json.dumps({
        "manifest_image": manifest["image_stats"],
        "manifest_video": manifest["video_stats"],
        "path1_image": image_report,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

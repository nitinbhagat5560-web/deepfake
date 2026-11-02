"""Report video calibration evidence without writing unproven thresholds."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules._common import imagelib as il
from sourcediff import paths


def frame_embedding(path: Path, timestamp_s: float, active_embedder: str) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = count / fps if fps > 0 else 0
    if timestamp_s >= duration:
        cap.release()
        raise RuntimeError(f"timestamp outside {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot decode frame from {path}")
    return il.embed_array(frame, active_embedder)


def sample_scores(source: Path, upload: Path, interval_s: float = 1.0) -> dict:
    active_embedder = il.embedder_for("video")
    source_cap = cv2.VideoCapture(str(source))
    upload_cap = cv2.VideoCapture(str(upload))
    if not source_cap.isOpened() or not upload_cap.isOpened():
        source_cap.release()
        upload_cap.release()
        raise RuntimeError("cannot open paired video")
    sfps = float(source_cap.get(cv2.CAP_PROP_FPS) or 0)
    ufps = float(upload_cap.get(cv2.CAP_PROP_FPS) or 0)
    scount = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ucount = int(upload_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_cap.release()
    upload_cap.release()
    duration = min(scount / sfps if sfps else 0, ucount / ufps if ufps else 0)
    scores = []
    timestamp = 0.0
    while timestamp < duration:
        a = frame_embedding(source, timestamp, active_embedder)
        b = frame_embedding(upload, timestamp, active_embedder)
        scores.append(float(a @ b))
        timestamp += interval_s
    return {
        "mean": float(np.mean(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "min": float(np.min(scores)) if scores else None,
        "frame_scores": scores,
        "duration_s": float(duration),
        "sample_count": len(scores),
    }


def source_family(path: Path) -> str:
    match = re.match(r"^(id\d+)_", path.stem, re.I)
    return match.group(1).lower() if match else path.parent.name.lower()


def split_family(path: Path) -> str:
    family = source_family(path)
    if family.startswith("id") and family[2:].isdigit():
        return "test" if int(family[2:]) % 5 == 4 else "development"
    digest = sum(ord(ch) for ch in family)
    return "test" if digest % 5 == 4 else "development"


def auc_roc(scores: list[float], labels: list[int]) -> float | None:
    if not scores or not any(labels) or all(labels):
        return None
    positive = [score for score, label in zip(scores, labels) if label == 1]
    negative = [score for score, label in zip(scores, labels) if label == 0]
    comparisons = 0
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            comparisons += 1
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return float(wins / comparisons) if comparisons else None


def choose_threshold(scores: list[float], labels: list[int]) -> float | None:
    if not scores:
        return None
    candidates = sorted(set([0.0, 1.0] + [round(i / 1000.0, 3) for i in range(1001)]))
    best = None
    for threshold in candidates:
        tp = sum(score >= threshold and label == 1 for score, label in zip(scores, labels))
        fn = sum(score < threshold and label == 1 for score, label in zip(scores, labels))
        fp = sum(score >= threshold and label == 0 for score, label in zip(scores, labels))
        tn = sum(score < threshold and label == 0 for score, label in zip(scores, labels))
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        youden = sensitivity + specificity - 1.0
        accuracy = (tp + tn) / len(scores) if scores else 0.0
        candidate = (youden, specificity, accuracy, -threshold, threshold)
        if best is None or candidate[:4] > best[:4]:
            best = candidate
    return best[4] if best else None


def threshold_metrics(scores: list[float], labels: list[int], threshold: float) -> dict:
    tp = sum(score >= threshold and label == 1 for score, label in zip(scores, labels))
    fn = sum(score < threshold and label == 1 for score, label in zip(scores, labels))
    fp = sum(score >= threshold and label == 0 for score, label in zip(scores, labels))
    tn = sum(score < threshold and label == 0 for score, label in zip(scores, labels))
    return {
        "threshold": threshold,
        "n": len(scores),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float((tp + tn) / len(scores)) if scores else None,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "false_negative_rate": float(fn / (tp + fn)) if tp + fn else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", default=r"F:\video_calibration")
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    root = Path(args.video_root)
    original_root = root / "orignal"
    manipulation_root = root / "manipulation"
    originals = sorted(original_root.glob("*.mp4"))
    original_by_name = {
        path.stem: path
        for path in originals
        if re.match(r"^id\d+_\d+$", path.stem, re.I)
    }
    face_swaps = sorted((manipulation_root / "face_swap and face edit").glob("*.mp4"))
    matched = []
    unmatched = []
    for path in face_swaps:
        match = re.match(r"^(id\d+)_(id\d+)_(\d+)$", path.stem, re.I)
        if not match:
            unmatched.append(str(path))
            continue
        source_name = f"{match.group(1)}_{match.group(3)}"
        source = original_by_name.get(source_name)
        if source is None:
            unmatched.append(str(path))
            continue
        try:
            matched.append({
                "source": str(source),
                "manipulated": str(path),
                "source_id": source_name,
                "label": "manipulated",
                "similarity": sample_scores(source, path),
            })
        except Exception as exc:
            unmatched.append(f"{path}: {exc}")

    standalone = []
    for folder in sorted(manipulation_root.iterdir()):
        if not folder.is_dir() or folder.name == "face_swap and face edit":
            continue
        standalone.extend(sorted(folder.glob("*.mp4")))

    calibration_pairs = []
    for item in matched:
        source = Path(item["source"])
        calibration_pairs.append({
            "source": item["source"],
            "upload": item["manipulated"],
            "family": source_family(source),
            "split": split_family(source),
            "label": 1,
            "score": item["similarity"]["mean"],
        })

    negative_pairs = []
    for index, left in enumerate(originals):
        right = originals[(index + 11) % len(originals)]
        if left == right or source_family(left) == source_family(right):
            right = originals[(index + 1) % len(originals)]
        if left == right or source_family(left) == source_family(right):
            continue
        try:
            similarity = sample_scores(left, right)["mean"]
            negative_pairs.append({
                "source": str(left),
                "upload": str(right),
                "family": source_family(left),
                "split": split_family(left),
                "label": 0,
                "score": similarity,
            })
        except Exception as exc:
            unmatched.append(f"negative {left.name}/{right.name}: {exc}")

    calibration_pairs.extend(negative_pairs)
    development = [row for row in calibration_pairs if row["split"] == "development"]
    heldout = [row for row in calibration_pairs if row["split"] == "test"]
    development_scores = [row["score"] for row in development]
    development_labels = [row["label"] for row in development]
    heldout_scores = [row["score"] for row in heldout]
    heldout_labels = [row["label"] for row in heldout]
    candidate_threshold = choose_threshold(development_scores, development_labels)
    heldout_metrics = (
        threshold_metrics(heldout_scores, heldout_labels, candidate_threshold)
        if candidate_threshold is not None else None
    )
    threshold_candidate_applied = False

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(root),
        "genuine_original_n": len(originals),
        "face_swap_n": len(face_swaps),
        "face_swaps_source_matched_n": len(matched),
        "face_swaps_unmatched_n": len(unmatched),
        "standalone_manipulation_n": len(standalone),
        "standalone_categories": {
            folder.name: len(list(folder.glob("*.mp4")))
            for folder in sorted(manipulation_root.iterdir())
            if folder.is_dir() and folder.name != "face_swap and face edit"
        },
        "matched_pairs": matched,
        "trial": {
            "method": "aligned-1-second-full-frame-tiny32-gray-cosine",
            "embedder": il.embedder_for("video"),
            "positive_pairs": len(matched),
            "negative_pairs": len(negative_pairs),
            "development_pairs": len(development),
            "heldout_pairs": len(heldout),
            "development_auc": auc_roc(development_scores, development_labels),
            "heldout_auc": auc_roc(heldout_scores, heldout_labels),
            "candidate_threshold": candidate_threshold,
            "candidate_threshold_selected_on_development": True,
            "candidate_threshold_written": threshold_candidate_applied,
            "heldout_metrics": heldout_metrics,
            "pairs": calibration_pairs,
        },
        "unmatched": unmatched[:100],
        "threshold_written": False,
        "limitations": [
            "No genuine reencoded or reuploaded variants were identified for these source videos.",
            "Only exact original-to-face-swap pairs have defensible Path 1 source matching.",
            "Standalone background, style-filter, object-edit, and lipsync clips cannot be source-paired safely.",
            "Video ranges for picture, transcript, and lip-sync remain uncalibrated until source-time aligned data is available.",
            "Cross-source originals are negative controls, not a replacement for genuine reuploads and compression variants.",
            "The threshold trial is deliberately not written to gates.json until genuine-negative and independent held-out evidence is stronger.",
        ],
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "calibration_external_video_path1.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "genuine_original_n": report["genuine_original_n"],
        "face_swap_n": report["face_swap_n"],
        "matched_pairs": report["face_swaps_source_matched_n"],
        "candidate_threshold": report["trial"]["candidate_threshold"],
        "development_auc": report["trial"]["development_auc"],
        "heldout_auc": report["trial"]["heldout_auc"],
        "threshold_written": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

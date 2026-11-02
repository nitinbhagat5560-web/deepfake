"""Calibrate the final Path 2 GenD PE-L image model on bundled datasets."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(r"D:\deepfake\path 2\GenD-main")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
SOURCEDIFF_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCEDIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCEDIFF_ROOT))

import cv2
import numpy as np

from app.run import DeepfakeDetector, MediaProcessor, DEVICE


def collect_samples() -> list[dict]:
    samples = []
    roots = [
        PROJECT / "datasets" / "FF",
        PROJECT / "datasets" / "CDFv2",
    ]
    for root in roots:
        for file in root.rglob("*.png"):
            name = str(file.relative_to(root)).lower()
            if "/real/" in "/" + name.replace("\\", "/") or "celeb-real" in name or "youtube-real" in name:
                label = 0
            elif "/df/" in "/" + name.replace("\\", "/") or "/f2f/" in "/" + name.replace("\\", "/") or "/fs/" in "/" + name.replace("\\", "/") or "/nt/" in "/" + name.replace("\\", "/") or "synthesis" in name:
                label = 1
            else:
                continue
            samples.append({"path": str(file), "label": label, "dataset": root.name})
    return samples


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    if not positives or not negatives:
        return None
    wins = ties = 0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1
            elif positive == negative:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def best_threshold(labels: list[int], scores: list[float]) -> tuple[float, float]:
    candidates = sorted(set(scores))
    best = (float(candidates[0]), -1.0)
    for threshold in candidates:
        tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
        tn = sum(y == 0 and s < threshold for y, s in zip(labels, scores))
        sensitivity = tp / sum(y == 1 for y in labels)
        specificity = tn / sum(y == 0 for y in labels)
        accuracy = (tp + tn) / len(labels)
        youden = sensitivity + specificity - 1.0
        score = accuracy + 0.1 * youden
        if score > best[1]:
            best = (float(threshold), score)
    threshold = best[0]
    tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
    tn = sum(y == 0 and s < threshold for y, s in zip(labels, scores))
    accuracy = (tp + tn) / len(labels)
    return threshold, accuracy


def main() -> None:
    samples = collect_samples()
    if len(samples) < 10:
        raise RuntimeError(f"Insufficient calibration samples: {len(samples)}")

    detector = DeepfakeDetector()
    processor = MediaProcessor(detector)
    model, preproc, dtype = detector.load_model(
        "Hugging Face", "yermandy/GenD_PE_L"
    )
    detector_face = detector.load_detector(0.5)

    rows = []
    try:
        for sample in samples:
            frame = cv2.imread(sample["path"])
            if frame is None:
                continue
            faces = detector.infer_faces(frame, detector_face, model, preproc, dtype, 1.3, None, 1)
            avg_fake = float(np.mean([score for _, score in faces])) if faces else None
            avg_ai = detector.infer_whole_frame(frame, model, preproc, dtype)
            rows.append({
                **sample,
                "face_score": avg_fake,
                "whole_frame_score": avg_ai,
                "num_faces": len(faces),
            })
    finally:
        detector.unload_models()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "yermandy/GenD_PE_L",
        "device": DEVICE,
        "samples": len(rows),
        "real_samples": sum(r["label"] == 0 for r in rows),
        "fake_samples": sum(r["label"] == 1 for r in rows),
        "method": "Development-set threshold selected by accuracy/Youden index on bundled FF and CDFv2 image samples. No held-out validation claim.",
    }

    for field in ("face_score", "whole_frame_score"):
        valid = [r for r in rows if r[field] is not None]
        labels = [r["label"] for r in valid]
        scores = [float(r[field]) for r in valid]
        threshold, accuracy = best_threshold(labels, scores)
        report[field] = {
            "threshold_fake_if_greater_equal": threshold,
            "development_accuracy": accuracy,
            "auc": auc(labels, scores),
            "n": len(valid),
        }

    report_path = Path(__file__).resolve().parents[1] / "reports" / "path2_calibration.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

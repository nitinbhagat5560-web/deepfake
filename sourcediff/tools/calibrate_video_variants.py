"""Calibrate the Path 1 video copy matcher against genuine variant families."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def score_pair(source: Path, variant: Path, interval_s: float = 1.0) -> dict:
    def metadata(path: Path) -> tuple[float, float]:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            return count / fps if fps > 0 else 0.0, fps
        finally:
            cap.release()

    source_duration, source_fps = metadata(source)
    variant_duration, variant_fps = metadata(variant)
    duration = min(source_duration, variant_duration)
    scores = []
    timestamp = 0.0
    while timestamp < duration:
        def frame(path: Path, fps: float, t: float) -> np.ndarray | None:
            cap = cv2.VideoCapture(str(path))
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    return None
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
                small -= small.mean()
                norm = float(np.linalg.norm(small))
                return small / norm if norm > 1e-6 else small
            finally:
                cap.release()

        a = frame(source, source_fps, timestamp)
        b = frame(variant, variant_fps, timestamp)
        if a is not None and b is not None:
            scores.append(float(a @ b))
        timestamp += interval_s

    return {
        "mean": float(np.mean(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "min": float(np.min(scores)) if scores else None,
        "max": float(np.max(scores)) if scores else None,
        "frame_scores": scores,
        "frame_count": len(scores),
        "duration_s": duration,
    }


def auc(scores: list[float], labels: list[int]) -> float | None:
    positive = [score for score, label in zip(scores, labels) if label == 1]
    negative = [score for score, label in zip(scores, labels) if label == 0]
    if not positive or not negative:
        return None
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def threshold_metrics(scores: list[float], labels: list[int], threshold: float) -> dict:
    tp = sum(score >= threshold and label == 1 for score, label in zip(scores, labels))
    fn = sum(score < threshold and label == 1 for score, label in zip(scores, labels))
    fp = sum(score >= threshold and label == 0 for score, label in zip(scores, labels))
    tn = sum(score < threshold and label == 0 for score, label in zip(scores, labels))
    return {
        "threshold": threshold,
        "n": len(scores),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(scores),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "false_positive_rate": fp / (tn + fp) if tn + fp else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
    }


def choose_threshold(scores: list[float], labels: list[int]) -> float | None:
    if not scores:
        return None
    best = None
    candidates = sorted(set([0.0, 1.0] + [round(i / 1000.0, 3) for i in range(1001)]))
    for threshold in candidates:
        metrics = threshold_metrics(scores, labels, threshold)
        sensitivity = metrics["sensitivity"] or 0.0
        specificity = metrics["specificity"] or 0.0
        accuracy = metrics["accuracy"] or 0.0
        candidate = (sensitivity + specificity - 1.0, specificity, accuracy, -threshold, threshold)
        if best is None or candidate[:4] > best[:4]:
            best = candidate
    return best[4] if best else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"C:\Users\Asus\Downloads\video_2_calib")
    parser.add_argument("--external-negative-root", default=r"F:\video_calibration\orignal")
    parser.add_argument("--output", default=str(ROOT / "reports" / "calibration_video_variants_path1.json"))
    parser.add_argument("--write", action="store_true", help="Write threshold only when held-out quality gates pass")
    args = parser.parse_args()

    root = Path(args.data_root)
    originals = root / "genuine_org"
    variant_root = root / "variants_output"
    manifest = json.loads((variant_root / "manifest.json").read_text(encoding="utf-8"))

    positive_pairs = []
    errors = []
    category_scores = {}
    for row in manifest:
        if row.get("label") != "genuine_positive_match":
            continue
        source = originals / row["source_file"]
        variant = variant_root / row["variant_file"]
        family = row["family_id"]
        split = "test" if family in {"fam006", "fam007"} else "development"
        try:
            similarity = score_pair(source, variant)
            score = similarity["mean"]
            if score is None:
                raise RuntimeError("no aligned frame scores")
            positive_pairs.append({
                "source": str(source),
                "variant": str(variant),
                "family_id": family,
                "split": split,
                "category": row["category"],
                "recipe": row["recipe"],
                "label": 1,
                "score": score,
                "similarity": similarity,
            })
            category_scores.setdefault(row["category"], []).append(score)
        except Exception as exc:
            errors.append({"file": str(variant), "error": str(exc)})

    negative_pairs = []
    source_rows = {}
    for row in manifest:
        source_name = row["source_file"]
        source_rows.setdefault(source_name, originals / source_name)
    source_paths = list(source_rows.values())
    for left_index, left in enumerate(source_paths):
        for right in source_paths[left_index + 1:]:
            try:
                similarity = score_pair(left, right)
                score = similarity["mean"]
                if score is None:
                    continue
                family = left.stem if left.stem else left.name
                split = "test" if family in {"id0_0000", "id0_0001"} else "development"
                negative_pairs.append({
                    "source": str(left),
                    "variant": str(right),
                    "family_id": left.stem,
                    "split": split,
                    "category": "independent_source_cross_comparison",
                    "recipe": "different-source-original",
                    "label": 0,
                    "score": score,
                    "similarity": similarity,
                })
            except Exception as exc:
                errors.append({"file": f"{left}|{right}", "error": str(exc)})

    external_negative_root = Path(args.external_negative_root)
    seen_source_hashes = {}
    if external_negative_root.is_dir():
        for source in source_paths:
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            seen_source_hashes[str(source)] = source_hash
        for external in sorted(external_negative_root.glob("*.mp4")):
            if external.name.startswith("id0_") or external.name.startswith("id1_"):
                continue
            external_hash = hashlib.sha256(external.read_bytes()).hexdigest()
            duplicate = external_hash in seen_source_hashes.values()
            if duplicate:
                continue
            for source in source_paths:
                try:
                    source_hash = seen_source_hashes[str(source)]
                    if external_hash == source_hash:
                        continue
                    similarity = score_pair(source, external)
                    score = similarity["mean"]
                    if score is None:
                        continue
                    family = external.stem
                    split = "test" if family.startswith(("id3_", "id4_")) else "development"
                    negative_pairs.append({
                        "source": str(source),
                        "variant": str(external),
                        "family_id": family,
                        "split": split,
                        "category": "independent_external_original",
                        "recipe": "external-source-original",
                        "label": 0,
                        "score": score,
                        "similarity": similarity,
                    })
                except Exception as exc:
                    errors.append({"file": f"{source}|{external}", "error": str(exc)})

    all_pairs = positive_pairs + negative_pairs
    development = [row for row in all_pairs if row["split"] == "development"]
    heldout = [row for row in all_pairs if row["split"] == "test"]
    development_scores = [row["score"] for row in development]
    development_labels = [row["label"] for row in development]
    heldout_scores = [row["score"] for row in heldout]
    heldout_labels = [row["label"] for row in heldout]
    threshold = choose_threshold(development_scores, development_labels)
    heldout_metrics = threshold_metrics(heldout_scores, heldout_labels, threshold) if threshold is not None else None
    development_auc_value = auc(development_scores, development_labels)
    heldout_auc_value = auc(heldout_scores, heldout_labels)

    activation_passed = bool(
        threshold is not None
        and development_auc_value is not None
        and heldout_auc_value is not None
        and development_auc_value >= 0.95
        and heldout_auc_value >= 0.95
        and heldout_metrics is not None
        and heldout_metrics["specificity"] >= 0.95
        and heldout_metrics["false_positive_rate"] == 0.0
        and heldout_metrics["false_negative_rate"] == 0.0
    )
    threshold_written = False
    if args.write and activation_passed:
        import sys
        sys.path.insert(0, str(ROOT))
        from sourcediff import paths
        gates_path = paths.config_dir() / "gates.json"
        gates = paths.load_json(gates_path)
        gates["video"]["trusted_threshold"] = threshold
        gates["video"]["calibrated"] = True
        gates["video"]["calibrated_embedder"] = "tiny32_gray_v1"
        gates["video"]["metric"] = "cosine"
        gates["video"]["n_positive"] = len(positive_pairs)
        gates["video"]["n_negative"] = len(negative_pairs)
        paths.write_json(gates_path, gates)
        threshold_written = True

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(root),
        "method": "1-second full-frame tiny32-gray cosine",
        "positive_variant_n": len(positive_pairs),
        "negative_pair_n": len(negative_pairs),
        "error_n": len(errors),
        "independent_negative_template_present": (variant_root / "independent_negatives_TEMPLATE.csv").is_file(),
        "independent_recordings_generated": False,
        "development_pair_n": len(development),
        "heldout_pair_n": len(heldout),
        "development_auc": development_auc_value,
        "heldout_auc": heldout_auc_value,
        "candidate_threshold": threshold,
        "activation_passed": activation_passed,
        "threshold_written": threshold_written,
        "heldout_metrics": heldout_metrics,
        "category_positive_statistics": {
            category: {
                "n": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for category, values in category_scores.items()
        },
        "errors": errors[:100],
        "limitations": [
            "Independent future recordings were not present.",
            "Cross-source original comparisons are provisional negatives, not an independent recording set.",
            "Held-out families fam006 and fam007 were excluded only from threshold selection.",
            "No threshold was written to gates.json.",
        ],
        "pairs": all_pairs,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "positive_variant_n": report["positive_variant_n"],
        "negative_pair_n": report["negative_pair_n"],
        "development_auc": report["development_auc"],
        "heldout_auc": report["heldout_auc"],
        "candidate_threshold": report["candidate_threshold"],
        "activation_passed": report["activation_passed"],
        "threshold_written": report["threshold_written"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

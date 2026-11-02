import argparse
import csv
import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from app.run import DeepfakeDetector, MediaProcessor


HF_MODELS = (
    "yermandy/GenD_PE_L",
    "yermandy/GenD_CLIP_L_14",
    "yermandy/GenD_DINOv3_L",
)


def auc(scores, labels):
    pairs = sorted(zip(scores, labels), reverse=True)
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average_rank = ((i + 1) + (j + 1)) / 2.0
        rank_sum += sum(pairs[k][1] for k in range(i, j + 1)) * average_rank
        i = j + 1
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def metrics_at(scores, labels, threshold):
    predicted = np.array(scores) >= threshold
    truth = np.array(labels) == 1
    tp = int(np.sum(predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    return {
        "threshold": float(threshold),
        "n": len(labels),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float((tp + tn) / len(labels)),
        "sensitivity_recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "false_negative_rate": float(fn / (tp + fn)) if tp + fn else None,
    }


def choose_threshold(development_scores, development_labels):
    best = None
    for threshold in np.linspace(0.0, 1.0, 1001):
        result = metrics_at(development_scores, development_labels, float(threshold))
        specificity = result["specificity"] or 0.0
        sensitivity = result["sensitivity_recall"] or 0.0
        youden = sensitivity + specificity - 1.0
        candidate = (youden, specificity, sensitivity, float(threshold))
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return {
        "youden_threshold": best[3],
        "conservative_threshold": conservative_threshold(development_scores, development_labels),
    }


def conservative_threshold(scores, labels):
    for threshold in np.linspace(0.0, 1.0, 1001):
        result = metrics_at(scores, labels, float(threshold))
        if (result["specificity"] or 0.0) >= 0.95 and (result["sensitivity_recall"] or 0.0) >= 0.0:
            return float(threshold)
    return 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=r"D:\deepfake\sourcediff\reports\external_calibration_manifest.json")
    parser.add_argument("--output", default=r"D:\deepfake\sourcediff\reports\calibration_external_image_path2.json")
    parser.add_argument("--models", nargs="+", default=list(HF_MODELS) + ["Smaug"])
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    image_rows = [
        row for row in manifest["rows"]
        if row["media_type"] == "image" and row["valid"]
    ]
    report = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "image_n": len(image_rows),
        "genuine_n": sum(row["label"] == "genuine" for row in image_rows),
        "manipulated_n": sum(row["label"] == "manipulated" for row in image_rows),
        "models": {},
        "note": "Experimental: thresholds must remain uncommitted until independent-held-out review.",
    }

    for selected in args.models:
        try:
            detector = DeepfakeDetector()
            processor = MediaProcessor(detector)
            smaug = selected == "Smaug"
            with tempfile.TemporaryDirectory(prefix="path2_cal_") as tmp:
                out_dir = Path(tmp)
                if smaug:
                    detector.load_smaug()
                    model = preproc = face_detector = None
                    dtype = torch.float32
                    mode = "Smaug"
                else:
                    model, preproc, dtype = detector.load_model("Hugging Face", selected)
                    face_detector = detector.load_detector(0.5)
                    mode = "Off"

                rows_out = []
                for row in image_rows:
                    try:
                        _, metrics = processor.process_image(
                            row["path"], face_detector, model, preproc, dtype,
                            1.3, None, out_dir, 1, mode
                        )
                        score = metrics.get("avg_p_ai") if smaug else metrics.get("avg_p_fake")
                        if score is not None:
                            rows_out.append({
                                "path": row["path"],
                                "source_id": row["source_id"],
                                "split": row["split"],
                                "label": row["label"],
                                "score": float(score),
                            })
                    except Exception as exc:
                        rows_out.append({
                            "path": row["path"],
                            "source_id": row["source_id"],
                            "split": row["split"],
                            "label": row["label"],
                            "error": str(exc),
                        })

                scored = [r for r in rows_out if "score" in r]
                development = [r for r in scored if r["split"] in ("calibration", "threshold")]
                heldout = [r for r in scored if r["split"] == "test"]

                def evaluate(items):
                    if not items:
                        return []
                    return items

                dev_scores = [r["score"] for r in development]
                dev_labels = [1 if r["label"] == "manipulated" else 0 for r in development]
                test_scores = [r["score"] for r in heldout]
                test_labels = [1 if r["label"] == "manipulated" else 0 for r in heldout]

                threshold_info = choose_threshold(dev_scores, dev_labels) if dev_scores else {
                    "youden_threshold": None,
                    "conservative_threshold": None,
                }
                thresholds = {}
                for name, threshold in threshold_info.items():
                    thresholds[name] = metrics_at(test_scores, test_labels, threshold) if threshold is not None else None

                report["models"][selected] = {
                    "status": "ok",
                    "scored_n": len(scored),
                    "error_n": len(rows_out) - len(scored),
                    "development_n": len(development),
                    "heldout_n": len(heldout),
                    "development_auc": auc(dev_scores, dev_labels) if dev_scores else None,
                    "heldout_auc": auc(test_scores, test_labels) if test_scores else None,
                    "threshold_info": threshold_info,
                    "heldout_threshold_metrics": thresholds,
                    "errors": [r["error"] for r in rows_out if "error" in r][:20],
                    "scores": rows_out,
                }
            detector.unload_models()
        except Exception as exc:
            report["models"][selected] = {"status": "error", "error": str(exc)}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "status", "path", "split", "label", "score", "error"])
        for model, result in report["models"].items():
            if result.get("status") != "ok":
                writer.writerow([model, "error", "", "", "", "", result.get("error", "")])
                continue
            for row in result["scores"]:
                writer.writerow([
                    model, "ok", row["path"], row["split"], row["label"],
                    row.get("score", ""), row.get("error", ""),
                ])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import gc
import json
import hashlib
import sys
import tempfile
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from app.run import DeepfakeDetector, MediaProcessor


MODELS = (
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
        average_rank = ((i + 1 + j + 1) / 2.0)
        rank_sum += sum(pairs[k][1] for k in range(i, j + 1)) * average_rank
        i = j + 1
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def split_for(path: Path) -> str:
    name = path.stem.lower()
    if name.startswith("id"):
        source_family = name.split("_", 1)[0]
    else:
        source_family = path.parent.name.lower()
    digest = hashlib.sha256(source_family.encode("utf-8")).hexdigest()
    return "test" if int(digest[:8], 16) % 5 == 4 else "development"


def metrics(scores, labels, threshold):
    predicted = [score >= threshold for score in scores]
    truth = [label == 1 for label in labels]
    tp = sum(p and t for p, t in zip(predicted, truth))
    tn = sum(not p and not t for p, t in zip(predicted, truth))
    fp = sum(p and not t for p, t in zip(predicted, truth))
    fn = sum(not p and t for p, t in zip(predicted, truth))
    return {
        "threshold": float(threshold),
        "n": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float((tp + tn) / len(labels)) if labels else None,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "false_negative_rate": float(fn / (tp + fn)) if tp + fn else None,
    }


def choose_threshold(scores, labels):
    best = None
    for threshold in [i / 1000.0 for i in range(1001)]:
        result = metrics(scores, labels, threshold)
        sensitivity = result["sensitivity"] or 0.0
        specificity = result["specificity"] or 0.0
        score = sensitivity + specificity - 1.0
        candidate = (score, specificity, sensitivity, threshold)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3] if best else 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", default=r"F:\video_calibration")
    parser.add_argument("--output", default=r"D:\deepfake\sourcediff\reports\calibration_external_video_path2.json")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=list(MODELS) + ["Smaug"])
    args = parser.parse_args()

    root = Path(args.video_root)
    videos = []
    for path in sorted((root / "orignal").glob("*.mp4")):
        videos.append({"path": path, "label": "genuine", "source_id": path.stem})
    for path in sorted((root / "manipulation").rglob("*.mp4")):
        videos.append({"path": path, "label": "manipulated", "source_id": path.stem})

    report = {
        "status": "ok",
        "video_n": len(videos),
        "genuine_n": sum(v["label"] == "genuine" for v in videos),
        "manipulated_n": sum(v["label"] == "manipulated" for v in videos),
        "stride": args.stride,
        "max_frames": args.max_frames,
        "models": {},
    }

    for selected in args.models:
        try:
            detector = DeepfakeDetector()
            processor = MediaProcessor(detector)
            smaug = selected == "Smaug"
            if smaug:
                detector.load_smaug()
                model = preproc = face_detector = None
                dtype = torch.float32
                mode = "Smaug"
            else:
                model, preproc, dtype = detector.load_model("Hugging Face", selected)
                face_detector = detector.load_detector(0.5)
                mode = "Off"

            scored = []
            with tempfile.TemporaryDirectory(prefix="path2_video_cal_") as tmp:
                out_dir = Path(tmp)
                for index, video in enumerate(videos):
                    try:
                        _, metrics_out = processor.process_video(
                            str(video["path"]), face_detector, model, preproc, dtype,
                            1.3, None, out_dir, args.stride, args.max_frames, 1, mode
                        )
                        score = metrics_out.get("avg_p_ai") if smaug else metrics_out.get("avg_p_fake")
                        if score is not None:
                            scored.append({
                                "path": str(video["path"]),
                                "label": video["label"],
                                "source_id": video["source_id"],
                                "split": split_for(video["path"]),
                                "score": float(score),
                                "num_frames": metrics_out.get("num_frames"),
                                "num_faces": metrics_out.get("num_faces"),
                            })
                        else:
                            scored.append({
                                "path": str(video["path"]),
                                "label": video["label"],
                                "source_id": video["source_id"],
                                "split": split_for(video["path"]),
                                "error": "no face or frame score produced",
                            })
                    except Exception as exc:
                        scored.append({
                            "path": str(video["path"]),
                            "label": video["label"],
                            "source_id": video["source_id"],
                            "split": split_for(video["path"]),
                            "error": str(exc),
                        })

            detector.unload_models()
            valid = [row for row in scored if "score" in row]
            development = [row for row in valid if row["split"] == "development"]
            heldout = [row for row in valid if row["split"] == "test"]
            dev_scores = [row["score"] for row in development]
            dev_labels = [1 if row["label"] == "manipulated" else 0 for row in development]
            test_scores = [row["score"] for row in heldout]
            test_labels = [1 if row["label"] == "manipulated" else 0 for row in heldout]
            threshold = choose_threshold(dev_scores, dev_labels) if dev_scores else None
            report["models"][selected] = {
                "status": "ok",
                "scored_n": len(valid),
                "error_n": len(scored) - len(valid),
                "development_n": len(development),
                "heldout_n": len(heldout),
                "development_auc": auc(dev_scores, dev_labels) if dev_scores else None,
                "heldout_auc": auc(test_scores, test_labels) if test_scores else None,
                "youden_threshold": threshold,
                "heldout_metrics": metrics(test_scores, test_labels, threshold) if threshold is not None else None,
                "errors": [row["error"] for row in scored if "error" in row][:30],
                "scores": scored,
            }
        except Exception as exc:
            report["models"][selected] = {"status": "error", "error": str(exc)}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.run import DeepfakeDetector, MediaProcessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--face-thresh", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--max-faces", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "path2_result.json"

    detector = DeepfakeDetector()
    processor = MediaProcessor(detector)
    try:
        if args.mode == "Smaug":
            detector.load_smaug()
            model = preproc = detector_face = None
            dtype = __import__("torch").float32
        else:
            model, preproc, dtype = detector.load_model("Hugging Face", args.model)
            detector_face = detector.load_detector(args.face_thresh)

        if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            from app.run import is_image
            if not is_image(str(input_path)):
                raise RuntimeError("Unsupported image input")
            artifact, metrics = processor.process_image(
                str(input_path), detector_face, model, preproc, dtype, 1.3, None,
                output_dir, args.max_faces, args.mode,
            )
        else:
            from app.run import is_video
            if not is_video(str(input_path)):
                raise RuntimeError("Unsupported video input")
            artifact, metrics = processor.process_video(
                str(input_path), detector_face, model, preproc, dtype, 1.3, None,
                output_dir, args.stride, args.max_frames, args.max_faces, args.mode,
            )

        result = {
            "status": "ok",
            "model": "Smaug AI-image detector" if args.mode == "Smaug" else args.model,
            "mode": args.mode,
            "artifact": Path(artifact).name,
            "metrics": metrics,
        }
        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": "ok", "result_file": str(result_file)}))
        return 0
    except Exception as exc:
        result = {"status": "error", "error": str(exc), "model": args.model, "mode": args.mode, "metrics": {}}
        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), file=sys.stderr)
        return 1
    finally:
        detector.unload_models()


if __name__ == "__main__":
    raise SystemExit(main())

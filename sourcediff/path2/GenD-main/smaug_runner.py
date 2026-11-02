import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_path = snapshot_download("abacusai/Smaug-AI-Image-Detector")
sys.path.insert(0, model_path)
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from smaug_image_detect import SmaugImageDetect  # noqa: E402

requested_device = os.environ.get("SMAUG_DEVICE", "auto").lower()
if requested_device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = requested_device

if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

model = SmaugImageDetect.from_pretrained(
    "abacusai/Smaug-AI-Image-Detector",
    device=device,
)

print(json.dumps({"status": "ready", "model_path": model_path, "device": device}), flush=True)

for line in sys.stdin:
    try:
        request = json.loads(line)
        image_path = Path(request["image"])
        with Image.open(image_path) as image:
            probability = float(model(image.convert("RGB"), crops=3))
        response = {"probability": max(0.0, min(1.0, probability))}
    except Exception as exc:
        response = {"error": repr(exc)}
    print(json.dumps(response), flush=True)

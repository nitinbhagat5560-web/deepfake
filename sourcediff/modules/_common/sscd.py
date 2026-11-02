"""Local inference adapter for the official SSCD TorchScript checkpoint."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from sourcediff import paths

EMBEDDER = "sscd_disc_mixup"
DIM = 512
MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
INPUT_EDGE = 288
_model = None
_device = None
_loaded_path: Path | None = None


def model_path() -> Path:
    return Path(os.environ.get(
        "SOURCEDIFF_SSCD_MODEL",
        paths.library_dir() / "models" / "sscd_disc_mixup.torchscript.pt",
    ))


def _load_model():
    global _model, _device, _loaded_path
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"SSCD checkpoint is missing at {path}; run "
            "'python tools/download_sscd.py' or set SOURCEDIFF_SSCD_MODEL"
        )
    if _model is not None and _loaded_path == path:
        return _model, _device

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SSCD requires PyTorch. Install the optional dependency with "
            "'python -m pip install -r requirements-sscd.txt'."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    _model, _device, _loaded_path = model, device, path
    return model, device


def embed_array(img: np.ndarray) -> np.ndarray:
    """Return one L2-normalized 512-dimensional descriptor for a BGR or gray image."""
    model, device = _load_model()
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"SSCD expects a grayscale or BGR image, got shape {img.shape}")

    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("SSCD cannot embed an empty image")
    scale = INPUT_EDGE / float(min(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])

    import torch
    with torch.inference_mode():
        result = model(torch.from_numpy(tensor).to(device))
    vector = result[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
    if vector.size != DIM or not np.isfinite(vector).all():
        raise RuntimeError(f"SSCD returned an invalid descriptor with shape {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("SSCD returned a zero-length descriptor")
    return vector / norm

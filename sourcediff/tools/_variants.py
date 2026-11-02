"""Helpers that make sample pictures, 'genuine re-uploads' and 'edited copies', for calibration and demos.

A genuine re-upload is what a chat app or social site does to a picture: shrink it, then recompress it.
An edit is what someone does on purpose: paste something in, blur or remove a region, copy one part over
another, write text on it. The edited copy is then re-uploaded the same way, because that is how it reaches
the checker in real life.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


# ---------------------------------------------------------------- sample pictures
def sample(name: str, max_side: int = 800) -> np.ndarray:
    """A picture bundled with scikit-image, as BGR uint8, shrunk to at most max_side pixels."""
    import skimage.data as sd
    a = getattr(sd, name)()
    if a.dtype != np.uint8:
        a = (a.astype(np.float32) / max(1.0, float(a.max())) * 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR)
    else:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    s = max_side / float(max(a.shape[:2]))
    return scale(a, s) if s < 1 else a


# ---------------------------------------------------------------- what a platform does
def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def scale(img: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return img.copy()
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def reupload(img: np.ndarray, q: int, s: float) -> np.ndarray:
    """Shrink, then recompress."""
    return jpeg(scale(img, s), q)


def crop(img: np.ndarray, frac: float) -> np.ndarray:
    """Centre crop keeping `frac` of each side."""
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    return img[y:y + ch, x:x + cw].copy()


# ---------------------------------------------------------------- edits
def _fit(img, x, y, size):
    h, w = img.shape[:2]
    size = min(size, h - 2, w - 2)
    return min(max(0, x), w - size), min(max(0, y), h - size), size


def paste_patch(img, patch, x, y, size):
    x, y, size = _fit(img, x, y, size)
    out = img.copy()
    out[y:y + size, x:x + size] = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return out


def blur_region(img, x, y, size, sigma=12):
    x, y, size = _fit(img, x, y, size)
    out = img.copy()
    out[y:y + size, x:x + size] = cv2.GaussianBlur(out[y:y + size, x:x + size], (0, 0), sigma)
    return out


def inpaint_region(img, x, y, size):
    """Remove an object: fill a square from its surroundings."""
    x, y, size = _fit(img, x, y, size)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y:y + size, x:x + size] = 255
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def copy_move(img, x, y, size, x2, y2):
    """Copy one region over another place in the same picture."""
    x, y, size = _fit(img, x, y, size)
    x2, y2, _ = _fit(img, x2, y2, size)
    out = img.copy()
    out[y:y + size, x:x + size] = img[y2:y2 + size, x2:x2 + size]
    return out


def put_text(img, x, y, text="RESIGNED"):
    out = img.copy()
    h, w = out.shape[:2]
    x, y = min(max(0, x), max(0, w - 260)), min(max(40, y + 60), h - 10)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    return out


def recolour_region(img, x, y, size, da=45, db=-35):
    """Shift Lab a*/b* by a fixed offset so brightness stays put but colour moves (hair-dye style).

    A pure hue rotation does almost nothing on near-grey hair/skin; dyeing injects chroma, so we do too.
    """
    x, y, size = _fit(img, x, y, size)
    out = img.copy()
    roi = out[y:y + size, x:x + size]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] = np.clip(lab[..., 1] + da, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + db, 0, 255)
    out[y:y + size, x:x + size] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def random_edits(img, donor, rng):
    """[(kind, edited_image)] covering the edit types, two random placements each."""
    def spot(size):
        h, w = img.shape[:2]
        size = min(size, h - 2, w - 2)
        return int(rng.integers(0, w - size + 1)), int(rng.integers(0, h - size + 1)), size

    out = []
    for size in (40, 80, 130, 190):
        for _ in range(2):
            x, y, sz = spot(size)
            out.append((f"patch{size}", paste_patch(img, donor, x, y, sz)))
    for size in (60, 100, 150):
        for _ in range(2):
            x, y, sz = spot(size)
            out.append((f"blur{size}", blur_region(img, x, y, sz)))
    for size in (60, 100):
        for _ in range(2):
            x, y, sz = spot(size)
            out.append((f"inpaint{size}", inpaint_region(img, x, y, sz)))
    for size in (60, 100):
        for _ in range(2):
            x, y, sz = spot(size)
            x2, y2, _ = spot(sz)
            out.append((f"copymove{size}", copy_move(img, x, y, sz, x2, y2)))
    for size in (60, 100, 150):
        for _ in range(2):
            x, y, sz = spot(size)
            out.append((f"recolour{size}", recolour_region(img, x, y, sz)))
    for _ in range(2):
        x, y, _ = spot(150)
        out.append(("text", put_text(img, x, y)))
    return out

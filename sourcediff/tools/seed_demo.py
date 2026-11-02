"""Fill library/trusted/image with 8 sample photos, build the index, and write the demo uploads to demo/media/.

REHEARSAL ONLY. The photos come from scikit-image's bundled sample data. At the hackathon the trusted
library is real originals (official speeches, press clips) and the test fakes are made by you.

    python tools/seed_demo.py

Demo uploads (each is the picture, shrunk and JPEG-recompressed the way a chat app would):
  genuine_reupload_*      an unchanged copy of a library picture                -> Authentic
  edited_*                a copy with something pasted, written or moved in it  -> Picture edited
  unrelated_retina.jpg    a picture that is not in the library                  -> Can't Verify (no original found)
  known_limit_crop_*      a centre CROP of a library picture. The 32x32 thumbnail stand-in does not survive
                          crops, so the original is not found                   -> Can't Verify
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2

import _variants as V
from modules._common import imagelib as il
from sourcediff import paths

TRUSTED = ["astronaut", "camera", "coins", "chelsea", "coffee", "rocket", "moon", "immunohistochemistry"]


def demo_images() -> dict:
    ast, cof, chel = V.sample("astronaut"), V.sample("coffee"), V.sample("chelsea")
    return {
        "genuine_reupload_astronaut.jpg": V.reupload(ast, 55, 0.75),
        "genuine_reupload_coffee.jpg": V.reupload(cof, 40, 0.6),
        "genuine_reupload_chelsea.jpg": V.reupload(chel, 70, 0.5),
        "edited_astronaut_patch.jpg": V.reupload(V.paste_patch(ast, chel, 190, 300, 130), 60, 0.8),
        "edited_coffee_text.jpg": V.reupload(V.put_text(cof, 200, 150), 60, 0.8),
        "edited_chelsea_copymove.jpg": V.reupload(V.copy_move(chel, 40, 40, 100, 250, 150), 60, 0.8),
        "unrelated_retina.jpg": V.jpeg(V.scale(V.sample("retina"), 0.8), 80),
        "known_limit_crop_astronaut.jpg": V.reupload(V.crop(ast, 0.6), 75, 1.0),
    }


def main():
    paths.ensure_library_layout()
    for name in TRUSTED:
        d = paths.library_entry_dir("trusted", "image", name)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "original.png"), V.sample(name))
        paths.write_json(d / "meta.json", {
            "source_id": name, "title": f"{name} (scikit-image sample photo)", "publisher": "scikit-image sample data",
            "url": "", "license": "rehearsal only; check scikit-image data licences before any public use",
            "file": "original.png", "door": "image",
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    n = il.build_index("trusted")
    print(f"trusted library: {n} images, index built")

    media = ROOT / "demo" / "media"
    media.mkdir(parents=True, exist_ok=True)
    out = demo_images()
    for fn, img in out.items():
        cv2.imwrite(str(media / fn), img)
    print("demo uploads:", ", ".join(sorted(out)))


if __name__ == "__main__":
    main()

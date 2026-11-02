"""Run the demo uploads end to end, show every file a run writes, and check the verdicts.

    python tools/rehearsal.py

Uses its own folder (data_rehearsal/) so it never touches your real history. Builds the library index
and the demo files first if they are missing. Ends with "ALL OK" or exits with an error.

After it finishes, open  data_rehearsal/cases/<case id>/  and read every file. That is the fastest way to
understand the system.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["SOURCEDIFF_DATA"] = str(ROOT / "data_rehearsal")

from sourcediff import history, paths
from sourcediff.ingest import create_case
from sourcediff.runner import run_case

EXPECT = {   # file -> (verdict, why it is expected)
    "genuine_reupload_astronaut.jpg": ("authentic", "unchanged copy, shrunk and recompressed"),
    "genuine_reupload_coffee.jpg": ("authentic", "unchanged copy, shrunk and recompressed"),
    "genuine_reupload_chelsea.jpg": ("authentic", "unchanged copy, shrunk and recompressed"),
    "edited_astronaut_patch.jpg": ("edited_picture", "another picture pasted over a region"),
    "edited_coffee_text.jpg": ("edited_picture", "text written on the picture"),
    "edited_chelsea_copymove.jpg": ("edited_picture", "one region copied over another"),
    "unrelated_retina.jpg": ("cant_verify", "not in the library: no original found"),
    "known_limit_crop_astronaut.jpg": ("authentic", "a crop; geometric alignment verifies it matches the original"),
}


def ensure_ready():
    media = ROOT / "demo" / "media"
    lib = paths.library_door_dir("trusted", "image")
    if not (media.exists() and any(media.glob("*.jpg"))) or not (lib.exists() and any(lib.iterdir())):
        import seed_demo
        seed_demo.main()
    elif not (paths.library_dir() / "index" / "image_trusted.faiss").exists():
        from modules._common import imagelib as il
        il.build_index("trusted")


def main():
    ensure_ready()
    shutil.rmtree(paths.data_dir(), ignore_errors=True)
    bad = []
    for fn, (want, why) in EXPECT.items():
        case_dir = create_case(ROOT / "demo" / "media" / fn, fn)
        out = run_case(case_dir)
        card = out["card"]
        history.add_submission("rehearsal", fn, card["input"]["sha256"], card["input"]["door"], out["verdict"],
                               out["label"], out["case_id"], out["stub_used"], out["reason_code"], card["card_sha256"])
        ok = out["verdict"] == want
        if not ok:
            bad.append(fn)
        print(f"{'ok ' if ok else 'BAD'} {fn:34s} -> {out['label']:22s} ({out['runtime_s']:.1f}s)  # {why}")
        print(f"      {card['reasons'][0] if card['reasons'] else ''}")
        files = sorted(str(p.relative_to(case_dir)) for p in case_dir.rglob("*") if p.is_file())
        print(f"      files: {', '.join(files)}")
    chain = history.verify_chain()
    print(f"\nhistory: {chain['rows']} rows, chain {'verifies' if chain['ok'] else 'BROKEN'}")
    for r in history.list_submissions(limit=3):
        print(f"   {r['submitted_at']}  {r['user_id']:10s} {r['filename']:34s} {r['label']}")
    print(f"\ncase folders: {paths.cases_dir()}")
    if bad or not chain["ok"]:
        print("MISMATCH:", ", ".join(bad) or "history chain")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()

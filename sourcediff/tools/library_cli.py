"""Manage the library on disk.

    python tools/library_cli.py init                       # create the media subfolders
    python tools/library_cli.py add-trusted FILE --id ID --title "..." --publisher "..." --url "..." --license "..."
    python tools/library_cli.py add-folder DIR --publisher "..." --license "..."   # bulk: every media file in DIR
    python tools/library_cli.py build-index                # ALWAYS after adding; the index is derived
    python tools/library_cli.py list
    python tools/library_cli.py migrate                    # move an old flat library into the per-media layout

Layout (one subfolder per media type; the type comes from the file extension):
    library/trusted/{image,video}/<id>/{original.ext, meta.json}             Gate 1 sources
    library/fake/...                                                         reserved for a later Gate 2
    library/index/<door>_{trusted,fake}.{faiss,json}                         derived: rebuild, never share or merge

Only use media you have the right to use (public domain, official sources, your own recordings).
Keep the URL and licence in meta.json; the tool stores whatever you pass.
"""
import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sourcediff import library, paths
from sourcediff.ingest import UnsupportedFile, door_for


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "item"


def add_trusted(file, source_id, title, publisher="", url="", license=""):
    """Same code as the web page (sourcediff/library.py), but strict=False: stores the file as given."""
    src = Path(file)
    door_for(src.name)                              # unsupported extension: raises UnsupportedFile, as before
    try:
        meta = library.add_item(src, src.name, title, publisher, license, url, source_id=source_id, strict=False)
    except library.LibraryError as exc:
        raise SystemExit(str(exc))
    return paths.library_entry_dir("trusted", meta["door"], meta["source_id"])


def add_folder(folder, publisher="", url="", license=""):
    """Add every media file found directly in FOLDER. Id = file name stem, made unique."""
    added, skipped = [], []
    for f in sorted(Path(folder).iterdir()):
        if not f.is_file():
            continue
        try:
            door = door_for(f.name)
        except UnsupportedFile:
            skipped.append(f.name)
            continue
        sid = _slug(f.stem)
        n = 2
        while paths.library_entry_dir("trusted", door, sid).exists():
            sid = f"{_slug(f.stem)}-{n}"
            n += 1
        add_trusted(f, sid, f.stem, publisher, url, license)
        added.append(f"{door}/{sid}")
    return added, skipped


def migrate_layout():
    """Old flat layout library/<kind>/<id>/ -> library/<kind>/<door>/<id>/. Safe to run twice."""
    moved = []
    for kind in paths.LIBRARY_KINDS:
        root = paths.library_dir() / kind
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in paths.DOORS:
                continue
            mf = d / "meta.json"
            if not mf.exists():
                continue
            door = paths.load_json(mf).get("door") or "image"
            dest = paths.library_entry_dir(kind, door, d.name)
            if dest.exists():
                raise SystemExit(f"cannot migrate {d}: {dest} already exists")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dest))
            moved.append(f"{kind}/{d.name} -> {kind}/{door}/{d.name}")
    paths.ensure_library_layout()
    return moved


def build_index():
    paths.ensure_library_layout()
    from modules._common import imagelib as il
    print(f"image/trusted: {il.build_index('trusted')} image(s) indexed")
    from modules._common import videolib as vl
    print(f"video/trusted: {vl.build_index('trusted')} frame(s) indexed")
def list_library():
    for kind in paths.LIBRARY_KINDS:
        for door in paths.DOORS:
            root = paths.library_door_dir(kind, door)
            ids = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
            if kind == "fake" and not ids:
                continue
            print(f"{kind}/{door} ({len(ids)}): {', '.join(ids) or '-'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    a = sub.add_parser("add-trusted")
    a.add_argument("file")
    a.add_argument("--id", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--publisher", default="")
    a.add_argument("--url", default="")
    a.add_argument("--license", default="")
    f = sub.add_parser("add-folder")
    f.add_argument("folder")
    f.add_argument("--publisher", default="")
    f.add_argument("--url", default="")
    f.add_argument("--license", default="")
    sub.add_parser("build-index")
    sub.add_parser("migrate")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "init":
        paths.ensure_library_layout()
        print("library layout ready under", paths.library_dir())
    elif args.cmd == "add-trusted":
        print("added", add_trusted(args.file, args.id, args.title, args.publisher, args.url, args.license))
    elif args.cmd == "add-folder":
        added, skipped = add_folder(args.folder, args.publisher, args.url, args.license)
        print(f"added {len(added)}: {', '.join(added) or '-'}")
        if skipped:
            print(f"skipped (not a supported media type): {', '.join(skipped)}")
        print("now run:  python tools/library_cli.py build-index")
    elif args.cmd == "build-index":
        build_index()
    elif args.cmd == "migrate":
        moved = migrate_layout()
        print("\n".join(moved) if moved else "nothing to migrate")
    else:
        list_library()


if __name__ == "__main__":
    main()

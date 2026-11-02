"""Run one file through the whole pipeline from the command line.

    python tools/run_case.py demo/media/genuine_reupload_astronaut.jpg
    python tools/run_case.py FILE --user alice          # the history row is written for 'alice'
    python tools/run_case.py FILE --no-history

Prints the evidence card and the case folder, so you can open every file the run produced.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sourcediff import history
from sourcediff.ingest import create_case
from sourcediff.runner import run_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--user", default="cli")
    ap.add_argument("--no-history", action="store_true")
    a = ap.parse_args()
    case_dir = create_case(a.file, Path(a.file).name)
    out = run_case(case_dir, log=print)
    card = out["card"]
    if not a.no_history:
        history.add_submission(a.user, card["input"]["filename"], card["input"]["sha256"], card["input"]["door"],
                               out["verdict"], out["label"], out["case_id"], out["stub_used"],
                               out["reason_code"], card["card_sha256"])
    print("\n" + (Path(case_dir) / "card.md").read_text(encoding="utf-8"))
    print("case folder:", case_dir)


if __name__ == "__main__":
    main()

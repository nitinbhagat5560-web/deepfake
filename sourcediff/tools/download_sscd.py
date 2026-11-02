"""Download the official SSCD TorchScript model into the configured library model directory."""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules._common import sscd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=sscd.model_path(),
                        help="checkpoint path (default: library/models/sscd_disc_mixup.torchscript.pt)")
    args = parser.parse_args(argv)
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    try:
        request = urllib.request.Request(
            sscd.MODEL_URL, headers={"User-Agent": "SourceDiff SSCD model downloader"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError("downloaded SSCD checkpoint is empty")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded official SSCD checkpoint to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

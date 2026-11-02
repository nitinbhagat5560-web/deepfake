"""Path 2 model runner used by the SourceDiff Uvicorn application."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


FINAL_MODEL = "yermandy/GenD_PE_L"
FINAL_MODE = "GenD"
HF_MODELS = (FINAL_MODEL,)
MODES = (FINAL_MODE,)


def run_model(case_dir: Path, input_file: Path, model: str = HF_MODELS[0],
              mode: str = "Off", face_thresh: float = 0.5,
              stride: int = 1, max_frames: int = -1,
              max_faces: int = 1, timeout_s: int = 900) -> dict:
    # SourceDiff exposes one production fallback configuration.
    model = FINAL_MODEL
    mode = FINAL_MODE
    project = Path(__file__).resolve().parents[1] / "path2" / "GenD-main"
    if not project.is_dir():
        legacy_project = Path(__file__).resolve().parents[2] / "path 2" / "GenD-main"
        if legacy_project.is_dir():
            project = legacy_project
    python = (
        project / ".venv-smaug" / "Scripts" / "python.exe"
        if model == "Smaug" or mode == "Smaug"
        else project / ".venv" / "Scripts" / "python.exe"
    )
    if not python.is_file():
        raise RuntimeError("Path 2 environment is missing: " + str(python))

    if model not in HF_MODELS and model != "Smaug":
        raise RuntimeError("Unknown Path 2 model: " + model)
    if mode not in MODES:
        raise RuntimeError("Unknown Path 2 analysis mode: " + mode)

    output_dir = case_dir / "path2_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = project / "path2_bridge.py"
    cp = subprocess.run(
        [str(python), str(bridge),
         "--input", str(input_file),
         "--output-dir", str(output_dir),
         "--model", model,
         "--mode", mode,
         "--face-thresh", str(face_thresh),
         "--stride", str(stride),
         "--max-frames", str(max_frames),
         "--max-faces", str(max_faces)],
        cwd=str(project),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )

    result_file = output_dir / "path2_result.json"
    if result_file.is_file():
        result = json.loads(result_file.read_text(encoding="utf-8"))
    else:
        last_error = cp.stderr.strip().splitlines()[-1] if cp.stderr.strip() else "no error output"
        raise RuntimeError(f"Path 2 failed (exit {cp.returncode}): {last_error}")
    if cp.returncode != 0 and result.get("status") != "ok":
        raise RuntimeError(result.get("error", f"Path 2 exited with {cp.returncode}"))
    return result

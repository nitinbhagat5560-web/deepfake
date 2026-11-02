"""The runner: ingest, then run each module as its own process, then the verdict engine.

Path 1 flow:
  ingest
  gate1                 (always: look for the original in library/trusted)
    matched   ->  align, path1      (diff the upload against the original)
    otherwise ->  nothing more to run; the engine answers "Can't Verify"
  verdict -> card

Each module is `python modules/<name>/run.py <case_dir>`, launched with PYTHONPATH=<project>.
registry.json "python" lets a module use its own virtualenv (SyncNet needs an older torch).
The runner never imports a module. A crash, a timeout or a missing result becomes
status="error", which the verdict engine treats as 'did not run', never as 'agrees'.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sourcediff import paths, report, verdict
from sourcediff.contract import ModuleResult
from sourcediff.ingest import ingest


def _write_status(case_dir: Path, name: str, status: str, reason: str) -> None:
    res = ModuleResult(module=name, status=status, reason=reason)
    paths.write_json(case_dir / "results" / f"{name}.json", res.to_dict())


def _launch(case_dir: Path, m: dict, log) -> str:
    name = m["name"]
    script = paths.ROOT / "modules" / name / "run.py"
    py = m.get("python") or sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    try:
        cp = subprocess.run([py, str(script), str(case_dir)], cwd=paths.ROOT, env=env,
                            capture_output=True, text=True, timeout=m.get("timeout_s", 600))
        out = cp.stdout + cp.stderr
        (case_dir / "logs").mkdir(exist_ok=True)
        (case_dir / "logs" / f"{name}.log").write_text(out, encoding="utf-8")
        if not (case_dir / "results" / f"{name}.json").exists():
            last = (cp.stderr.strip().splitlines() or ["no output"])[-1]
            _write_status(case_dir, name, "error", f"exit {cp.returncode}: {last}")
    except subprocess.TimeoutExpired:
        _write_status(case_dir, name, "error", f"timeout after {m.get('timeout_s')} s")
    log(f"  {name:22s} {time.time() - t0:5.2f}s")
    return name


def _gate(case_dir: Path, module: str):
    f = case_dir / "results" / f"{module}.json"
    if not f.exists():
        return None
    try:
        res = ModuleResult.from_dict(paths.load_json(f))
    except Exception:
        return None
    return res.gate if res.status == "ok" else None


def run_case(case_dir, log=lambda s: None) -> dict:
    case_dir = Path(case_dir)
    shutil.rmtree(case_dir / "results", ignore_errors=True)       # a re-run starts clean
    for stale in ("alignment.json", "source_ref.json", "chunk_map.json", "gate1_detail.json",
                  "diff_picture_map.png", "video_timeline.png", "verdict.json", "card.json", "card.md"):
        (case_dir / stale).unlink(missing_ok=True)

    t0 = time.time()
    ing = ingest(case_dir)
    door, flags = ing["door"], ing["flags"]
    reg = paths.config("registry")
    roles = paths.config("verdict")["roles"]
    stages_run = []

    def stage(name):
        stages_run.append(name)
        log(f"stage {name}")
        for m in reg["modules"]:
            if m["stage"] != name or not m.get("enabled", True) or door not in m["doors"]:
                continue
            unmet = [n for n in m["needs"] if not flags.get(n)]
            if unmet:
                _write_status(case_dir, m["name"], "skipped", "not applicable: needs " + ", ".join(unmet))
                continue
            _launch(case_dir, m, log)

    stage("gate1")
    g1 = _gate(case_dir, roles["gate1"][door])
    if g1 and g1.matched and not g1.exact_match:
        stage("align")
        stage("path1")

    v = verdict.decide(case_dir)
    v["stages_run"] = stages_run
    v["runtime_s"] = round(time.time() - t0, 2)
    paths.write_json(case_dir / "verdict.json", v)
    card = report.write_card(case_dir)
    return {"case_dir": str(case_dir), "case_id": case_dir.name, "verdict": v["verdict"],
            "label": v["label"], "reason_code": v["reason_code"], "card": card,
            "stub_used": v["stub_used"], "runtime_s": v["runtime_s"]}

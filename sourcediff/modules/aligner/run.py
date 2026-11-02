"""aligner  [align]  doors: image and video

Writes alignment.json, which the verdict engine reads as the 'alignment' channel:
  ok          one consistent alignment was found
  consistent  false if the timeline is not one straight line (a jump means cut/moved pieces)
  jumps       [{at_s, size_s}] where the offset-and-speed line breaks

VIDEO: robustly fit source_s = speed * upload_s + offset to Gate 1's chunk_map using the median
adjacent frame/fingerprint step. A timeline discontinuity or speed outside 0.9x-1.1x is inconsistent.
"""
import numpy as np

from sourcediff import paths
from sourcediff.contract import ModuleResult
from sourcediff.module_api import in_scenario, run_module, skipped

NAME = "aligner"


def run(ctx):
    sc = ctx.case_dir / "scenario" / "alignment.json"
    if in_scenario(ctx):
        if sc.exists():
            ctx.write_json("alignment.json", paths.load_json(sc))
            return ModuleResult(module=NAME, stub=True, artifacts={"alignment": "alignment.json"})
        return skipped(NAME, "scenario has no alignment result", stub=True)
    if ctx.door != "image":
        if ctx.door != "video":
            return skipped(NAME, f"unsupported alignment door: {ctx.door}")
        chunk_file = ctx.case_dir / "chunk_map.json"
        if not chunk_file.exists():
            return skipped(NAME, "video gate did not produce a frame map")
        points = np.asarray(ctx.read_json("chunk_map.json"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            return skipped(NAME, "video frame map needs at least two finite upload/source timestamp pairs")
        upload_t, source_t = points[:, 0], points[:, 1]
        upload_steps = np.diff(upload_t)
        if np.any(upload_steps <= 0):
            return skipped(NAME, "video frame map is not ordered by upload time")
        adjacent_speeds = np.diff(source_t) / upload_steps
        if not len(adjacent_speeds):
            return skipped(NAME, "video frame map has no increasing upload timestamps")
        speed = float(np.median(adjacent_speeds))
        offset = float(np.median(source_t - speed * upload_t))
        residual = source_t - (speed * upload_t + offset)
        jumps = []
        for i in range(1, len(points)):
            residual_jump = float(residual[i] - residual[i - 1])
            if abs(residual_jump) > 1.0:
                jumps.append({"at_s": round(float(upload_t[i]), 3),
                              "size_s": round(residual_jump, 3)})
        consistent = 0.9 <= speed <= 1.1 and not jumps
        alignment = {
            "ok": True, "door": ctx.door, "method": "median_adjacent_sample_steps",
            "consistent": consistent, "jumps": jumps, "offset_s": round(offset, 4),
            "speed": round(speed, 5),
            "max_residual_s": round(float(np.max(np.abs(residual))), 4),
        }
        ctx.write_json("alignment.json", alignment)
        return ModuleResult(module=NAME, artifacts={"alignment": "alignment.json"}, version=f"{ctx.door}-0.1")
    from modules._common import imagelib as il, picturediff as pd

    ref = ctx.read_json("source_ref.json")
    src = il.read_bgr(paths.library_dir() / ref["file"])
    up = il.read_bgr(ctx.input_path)
    al = pd.align(il.gray(up), il.gray(src))
    al.update(door="image", source_id=ref["source_id"], consistent=al["ok"], jumps=[],
              upload_size=list(up.shape[:2][::-1]), source_size=list(src.shape[:2][::-1]))
    ctx.write_json("alignment.json", al)
    return ModuleResult(module=NAME, artifacts={"alignment": "alignment.json"}, version="rehearsal-0.2")


if __name__ == "__main__":
    run_module(NAME, run)

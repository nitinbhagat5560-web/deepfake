"""Build case folders out of hand-written results, so the verdict engine can be tested without any model."""
import json
from pathlib import Path

from sourcediff import paths
from sourcediff.contract import Evidence, Gate, ModuleResult

SHA = "a" * 64

# module -> (channel, [metrics]) for every Path 1 channel
CHANNEL_MODULES = {
    "picture": ("diff_picture", ["block_diff_max", "detail_dev_max", "block_chroma_diff_max"]),
    "transcript": ("diff_transcript", ["word_error_rate", "audio_correlation"]),
    "sync": ("diff_sync", ["mouth_sharpness_dev", "mouth_motion_diff", "av_sync_correlation"]),
}
GATE_MODULE = {"image": "g1_visual_match", "video": "g1_visual_match"}


def range_of(module, metric):
    return paths.config("genuine_ranges")[f"{module}.{metric}"]["range"]


def inside(module, metric):
    lo, hi = range_of(module, metric)
    return (lo + hi) / 2 if hi > lo else lo


def outside(module, metric):
    lo, hi = range_of(module, metric)
    return hi * 3 + 1.0 if hi > 0 else 5.0


def gate_result(door, matched=True, coverage=1.0, unmatched=(), matched_ranges=(), source="src1", sha=SHA,
                status="ok", stub=False):
    mod = GATE_MODULE[door]
    if status != "ok":
        return ModuleResult(module=mod, status=status, reason="test", input_sha256=sha)
    g = Gate(library="trusted", matched=matched, source_id=source if matched else None,
             coverage_of_upload=coverage if matched else 0.0,
             matched_ranges=[list(r) for r in matched_ranges], unmatched_ranges=[list(r) for r in unmatched])
    return ModuleResult(module=mod, status="ok", gate=g, input_sha256=sha, stub=stub)


def evidence_result(channel, door, values=None, sha=SHA, status="ok", stub=False, supports="neutral",
                    module=None, family="source_diff", chan_override=None):
    """values: {metric: number}; None = every metric of the channel, inside its range."""
    mod, metrics = CHANNEL_MODULES[channel]
    mod = module or mod
    if status != "ok":
        return ModuleResult(module=mod, status=status, reason="test", input_sha256=sha)
    values = values if values is not None else {m: inside(mod, m) for m in metrics}
    rows = [Evidence(module=mod, door=door, family=family, channel=chan_override or channel, metric=m, value=v,
                     supports=supports, stub=stub) for m, v in values.items()]
    return ModuleResult(module=mod, status="ok", evidence=rows, input_sha256=sha, stub=stub)


def aligner_result(sha=SHA, status="ok"):
    return ModuleResult(module="aligner", status=status, reason="" if status == "ok" else "test", input_sha256=sha)


def make_case(root, door="image", flags=None, duration=None, results=(), alignment=None, sha=SHA, probe_ok=True,
              raw_results=None):
    """Write ingest.json, results/*.json and (optionally) alignment.json under `root`/case and return the path."""
    case = Path(root) / "case"
    if case.exists():
        import shutil
        shutil.rmtree(case)
    (case / "results").mkdir(parents=True)
    f = {"has_face": False}
    f.update(flags or {})
    paths.write_json(case / "ingest.json", {"filename": "x.dat", "sha256": sha, "door": door, "duration_s": duration,
                                            "flags": f, "probe_ok": probe_ok, "notes": [] if probe_ok else ["probe failed: test"]})
    for r in results:
        paths.write_json(case / "results" / f"{r.module}.json", r.to_dict())
    for name, text in (raw_results or {}).items():
        (case / "results" / f"{name}.json").write_text(text, encoding="utf-8")
    if alignment is not None:
        paths.write_json(case / "alignment.json", alignment)
    return case


GOOD_ALIGNMENT = {"ok": True, "consistent": True, "jumps": []}

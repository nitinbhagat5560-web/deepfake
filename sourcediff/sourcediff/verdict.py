"""The verdict engine. A deterministic rule table: no model, no LLM, no randomness.

Reads ONLY JSON in the case folder (ingest.json, results/*.json, alignment.json) and
config/ (verdict.json, registry.json, genuine_ranges.json). Writes nothing; the runner
saves the returned dict as verdict.json.

Path 1 rules, in order:
  SHA-256 exactly matches a verified trusted source ............ authentic by byte identity
  Gate 1 matched, full coverage of the UPLOAD, one alignment
      any channel differs ............ edited_picture / edited_spliced
      a channel did not run .......... cant_verify (incomplete diff)   <- never Authentic
      every channel ran and agrees ... authentic                       <- the ONLY way
  Non-exact video matches use video_edited / video_matched_no_edit / video_incomplete;
  "matched" is deliberately not an authenticity claim.
  Gate 1 matched, partial coverage ... not_authentic_partial (unmatched ranges are NOT judged in this build)
  File could not be probed ........... cant_verify (flags are meaningless, never let them switch a channel off)
  Gate 1 did not match ............... cant_verify: "No original found. This does not mean the media is genuine."
  Gate 1 did not run ................. cant_verify: the search did not run for this door

Safety properties, checked by tests/test_golden_rule_fuzz.py on random cases:
  1. authentic is returned only when the rule above holds;
  2. a module that is not named in verdict.json "roles" can never change the verdict;
  3. a result computed on another file never counts.

What this build does NOT have: Gate 2 (known fakes) and Path B (detectors on a file with no original).
Those add the verdicts "likely_synthetic_repost" and "likely_synthetic". Without them a file that has no
original can only ever be "Can't Verify".
"""
from __future__ import annotations

from pathlib import Path

from sourcediff import paths
from sourcediff.contract import Evidence, Gate, ModuleResult, outside_range

ENGINE_VERSION = "0.2-path1"
NO_ORIGINAL = "No original found. This does not mean the media is genuine."
NOT_SEARCHED = ("The original search did not run for this door, so no original was checked. "
                "This does not mean the media is genuine.")

LABELS = {
    "authentic": "Authentic",
    "edited_picture": "Picture edited",
    "edited_video": "Video frames edited",
    "edited_spliced": "Edited / spliced",
    "edited_audio": "Voice / Transcript edited",
    "lips_resynthesized": "Lip sync / Mouth movement altered",
    "not_authentic_partial": "Not Authentic (partial match)",
    "cant_verify": "Can't Verify",
    "video_edited": "Edited",
    "video_matched_no_edit": "Matched — no edit found",
    "video_incomplete": "Can't Verify (incomplete)",
}


# ------------------------------------------------------------------ loading
def _load_results(case_dir: Path, sha: str) -> dict:
    out = {}
    rdir = case_dir / "results"
    for p in sorted(rdir.glob("*.json")) if rdir.exists() else []:
        try:
            res = ModuleResult.from_dict(paths.load_json(p))
        except Exception as exc:
            res = ModuleResult(module=p.stem, status="error", reason=f"unreadable result: {exc}")
        # a result computed on another file must never count
        if res.status == "ok" and res.input_sha256 and sha and res.input_sha256 != sha:
            res = ModuleResult(module=res.module, status="error",
                               reason="result was computed on a different file")
        out[p.stem] = res
    return out


def _applicable(reg_mod: dict, door: str, flags: dict) -> bool:
    return door in reg_mod["doors"] and all(flags.get(n) for n in reg_mod["needs"])


# ------------------------------------------------------------------ evidence judging
def _judge(ev: Evidence, ranges: dict, allow_uncal: bool, uncal_used: set) -> dict:
    """Re-derive 'outside genuine range' from config. The module's own `supports` is ignored."""
    key = f"{ev.module}.{ev.metric}"
    spec = ranges.get(key)
    row = {"module": ev.module, "metric": ev.metric, "value": ev.value, "unit": ev.unit,
           "family": ev.family, "channel": ev.channel, "timestamps": ev.timestamps,
           "limits": ev.limits, "stub": ev.stub, "genuine_range": None, "calibrated": False,
           "outside": None, "counted": False, "role": None, "note": ""}
    if not spec:
        row["note"] = "no genuine range configured, cannot judge"
        return row
    row["genuine_range"] = spec["range"]
    row["calibrated"] = bool(spec.get("calibrated"))
    if not row["calibrated"] and not allow_uncal:
        row["note"] = "range not calibrated, cannot judge"
        return row
    if not row["calibrated"]:
        uncal_used.add(key)
    row["outside"] = outside_range(ev.value, spec["range"])
    return row


# ------------------------------------------------------------------ coverage
def _full_coverage(gate: Gate, duration, cfg) -> bool:
    """Full coverage OF THE UPLOAD: at least min_coverage matched, and any unmatched piece is a
    short sliver at the very start or end (edge_tolerance_s). A cut in the middle is never full."""
    if not gate.matched or gate.coverage_of_upload < cfg["min_coverage"]:
        return False
    tol = cfg["edge_tolerance_s"]
    for a, b in gate.unmatched_ranges:
        at_edge = a <= tol or (duration is not None and b >= duration - tol)
        if (b - a) > tol or not at_edge:
            return False
    return True


# ------------------------------------------------------------------ the engine
def decide(case_dir) -> dict:
    case_dir = Path(case_dir)
    cfg = paths.config("verdict")
    reg = {m["name"]: m for m in paths.config("registry")["modules"]}
    ranges = paths.config("genuine_ranges")
    ingest = paths.load_json(case_dir / "ingest.json")
    door, flags, duration = ingest["door"], ingest.get("flags", {}), ingest.get("duration_s")
    roles = cfg["roles"]
    results = _load_results(case_dir, ingest.get("sha256", ""))

    uncal_used: set = set()
    rows = []
    for res in results.values():
        if res.status == "ok":
            for ev in res.evidence:
                rows.append(_judge(ev, ranges, cfg["allow_uncalibrated"], uncal_used))

    # which modules the engine is allowed to listen to at all
    listed = set()
    for v in roles.values():
        if isinstance(v, dict):
            listed.update(v.values())
        elif isinstance(v, list):
            listed.update(v)
        elif isinstance(v, str):
            listed.add(v)
    for r in rows:
        if r["module"] not in listed:
            r["note"] = "never counted: this module is not a verdict role"

    # ---- channel states for Path 1
    def channel_state(ch):
        if ch == "alignment":
            al_path = case_dir / "alignment.json"
            aligner = results.get(roles["aligner"])
            if not al_path.exists() or aligner is None or aligner.status != "ok":
                return "not_run", "the aligner did not run"
            al = paths.load_json(al_path)
            if not al.get("ok"):
                return "not_run", "could not fit one consistent alignment to the source"
            if al.get("consistent") is False or al.get("jumps"):
                jumps = al.get("jumps") or []
                if jumps:
                    places = ", ".join(
                        f"upload {float(jump['at_s']):.3f}s (jump {float(jump['size_s']):+.3f}s)"
                        for jump in jumps
                    )
                    return "differs", f"timeline has discontinuities at {places}"
                return "differs", "alignment is inconsistent"
            return "agrees", "one consistent alignment"
        mod = roles["channel_modules"][ch]
        if not _applicable(reg[mod], door, flags):
            return "na", "not applicable to this file"
        res = results.get(mod)
        if res is None or res.status != "ok":
            return "not_run", f"{mod} did not run" + (f" ({res.reason})" if res and res.reason else "")
        rs = [r for r in rows if r["module"] == mod and r["family"] == "source_diff" and r["channel"] == ch]
        if not rs or any(r["outside"] is None for r in rs):
            why = "; ".join(sorted({r["note"] for r in rs if r["outside"] is None and r["note"]}))
            return "not_run", f"{mod} produced no judgeable evidence" + (f" ({why})" if why else "")
        for r in rs:
            r["counted"], r["role"] = True, f"channel:{ch}"
        if any(r["outside"] for r in rs):
            details = []
            for row in rs:
                if not row["outside"]:
                    continue
                where = ", ".join(
                    f"upload {float(start):.3f}–{float(end):.3f}s"
                    for start, end in row["timestamps"]
                ) or "whole file"
                details.append(
                    f"{row['metric']}={row['value']} {row['unit']} outside {row['genuine_range']} at {where}"
                )
            return "differs", "; ".join(details)
        return "agrees", "inside the genuine range"

    g1_res = results.get(roles["gate1"][door])
    g1 = g1_res.gate if g1_res is not None and g1_res.status == "ok" else None
    g1_ran = g1 is not None
    out = {
        "verdict": "cant_verify", "reason_code": "", "reasons": [], "notes": [], "warnings": [],
        "door": door, "input_sha256": ingest.get("sha256", ""), "engine_version": ENGINE_VERSION,
        "gate1": g1.to_dict() if g1 else None,
        "channels": {}, "channel_detail": {}, "differing_channels": [], "segments": [],
    }

    def finish(verdict, code, reasons, notes=()):
        out["verdict"], out["reason_code"] = verdict, code
        if door == "video" and verdict == "edited_picture":
            out["label"] = "Video frames edited"
        else:
            out["label"] = LABELS[verdict]
        out["reasons"] += list(reasons)
        out["notes"] += list(notes)
        out["evidence"] = rows
        out["stub_used"] = any(r.stub and r.status == "ok" for r in results.values())
        out["uncalibrated_ranges_used"] = sorted(uncal_used)
        if out["stub_used"]:
            out["warnings"].append("REHEARSAL: stub evidence was used. This verdict is not a real result.")
        if uncal_used:
            out["warnings"].append("Genuine ranges not calibrated: " + ", ".join(sorted(uncal_used)))
        bad = [n for n, r in results.items() if r.status == "error"]
        if bad:
            out["warnings"].append("Modules with errors: " + ", ".join(bad))
        return out

    # ---- a file we could not even read: its flags are meaningless, and a wrong flag switches channels off
    if not ingest.get("probe_ok", True):
        return finish("cant_verify", "unreadable_file",
                      ["The file could not be read or probed, so no check can be trusted."]
                      + [n for n in ingest.get("notes", []) if n.startswith("probe failed")])

    if g1_ran and g1.matched and g1.exact_match:
        if not _full_coverage(g1, duration, cfg):
            return finish("cant_verify", "incomplete_exact_match",
                          ["The exact source identity result did not cover the complete upload."])
        out["exact_identity"] = True
        out["channels"] = {"identity": "identical"}
        out["channel_detail"] = {
            "identity": "SHA-256 matches the verified bytes of the trusted library source."
        }
        out["gate1_full_coverage"] = True
        return finish("authentic", "exact_source_identity",
                      [f"The upload is byte-for-byte identical to trusted source {g1.source_id}; "
                       "no independent media analysis was needed."],
                      ["Exact identity verifies the complete file bytes, not just sampled frames."])

    # ---- Gate 1 matched: Path 1
    if g1_ran and g1.matched:
        order = [c for c in cfg["channel_order"] if c in cfg["door_channels"][door]]
        for ch in order:
            st, detail = channel_state(ch)
            if st != "na":
                out["channels"][ch], out["channel_detail"][ch] = st, detail
        out["differing_channels"] = [c for c in out["channels"] if out["channels"][c] == "differs"]
        not_run = [c for c in out["channels"] if out["channels"][c] == "not_run"]
        full = _full_coverage(g1, duration, cfg)
        out["gate1_full_coverage"] = full

        if door == "video":
            if out["differing_channels"]:
                names = ", ".join(out["differing_channels"])
                return finish("video_edited", "video_channel_differs",
                              [f"The upload matches source {g1.source_id} but at least one video check found a difference: "
                               f"{names}."]
                              + [f"{c}: {out['channel_detail'][c]}" for c in out["differing_channels"]])
            if not full:
                out["segments"].append({
                    "ranges": g1.matched_ranges, "status": "matched",
                    "note": "these ranges matched the trusted source; the unmatched ranges were not checked",
                })
                out["segments"].append({
                    "ranges": g1.unmatched_ranges, "status": "not_judged",
                    "note": "incomplete source coverage",
                })
                return finish("video_incomplete", "incomplete_video_coverage",
                              [f"The upload only partially matched source {g1.source_id}; unmatched portions were not checked."])
            if not_run:
                return finish("video_incomplete", "incomplete_video_checks",
                              [f"The upload matches source {g1.source_id}, but these checks did not run: "
                               f"{', '.join(not_run)}."]
                              + [f"{c}: {out['channel_detail'][c]}" for c in not_run])
            return finish("video_matched_no_edit", "video_checks_inside_ranges",
                          [f"Every applicable check was inside its calibrated range for source {g1.source_id}. "
                           "No edit was found; this is not a claim that the video is authentic."])

        if not full:
            agree = not out["differing_channels"] and not not_run
            out["segments"].append({"ranges": g1.matched_ranges, "status": "verified" if agree else "differs",
                                    "note": "matched the trusted source" + ("" if agree else ", but a diff channel differs or did not run")})
            out["segments"].append({"ranges": g1.unmatched_ranges, "status": "not_judged",
                                    "note": "Path B is not part of this build, so these ranges were not judged."})
            return finish("not_authentic_partial", "partial_coverage",
                          [f"Only part of the upload ({g1.coverage_of_upload:.0%}) matches source {g1.source_id}, "
                           "so the file cannot be called Authentic."])
        if out["differing_channels"]:
            head = out["differing_channels"][0]
            names = ", ".join(out["differing_channels"])
            return finish(cfg["channel_verdicts"][head], "channel_differs",
                          [f"The upload matches source {g1.source_id} but differs on: {names}."]
                          + [f"{c}: {out['channel_detail'][c]}" for c in out["differing_channels"]])
        if not_run:
            return finish("cant_verify", "incomplete_diff",
                          [f"Source {g1.source_id} matched, but these channels did not run: {', '.join(not_run)}. "
                           "An unchecked channel can never count as agreeing."]
                          + [f"{c}: {out['channel_detail'][c]}" for c in not_run])
        return finish("authentic", "full_coverage_all_agree",
                      [f"Every part of the upload matches source {g1.source_id} on one alignment, "
                       "and every channel ran and agrees: " + ", ".join(out["channels"].keys()) + "."])

    # ---- no Gate 1 match: nothing in this build can call it genuine
    if g1_ran:
        return finish("cant_verify", "no_original",
                      ["No trusted original was found in the library."], [NO_ORIGINAL])
    return finish("cant_verify", "search_not_run",
                  ["The original search (Gate 1) did not run for this file type in this build."], [NOT_SEARCHED])

"""Evidence card. Template text only: the engine's verdict is explained, never created, here.

Later, a local Llama (Ollama) can rewrite `summary` in plain language, with a number-check that compares
every figure it wrote against verdict.json and falls back to this template on any difference.
The template is therefore the always-correct baseline.

Hindi strings are a DRAFT and need review by a native speaker before any demo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sourcediff import paths
from sourcediff.audit import canonical, sha256_text
from sourcediff.verdict import ENGINE_VERSION, NO_ORIGINAL

LABELS_HI = {   # DRAFT, needs native review
    "authentic": "प्रामाणिक (मूल स्रोत से मेल खाता है)",
    "edited_picture": "तस्वीर में बदलाव किया गया",
    "edited_video": "वीडियो फ़्रेम में बदलाव किया गया",
    "edited_spliced": "संपादित / काट-जोड़ किया गया",
    "edited_audio": "आवाज़ / प्रतिलेख में बदलाव किया गया",
    "lips_resynthesized": "होंठों की हरकत (लिप-सिंक) बदली गई",
    "not_authentic_partial": "प्रामाणिक नहीं (आंशिक मिलान)",
    "cant_verify": "सत्यापित नहीं किया जा सका",
    "video_edited": "संपादित",
    "video_matched_no_edit": "मिलान — कोई बदलाव नहीं मिला",
    "video_incomplete": "सत्यापित नहीं (जांच अधूरी)",
}
NO_ORIGINAL_HI = "कोई मूल स्रोत नहीं मिला। इसका मतलब यह नहीं कि सामग्री असली है।"

SUMMARY = {
    "authentic": "Every part of this file matches a trusted source, and every check agrees.",
    "edited_picture": "This file comes from a known source, but the picture was changed.",
    "edited_video": "This video comes from a known source, but video frames differ from the original.",
    "edited_spliced": "This file comes from a known source, but pieces were cut, moved or joined.",
    "edited_audio": "This file comes from a known source, but the voice / speech transcript was changed.",
    "lips_resynthesized": "This video comes from a known source, but mouth movements / lip sync differ from the original.",
    "not_authentic_partial": "Only part of this file matches a trusted source. It cannot be called authentic.",
    "cant_verify": "The evidence is not enough to say either way. This is not a finding that the file is genuine.",
    "video_edited": "At least one video comparison found a difference outside its calibrated range.",
    "video_matched_no_edit": "All applicable calibrated video checks were inside range. No edit was found; this is not an authenticity claim.",
    "video_incomplete": "A required video check or part of the upload could not be judged, so the result is incomplete.",
}

CASE_FILES = ("card.md", "card.json", "verdict.json", "ingest.json", "alignment.json",
              "gate1_detail.json", "source_ref.json", "diff_picture_map.png", "video_timeline.png",
              "diff_detail.json", "transcript_diff.json", "sync_diff.json", "path2_result.json",
              "path2_output.jpg", "path2_output.png", "path2_output.webp", "path2_output.bmp",
              "path2_output.mp4", "path2_output.avi", "path2_output.mov", "path2_output.mkv",
              "path2_output.webm")


def build_change_explanation(case_dir: Path, v: dict, ing: dict) -> dict:
    vid = v.get("verdict", "")
    door = ing.get("door", "image")
    is_edited = vid in ("edited_picture", "edited_video", "edited_spliced",
                        "edited_audio", "lips_resynthesized", "video_edited")

    items = []
    edit_types = []
    narrative_parts = []
    location_summary = "Whole file"
    bounding_box = None
    affected_area_pct = None

    def finite_number(value, default=0.0):
        if isinstance(value, (int, float)):
            try:
                value = float(value)
                if value == value and abs(value) != float("inf"):
                    return value
            except (TypeError, ValueError, OverflowError):
                pass
        return default

    diff_detail_file = case_dir / "diff_detail.json"
    diff_detail = paths.load_json(diff_detail_file) if diff_detail_file.exists() else {}

    if diff_detail and diff_detail.get("has_changes"):
        if diff_detail.get("location_summary"):
            location_summary = diff_detail["location_summary"]
        bounding_box = diff_detail.get("bounding_box")
        affected_area_pct = diff_detail.get("affected_area_pct")
        if diff_detail.get("edit_types"):
            edit_types.extend(diff_detail["edit_types"])
        if diff_detail.get("items"):
            items.extend(diff_detail["items"])
        if diff_detail.get("narrative"):
            narrative_parts.append(diff_detail["narrative"])

    al_file = case_dir / "alignment.json"
    if al_file.exists():
        al = paths.load_json(al_file)
        jumps = al.get("jumps") or []
        if jumps:
            edit_types.append("Timeline Spliced / Cuts")
            jump_desc = ", ".join(f"at upload {j['at_s']:.2f}s (jump {j['size_s']:+.2f}s)" for j in jumps)
            narrative_parts.append(f"Timeline discontinuity detected: video segments were cut or spliced ({jump_desc}).")
            for j in jumps:
                items.append({
                    "where": f"Upload time {j['at_s']:.2f}s",
                    "what": "Timeline splice / cut",
                    "details": f"Source time jump of {j['size_s']:+.2f}s",
                })

    td_file = case_dir / "transcript_diff.json"
    if td_file.exists():
        td = paths.load_json(td_file)
        diffs = td.get("differences") or []
        if diffs or finite_number(td.get("word_error_rate")) > 0.35:
            edit_types.append("Speech Altered")
            replaced_words = [d.get("upload") for d in diffs if d.get("upload")]
            w_str = f" ('{', '.join(replaced_words[:3])}')" if replaced_words else ""
            narrative_parts.append(
                f"Audio speech content was altered from the trusted source transcript (WER {td.get('word_error_rate', 0):.2f}){w_str}."
            )
            for d in diffs[:5]:
                t_str = f"Upload {d['timestamps'][0][0]:.2f}–{d['timestamps'][0][1]:.2f}s" if d.get("timestamps") else "Audio track"
                items.append({
                    "where": t_str,
                    "what": f"Speech {d.get('type', 'change')}",
                    "details": f"Original: \"{d.get('original', '')}\" → Upload: \"{d.get('upload', '')}\"",
                })

    sd_file = case_dir / "sync_diff.json"
    if sd_file.exists():
        sd = paths.load_json(sd_file)
        mouth_sharpness = finite_number(sd.get("mouth_sharpness_dev"))
        av_sync = sd.get("av_sync_correlation")
        av_sync_value = finite_number(av_sync, default=1.0) if av_sync is not None else None
        if mouth_sharpness > 0.35 or (av_sync_value is not None and av_sync_value < 0.55):
            edit_types.append("Lip-Sync / Mouth Motion Modified")
            narrative_parts.append("Facial lip-sync or mouth motion deviates from the authentic acoustic audio track.")
            items.append({
                "where": "Mouth region",
                "what": "Lip-sync modification",
                "details": f"Mouth sharpness deviation: {sd.get('mouth_sharpness_dev', 0)}, AV sync correlation: {sd.get('av_sync_correlation', 0)}",
            })

    if not is_edited:
        if vid in ("authentic", "video_matched_no_edit"):
            return {
                "has_changes": False,
                "headline": "No visual or audio alterations found",
                "narrative": "Every analyzed region, frame, and track of this upload matches the trusted original within genuine tolerances.",
                "narrative_hi": "फ़ाइल का हर हिस्सा मूल स्रोत से मेल खाता है। कोई बदलाव नहीं पाया गया।",
                "edit_types": ["Authentic Match"],
                "items": [],
            }
        return {
            "has_changes": False,
            "headline": "Verification incomplete",
            "narrative": "No trusted original was matched in the library, or comparison evidence was incomplete.",
            "narrative_hi": "पुष्टि अधूरी है: पुस्तकालय में कोई मूल स्रोत नहीं मिला।",
            "edit_types": [],
            "items": [],
        }

    if not narrative_parts:
        diff_channels = v.get("differing_channels") or []
        channel_names = ", ".join(diff_channels) or "media"
        narrative_parts.append(f"Modifications were detected in the {channel_names} channel compared to the trusted original.")
        for ch in diff_channels:
            items.append({
                "where": "Upload file",
                "what": f"Discrepancy in {ch}",
                "details": v.get("channel_detail", {}).get(ch, "Differs from genuine range"),
            })

    narrative = " ".join(narrative_parts)
    headline = f"Modifications detected: {', '.join(set(edit_types)) or 'edited content'}"

    return {
        "has_changes": True,
        "headline": headline,
        "location_summary": location_summary,
        "affected_area_pct": affected_area_pct,
        "bounding_box": bounding_box,
        "edit_types": list(dict.fromkeys(edit_types)),
        "items": items,
        "narrative": narrative,
        "narrative_hi": f"फ़ाइल में बदलाव पाया गया: {', '.join(edit_types) or 'संपादित सामग्री'}।",
    }


def build_card(case_dir) -> dict:
    case_dir = Path(case_dir)
    v = paths.load_json(case_dir / "verdict.json")
    ing = paths.load_json(case_dir / "ingest.json")
    vid = v["verdict"]
    ev = sorted(v["evidence"], key=lambda r: (not r["counted"], r["module"], r["metric"]))
    images = {}
    if (case_dir / "diff_picture_map.png").exists():
        images["comparison"] = "diff_picture_map.png"
    if ing["door"] == "video" and (case_dir / "video_timeline.png").exists():
        images["video_timeline"] = "video_timeline.png"
    if (case_dir / "path2_output.png").exists():
        images["path2_face_or_frame_analysis"] = "path2_output.png"
    if any((case_dir / f"path2_output{suffix}").exists() for suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")):
        video_artifact = next(
            f"path2_output{suffix}" for suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")
            if (case_dir / f"path2_output{suffix}").exists()
        )
        images["path2_face_or_frame_analysis"] = video_artifact
    exact_identity = bool(v.get("exact_identity"))
    card = {
        "title": v["label"],
        "verdict": vid,
        "reason_code": v["reason_code"],
        "summary": ("This upload is byte-for-byte identical to a trusted library original."
                    if exact_identity else (
                        "This video comes from a known source, but video frames differ from the original."
                        if (ing.get("door") == "video" and vid == "edited_picture") else SUMMARY[vid]
                    )),
        "summary_hi": ("यह फ़ाइल विश्वसनीय मूल फ़ाइल के bytes से हूबहू मेल खाती है।"
                       if exact_identity else (
                           "यह वीडियो विश्वसनीय स्रोत से है, पर इसके वीडियो फ़्रेम में बदलाव किया गया है।"
                           if (ing.get("door") == "video" and vid == "edited_picture") else LABELS_HI[vid] + ("। " + NO_ORIGINAL_HI if NO_ORIGINAL in v["notes"] else "")
                       )),
        "reasons": v["reasons"],
        "notes": v["notes"],
        "warnings": v["warnings"],
        "input": {"filename": ing["filename"], "sha256": ing["sha256"], "door": ing["door"],
                  "duration_s": ing.get("duration_s")},
        "source": (v.get("gate1") or {}).get("source_id") if (v.get("gate1") or {}).get("matched") else None,
        "channels": [{"channel": c, "state": s, "detail": v["channel_detail"].get(c, "")}
                     for c, s in v["channels"].items()],
        "exact_identity": exact_identity,
        "segments": v["segments"],
        "evidence": [{k: r[k] for k in ("module", "metric", "value", "unit", "genuine_range",
                                        "timestamps", "calibrated", "outside", "counted", "role", "note", "limits")}
                     for r in ev],
        "images": images,
        "stub_used": v["stub_used"],
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if (case_dir / "transcript_diff.json").exists():
        card["transcript_diff"] = paths.load_json(case_dir / "transcript_diff.json")
    if (case_dir / "sync_diff.json").exists():
        card["sync_diff"] = paths.load_json(case_dir / "sync_diff.json")
    card["change_explanation"] = build_change_explanation(case_dir, v, ing)
    card["card_sha256"] = sha256_text(canonical({k: x for k, x in card.items() if k != "generated_at"}))
    return card


def to_markdown(card: dict) -> str:
    L = [f"# {card['title']}", "", card["summary"], ""]
    for w in card["warnings"]:
        L.append(f"> WARNING: {w}")
    if card["warnings"]:
        L.append("")
    if card.get("change_explanation") and card["change_explanation"].get("has_changes"):
        ce = card["change_explanation"]
        L += ["## 🔍 What and Where Was Changed", "",
              ce.get("narrative", ""), ""]
        if ce.get("items"):
            L += ["| Where | What Changed | Details / Evidence |", "|---|---|---|"]
            for it in ce["items"]:
                L.append(f"| {it['where']} | {it['what']} | {it['details']} |")
            L.append("")
    if card.get("path2"):
        p2 = card["path2"]
        explanation = p2.get("explanation") or {}
        L += ["## Path 2 Explainable Evidence", "",
              f"- **Model:** {p2.get('model', 'Unknown')}",
              f"- **Calibration:** {explanation.get('calibration_status', 'uncalibrated')}",
              f"- **Explanation:** {explanation.get('headline', 'Experimental model evidence')}", ""]
        if explanation.get("observations"):
            L += ["### Observations"] + [
                f"- {observation}" for observation in explanation["observations"]
            ] + [""]
        L += ["### Interpretation", explanation.get("summary", ""), ""]
        if explanation.get("generative_report", {}).get("narrative"):
            L += ["### Generated Reporter Narrative",
                  explanation["generative_report"]["narrative"], ""]
        if explanation.get("limitations"):
            L += ["### Limitations"] + [
                f"- {limitation}" for limitation in explanation["limitations"]
            ] + [""]
    if card["reasons"]:
        L += ["## Why"] + [f"- {r}" for r in card["reasons"]] + [""]
    if card.get("transcript_diff"):
        td = card["transcript_diff"]
        L += ["## Speech / Transcript Comparison", "",
              f"- **Original Source Speech:** \"{td.get('source_transcript', '')}\"",
              f"- **Uploaded Video Speech:** \"{td.get('upload_transcript', '')}\"",
              f"- **Word Error Rate (WER):** {td.get('word_error_rate', 0.0)}",
              f"- **Acoustic Correlation:** {td.get('audio_correlation', 0.0)}", ""]
        if td.get("differences"):
            L += ["| Type | Original Speech | Uploaded Speech |", "|---|---|---|"]
            for d in td["differences"]:
                L.append(f"| {d['type']} | {d.get('original', '')} | {d.get('upload', '')} |")
            L.append("")
    if card.get("sync_diff"):
        sd = card["sync_diff"]
        L += ["## Lip-Sync & Face Motion Analysis", "",
              f"- **Mouth Sharpness Deviation:** {sd.get('mouth_sharpness_dev', 0.0)}",
              f"- **Mouth Motion Difference:** {sd.get('mouth_motion_diff', 0.0)}",
              f"- **Audio-Visual Sync Correlation:** {sd.get('av_sync_correlation', 0.0)}", ""]
    if card["channels"]:
        L += ["## Checks against the original", "", "| Check | Result | Detail |", "|---|---|---|"]
        L += [f"| {c['channel']} | {c['state']} | {c['detail']} |" for c in card["channels"]] + [""]
    if card["segments"]:
        L += ["## Segments"] + [f"- {s['status']}: {s['ranges']} {s.get('note', '')}" for s in card["segments"]] + [""]
    used = [e for e in card["evidence"] if e["counted"]]
    if used:
        L += ["## Evidence that counted", "",
              "| Module | Metric | Value | Genuine range | Upload time |",
              "|---|---|---|---|---|"]
        L += [
            f"| {e['module']} | {e['metric']} | {e['value']} {e['unit']} | {e['genuine_range']} | "
            f"{', '.join(f'{start:.3f}–{end:.3f}s' for start, end in e['timestamps']) or 'whole file'} |"
            for e in used
        ] + [""]
    if card["images"]:
        L += ["## Review artifacts", "", ", ".join(sorted(card["images"].values())), ""]
    if card["notes"]:
        L += ["## Notes"] + [f"- {n}" for n in card["notes"]] + [""]
    L += [f"File: {card['input']['filename']} ({card['input']['door']})",
          f"SHA-256: {card['input']['sha256']}", f"Card hash: {card['card_sha256']}",
          f"Engine {card['engine_version']} · {card['generated_at']}", "",
          card["summary_hi"] + "  (draft Hindi)"]
    return "\n".join(L) + "\n"


def write_card(case_dir) -> dict:
    case_dir = Path(case_dir)
    card = build_card(case_dir)
    paths.write_json(case_dir / "card.json", card)
    (case_dir / "card.md").write_text(to_markdown(card), encoding="utf-8")
    return card

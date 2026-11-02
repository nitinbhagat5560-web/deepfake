"""The runner and the module protocol, using hand-written scenario files and a dummy input file."""
import json

import pytest

from sourcediff import paths, runner
from sourcediff.ingest import create_case
from sourcediff.module_api import run_module
from sourcediff.contract import ModuleResult

VIDEO = {"door": "video", "duration_s": 60, "flags": {"has_face": True}}
GATE_OK = {"gate": {"matched": True, "source_id": "orig1", "coverage_of_upload": 1.0, "matched_ranges": [[0, 60]],
                    "unmatched_ranges": []}}
ALIGN_OK = {"ok": True, "consistent": True, "jumps": []}
AGREE = {
    "diff_picture": {"evidence": [{"metric": "block_diff_max", "value": 1.0}]},
    "diff_transcript": {"evidence": [{"metric": "word_error_rate", "value": 0.01}, {"metric": "audio_correlation", "value": 0.95}]},
    "diff_sync": {"evidence": [{"metric": "mouth_sharpness_dev", "value": 0.1}, {"metric": "mouth_motion_diff", "value": 5.0}]},
}


def scenario_case(tmp_path, files):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"not really a video")
    case = create_case(src, "clip.mp4", root=tmp_path / "cases")
    for name, obj in files.items():
        paths.write_json(case / "scenario" / f"{name}.json", obj)
    return case


def test_scenario_video_all_channels_agree_is_conservative_video_match(cfg, tmp_path):
    case = scenario_case(tmp_path, {"ingest": VIDEO, "g1_visual_match": GATE_OK, "alignment": ALIGN_OK, **AGREE})
    out = runner.run_case(case)
    assert out["verdict"] == "video_matched_no_edit" and out["stub_used"]
    card = paths.load_json(case / "card.json")
    assert any("REHEARSAL" in w for w in card["warnings"])
    assert (case / "card.md").exists() and (case / "verdict.json").exists()
    assert out["card"]["card_sha256"] == card["card_sha256"]


def test_scenario_alignment_jump_is_edited_spliced(cfg, tmp_path):
    jump = {"ok": True, "consistent": False, "jumps": [{"at_s": 20, "size_s": 5}]}
    out = runner.run_case(scenario_case(tmp_path, {"ingest": VIDEO, "g1_visual_match": GATE_OK, "alignment": jump, **AGREE}))
    assert out["verdict"] == "video_edited"


def test_a_video_with_unbuilt_modules_is_cant_verify(cfg, tmp_path):
    """No scenario files: the video modules are stubs and report 'did not run'."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    case = create_case(src, "clip.mp4", root=tmp_path / "cases")
    out = runner.run_case(case)
    assert out["verdict"] == "cant_verify"
    assert out["reason_code"] in ("search_not_run", "unreadable_file")


def test_stages_after_gate1_run_only_when_the_original_was_found(cfg, tmp_path):
    out = runner.run_case(scenario_case(tmp_path, {"ingest": VIDEO}))              # gate 1 has no scenario: did not run
    v = paths.load_json(tmp_path / "cases" / out["case_id"] / "verdict.json")
    assert v["stages_run"] == ["gate1"]
    out = runner.run_case(scenario_case(tmp_path, {"ingest": VIDEO, "g1_visual_match": GATE_OK, "alignment": ALIGN_OK}))
    v = paths.load_json(tmp_path / "cases" / out["case_id"] / "verdict.json")
    assert v["stages_run"] == ["gate1", "align", "path1"]


def test_audio_in_video_runs_audio_analysis_channels(cfg, tmp_path):
    with_audio = {"door": "video", "duration_s": 60, "flags": {"has_audio": True, "has_speech": True}}
    case = scenario_case(tmp_path, {"ingest": with_audio, "g1_visual_match": GATE_OK, "alignment": ALIGN_OK,
                                    **AGREE})
    out = runner.run_case(case)
    assert out["verdict"] == "video_matched_no_edit"
    assert set(paths.load_json(case / "verdict.json")["channels"]) == {"alignment", "picture", "transcript"}


def test_a_module_that_cannot_start_becomes_an_error_not_a_crash(cfg, tmp_path):
    case = scenario_case(tmp_path, {"ingest": VIDEO})
    paths.write_json(case / "ingest.json", {"filename": "clip.mp4", "sha256": "x", "door": "video", "flags": {}})
    runner._launch(case, {"name": "no_such_module", "timeout_s": 30}, lambda s: None)
    res = ModuleResult.from_dict(paths.load_json(case / "results" / "no_such_module.json"))
    assert res.status == "error" and res.reason


def test_a_module_that_raises_is_recorded_as_an_error(tmp_path):
    case = tmp_path / "case"
    (case / "input").mkdir(parents=True)
    paths.write_json(case / "ingest.json", {"filename": "x.jpg", "sha256": "s" * 64, "door": "image", "flags": {}})

    def boom(ctx):
        raise RuntimeError("model exploded")

    run_module("boom", boom, argv=["run.py", str(case)])
    res = ModuleResult.from_dict(paths.load_json(case / "results" / "boom.json"))
    assert res.status == "error" and "model exploded" in res.reason and res.input_sha256 == "s" * 64


def test_a_module_that_returns_something_invalid_is_recorded_as_an_error(tmp_path):
    case = tmp_path / "case"
    (case / "input").mkdir(parents=True)
    paths.write_json(case / "ingest.json", {"filename": "x.jpg", "sha256": "s" * 64, "door": "image", "flags": {}})
    run_module("bad", lambda ctx: ModuleResult(module="bad", status="skipped", reason="r", evidence=["junk"]),
               argv=["run.py", str(case)])
    assert paths.load_json(case / "results" / "bad.json")["status"] == "error"


def test_rerunning_a_case_starts_clean(cfg, tmp_path):
    case = scenario_case(tmp_path, {"ingest": VIDEO, "g1_visual_match": GATE_OK, "alignment": ALIGN_OK, **AGREE})
    assert runner.run_case(case)["verdict"] == "video_matched_no_edit"
    (case / "scenario" / "diff_picture.json").unlink()
    out = runner.run_case(case)
    assert out["verdict"] == "video_incomplete"                    # missing channel evidence never gives a clean match

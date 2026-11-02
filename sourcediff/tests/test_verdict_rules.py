"""The verdict rules, one by one, on hand-written results. No model is needed."""
import pytest
from helpers import (CHANNEL_MODULES, GOOD_ALIGNMENT, SHA, aligner_result, evidence_result, gate_result, inside,
                     make_case, outside)

from conftest import set_config
from sourcediff import verdict

AV = {"has_face": True}
SILENT = {"has_face": True}
CHANNELS_OF = {"image": ["picture"], "video": ["picture"]}


def all_agree(door, dur=None, skip=(), replace=None):
    """Results of a perfect run: Gate 1 matched the whole upload, one alignment, every channel inside its range."""
    replace = replace or {}
    res = [gate_result(door, matched_ranges=[(0, dur)] if dur else ()), aligner_result()]
    for ch in CHANNELS_OF[door]:
        if ch in skip:
            continue
        res.append(replace.get(ch) or evidence_result(ch, door))
    return res


def decide(tmp_path, door, flags, results, dur=None, alignment=GOOD_ALIGNMENT, **kw):
    return verdict.decide(make_case(tmp_path, door, flags, dur, results, alignment, **kw))


# ------------------------------------------------------------------ the only way to Authentic
def test_image_authentic(cfg, tmp_path):
    v = decide(tmp_path, "image", {}, all_agree("image"))
    assert v["verdict"] == "authentic" and v["reason_code"] == "full_coverage_all_agree"
    assert v["channels"] == {"alignment": "agrees", "picture": "agrees"}


def test_nonexact_video_never_says_authentic(cfg, tmp_path):
    v = decide(tmp_path, "video", AV, all_agree("video", 60), dur=60)
    assert v["verdict"] == "video_matched_no_edit"
    assert v["label"] == "Matched — no edit found"
    assert set(v["channels"]) == {"alignment", "picture"}


@pytest.mark.parametrize("missing", ["picture"])
def test_missing_channel_is_not_authentic(cfg, tmp_path, missing):
    v = decide(tmp_path, "image", {}, all_agree("image", skip=(missing,)))
    assert v["verdict"] == "cant_verify" and v["reason_code"] == "incomplete_diff"
    assert v["channels"][missing] == "not_run"


@pytest.mark.parametrize("status", ["error", "skipped"])
def test_failed_or_skipped_channel_is_not_authentic(cfg, tmp_path, status):
    res = all_agree("image", replace={"picture": evidence_result("picture", "image", status=status)})
    v = decide(tmp_path, "image", {}, res)
    assert v["verdict"] == "cant_verify" and v["channels"]["picture"] == "not_run"


def test_video_with_missing_picture_check_is_not_authentic(cfg, tmp_path):
    v = decide(tmp_path, "video", AV, all_agree("video", 60, skip=("picture",)), dur=60)
    assert v["verdict"] == "video_incomplete"
    assert "picture" in " ".join(v["reasons"])


# ------------------------------------------------------------------ a channel differs
@pytest.mark.parametrize("channel,label", [("picture", "video_edited")])
def test_differing_channel_names_the_verdict(cfg, tmp_path, channel, label):
    mod, metrics = CHANNEL_MODULES[channel]
    bad = evidence_result(channel, "video", {metrics[0]: outside(mod, metrics[0])})
    bad.evidence[0].timestamps = [[7.0, 8.0]]
    v = decide(tmp_path, "video", AV, all_agree("video", 60, replace={channel: bad}), dur=60)
    assert v["verdict"] == label and v["differing_channels"] == [channel]
    assert "7.000–8.000s" in v["channel_detail"][channel]
    assert metrics[0] in " ".join(v["reasons"])


def test_alignment_jump_is_edited_spliced_even_if_every_channel_agrees(cfg, tmp_path):
    jump = {"ok": True, "consistent": False, "jumps": [{"at_s": 20.0, "size_s": 5.0}]}
    v = decide(tmp_path, "video", AV, all_agree("video", 60), dur=60, alignment=jump)
    assert v["verdict"] == "video_edited" and v["channels"]["alignment"] == "differs"
    assert "upload 20.000s" in v["channel_detail"]["alignment"]


def test_jumps_without_consistent_flag_still_count(cfg, tmp_path):
    jump = {"ok": True, "jumps": [{"at_s": 3.0, "size_s": 1.0}]}
    assert decide(tmp_path, "video", AV, all_agree("video", 60), dur=60, alignment=jump)["verdict"] == "video_edited"


def test_alignment_that_could_not_be_fitted_is_cant_verify(cfg, tmp_path):
    v = decide(tmp_path, "image", {}, all_agree("image"), alignment={"ok": False, "reason": "too few matches"})
    assert v["verdict"] == "cant_verify" and v["channels"]["alignment"] == "not_run"


def test_missing_alignment_file_is_cant_verify(cfg, tmp_path):
    assert decide(tmp_path, "image", {}, all_agree("image"), alignment=None)["verdict"] == "cant_verify"


# ------------------------------------------------------------------ Gate 1
def test_no_match_is_cant_verify_with_the_mandatory_sentence(cfg, tmp_path):
    v = decide(tmp_path, "image", {}, [gate_result("image", matched=False)])
    assert v["verdict"] == "cant_verify" and v["reason_code"] == "no_original"
    assert verdict.NO_ORIGINAL in v["notes"]


def test_gate_that_did_not_run_is_cant_verify(cfg, tmp_path):
    for res in ([], [gate_result("video", status="error")], [gate_result("video", status="skipped")]):
        v = decide(tmp_path, "video", AV, res, dur=30)
        assert v["verdict"] == "cant_verify" and v["reason_code"] == "search_not_run"
        assert verdict.NOT_SEARCHED in v["notes"]


# ------------------------------------------------------------------ coverage is measured on the UPLOAD
def test_genuine_30s_cut_of_a_long_original_is_matched_not_authentic(cfg, tmp_path):
    res = all_agree("video", 30)                        # the whole 30 s upload matched a 2-minute source
    assert decide(tmp_path, "video", AV, res, dur=30)["verdict"] == "video_matched_no_edit"


def test_half_matched_upload_is_not_authentic(cfg, tmp_path):
    res = [gate_result("video", coverage=0.5, matched_ranges=[(0, 15)], unmatched=[(15, 30)]), aligner_result()] \
        + [evidence_result(c, "video") for c in CHANNELS_OF["video"]]
    v = decide(tmp_path, "video", AV, res, dur=30)
    assert v["verdict"] == "video_incomplete"
    assert [s["status"] for s in v["segments"]] == ["matched", "not_judged"]


@pytest.mark.parametrize("unmatched,full", [([(0, 0.6)], True),            # sliver at the start
                                            ([(29.4, 30)], True),          # sliver at the end
                                            ([(10, 10.6)], False),         # same sliver in the middle
                                            ([(0, 3)], False)])            # an edge piece that is too long
def test_unmatched_pieces(cfg, tmp_path, unmatched, full):
    res = [gate_result("video", coverage=0.98, matched_ranges=[(0, 30)], unmatched=unmatched), aligner_result()] \
        + [evidence_result(c, "video") for c in CHANNELS_OF["video"]]
    v = decide(tmp_path, "video", AV, res, dur=30)
    assert v["verdict"] == ("video_matched_no_edit" if full else "video_incomplete")


# ------------------------------------------------------------------ calibration
def test_uncalibrated_range_cannot_be_judged(cfg, tmp_path):
    import json
    p = cfg / "genuine_ranges.json"
    gr = json.loads(p.read_text(encoding="utf-8"))
    gr["diff_picture.block_diff_max"]["calibrated"] = False
    p.write_text(json.dumps(gr), encoding="utf-8")
    v = decide(tmp_path, "image", {}, all_agree("image"))
    assert v["verdict"] == "cant_verify" and v["channels"]["picture"] == "not_run"
    assert "not calibrated" in v["channel_detail"]["picture"]


def test_allow_uncalibrated_judges_but_warns(cfg, tmp_path):
    import json
    p = cfg / "genuine_ranges.json"
    gr = json.loads(p.read_text(encoding="utf-8"))
    gr["diff_picture.block_diff_max"]["calibrated"] = False
    p.write_text(json.dumps(gr), encoding="utf-8")
    set_config(cfg, "verdict", allow_uncalibrated=True)
    v = decide(tmp_path, "image", {}, all_agree("image"))
    assert v["verdict"] == "authentic"
    assert any("not calibrated" in w for w in v["warnings"])


def test_metric_with_no_configured_range_cannot_be_judged(cfg, tmp_path):
    res = all_agree("image", replace={"picture": evidence_result("picture", "image", {"made_up_metric": 1.0})})
    assert decide(tmp_path, "image", {}, res)["verdict"] == "cant_verify"


def test_range_edges_are_inside(cfg, tmp_path):
    lo, hi = 0.0, None
    from helpers import range_of
    hi = range_of("diff_picture", "block_diff_max")[1]
    ok = evidence_result("picture", "image", {"block_diff_max": hi, "detail_dev_max": inside("diff_picture", "detail_dev_max")})
    assert decide(tmp_path, "image", {}, all_agree("image", replace={"picture": ok}))["verdict"] == "authentic"
    over = evidence_result("picture", "image", {"block_diff_max": hi + 0.001, "detail_dev_max": inside("diff_picture", "detail_dev_max")})
    assert decide(tmp_path, "image", {}, all_agree("image", replace={"picture": over}))["verdict"] == "edited_picture"


# ------------------------------------------------------------------ what a module may and may not do
def test_module_opinion_is_ignored(cfg, tmp_path):
    lying_genuine = evidence_result("picture", "image", {"block_diff_max": outside("diff_picture", "block_diff_max"),
                                                         "detail_dev_max": inside("diff_picture", "detail_dev_max")},
                                    supports="genuine")
    assert decide(tmp_path, "image", {}, all_agree("image", replace={"picture": lying_genuine}))["verdict"] == "edited_picture"
    lying_synthetic = evidence_result("picture", "image", supports="synthetic")
    assert decide(tmp_path, "image", {}, all_agree("image", replace={"picture": lying_synthetic}))["verdict"] == "authentic"


def test_unlisted_module_never_moves_the_verdict(cfg, tmp_path):
    stray = evidence_result("picture", "image", {"block_diff_max": 999.0}, module="evil_module")
    v = decide(tmp_path, "image", {}, all_agree("image") + [stray])
    assert v["verdict"] == "authentic"
    assert any(r["module"] == "evil_module" for r in v["evidence"])        # shown, never counted
    assert not any(r["counted"] for r in v["evidence"] if r["module"] == "evil_module")


def test_result_computed_on_another_file_never_counts(cfg, tmp_path):
    res = all_agree("image", replace={"picture": evidence_result("picture", "image", sha="b" * 64)})
    v = decide(tmp_path, "image", {}, res)
    assert v["verdict"] == "cant_verify" and "different file" in v["channel_detail"]["picture"]


def test_gate_result_for_another_file_never_counts(cfg, tmp_path):
    res = [gate_result("image", sha="b" * 64)] + all_agree("image")[1:]
    assert decide(tmp_path, "image", {}, res)["verdict"] == "cant_verify"


def test_unreadable_result_file_counts_as_not_run(cfg, tmp_path):
    res = [r for r in all_agree("image") if r.module != "diff_picture"]
    v = decide(tmp_path, "image", {}, res, raw_results={"diff_picture": "{ this is not json"})
    assert v["verdict"] == "cant_verify" and any("diff_picture" in w for w in v["warnings"])


def test_unprobeable_file_is_cant_verify_whatever_the_results_say(cfg, tmp_path):
    v = decide(tmp_path, "video", SILENT, all_agree("video", 60), dur=60, probe_ok=False)
    assert v["verdict"] == "cant_verify" and v["reason_code"] == "unreadable_file"


def test_stub_evidence_is_flagged_as_rehearsal(cfg, tmp_path):
    res = all_agree("image", replace={"picture": evidence_result("picture", "image", stub=True)})
    v = decide(tmp_path, "image", {}, res)
    assert v["stub_used"] and any("REHEARSAL" in w for w in v["warnings"])


def test_every_verdict_has_a_label(cfg):
    assert set(verdict.LABELS) >= {"authentic", "edited_picture", "edited_spliced",
                                   "not_authentic_partial", "cant_verify", "video_edited",
                                   "video_matched_no_edit", "video_incomplete"}

"""Core sampled-frame video path, with calibration kept fail-closed."""
import shutil

import cv2
import numpy as np
import pytest

import _variants as V
from modules._common import imagelib as il, picturediff as pd, videolib
from sourcediff import paths, runner
from sourcediff.contract import Gate
from sourcediff.ingest import create_case, sha256_file
from sourcediff.module_api import Ctx
from sourcediff import library


def make_video(path, seconds=7, seed_offset=0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable")
    for second in range(seconds):
        frame = np.random.default_rng(seed_offset + second).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def video_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_LIBRARY", str(tmp_path / "library"))
    monkeypatch.setenv("SOURCEDIFF_DATA", str(tmp_path / "data"))
    config_dir = tmp_path / "config"
    shutil.copytree(paths.config_dir(), config_dir)
    monkeypatch.setenv("SOURCEDIFF_CONFIG", str(config_dir))
    return tmp_path


def test_silent_video_runs_gate_alignment_and_frame_comparison(video_data, monkeypatch):
    gates = paths.load_json(paths.config_dir() / "gates.json")
    gates["video"]["calibrated"] = True
    paths.write_json(paths.config_dir() / "gates.json", gates)
    source = make_video(video_data / "source.avi")
    library.add_item(source, source.name, "source", source_id="source", strict=False, rebuild=True)
    unrelated = make_video(video_data / "unrelated.avi", seed_offset=100)
    unrelated_samples, unrelated_duration = videolib.sample_frames(unrelated)
    threshold = paths.config("gates")["video"]["trusted_threshold"]
    assert not videolib.match_upload(unrelated_samples, unrelated_duration, threshold)["matched"]

    upload = video_data / "upload.avi"
    shutil.copyfile(source, upload)
    with upload.open("ab") as copied_video:
        copied_video.write(b"container trailing metadata")
    case = create_case(upload, upload.name, root=video_data / "cases")
    _, duration = videolib.sample_frames(upload)
    ingest_info = {
        "filename": upload.name, "sha256": sha256_file(case / "input" / upload.name),
        "door": "video", "duration_s": duration, "flags": {"has_audio": False, "has_speech": False},
        "probe_ok": True,
    }

    def fake_ingest(case_dir):
        paths.write_json(case_dir / "ingest.json", ingest_info)
        return ingest_info

    monkeypatch.setattr(runner, "ingest", fake_ingest)
    output = runner.run_case(case)

    gate = paths.load_json(case / "results" / "g1_visual_match.json")["gate"]
    alignment = paths.load_json(case / "alignment.json")
    picture = paths.load_json(case / "results" / "diff_picture.json")
    assert gate["matched"] and gate["source_id"] == "source"
    assert gate["coverage_of_upload"] == 1.0
    assert alignment["ok"] and alignment["consistent"]
    assert picture["status"] == "ok"
    assert len(picture["evidence"]) == len(videolib.frame_timestamps(upload)[0]) * 3
    assert all(row["door"] == "video" and row["timestamps"] for row in picture["evidence"])
    assert (case / "diff_picture_map.png").is_file()
    assert (case / "video_timeline.png").is_file()
    assert output["card"]["images"]["video_timeline"] == "video_timeline.png"
    timeline = cv2.imread(str(case / "video_timeline.png"))
    assert timeline is not None and timeline.shape[0] <= 3 * (260 + 28 + 12)
    assert output["verdict"] == "video_matched_no_edit"
    verdict = paths.load_json(case / "verdict.json")
    assert verdict["channels"]["picture"] == "agrees"
    assert all(not row["note"].startswith("range not calibrated") for row in verdict["evidence"])


def test_downscaled_compressed_video_frames_align_and_compare(tmp_path):
    source_path = tmp_path / "source.avi"
    upload_path = tmp_path / "upload.avi"
    source_writer = cv2.VideoWriter(str(source_path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (320, 240))
    upload_writer = cv2.VideoWriter(str(upload_path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (160, 120))
    if not source_writer.isOpened() or not upload_writer.isOpened():
        source_writer.release()
        upload_writer.release()
        pytest.skip("OpenCV MJPG video writer is unavailable")

    scene = cv2.resize(V.sample("camera"), (320, 240), interpolation=cv2.INTER_AREA)
    try:
        for second in range(4):
            frame = scene.copy()
            cv2.putText(frame, str(second // 2), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            for _ in range(2):
                source_writer.write(frame)
                upload_writer.write(V.reupload(frame, 30, 0.5))
    finally:
        source_writer.release()
        upload_writer.release()

    timestamps = [0.0, 1.0, 2.0, 3.0]
    uploads = videolib.iter_frames_at(upload_path, timestamps)
    originals = videolib.iter_frames_at(source_path, timestamps)
    for upload, original in zip(uploads, originals):
        alignment = pd.align(il.gray(upload), il.gray(original))
        assert alignment["ok"] and 1.8 < alignment["scale"] < 2.2
        comparison = pd.compare(upload, original, alignment["H"])
        assert comparison["ok"] and comparison["block_diff_max"] < 5.0


def test_spatially_cropped_video_matches_source_and_covers_the_upload(video_data, tmp_path):
    source_path = video_data / "source.avi"
    upload_path = video_data / "cropped.avi"
    source_writer = cv2.VideoWriter(str(source_path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (320, 240))
    upload_writer = cv2.VideoWriter(str(upload_path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (160, 120))
    if not source_writer.isOpened() or not upload_writer.isOpened():
        source_writer.release()
        upload_writer.release()
        pytest.skip("OpenCV MJPG video writer is unavailable")

    scene = cv2.resize(V.sample("camera"), (320, 240), interpolation=cv2.INTER_AREA)
    try:
        for second in range(7):
            frame = scene.copy()
            cv2.putText(frame, str(second), (90 + second * 5, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            crop = frame[60:180, 80:240]
            for _ in range(2):
                source_writer.write(frame)
                upload_writer.write(crop)
    finally:
        source_writer.release()
        upload_writer.release()

    library.add_item(source_path, source_path.name, "source", source_id="source",
                     strict=False, rebuild=True)
    samples, duration = videolib.sample_frames(upload_path)
    threshold = float(paths.config("gates")["video"]["trusted_threshold"])
    result = videolib.match_upload(samples, duration, threshold)

    assert result["matched"] and result["source_id"] == "source", result["sample_hits"]
    assert result["coverage_of_upload"] == 1.0
    assert result["matched_ranges"] == [[0.0, duration]]
    upload_frame = samples[0][1]
    source_frame = next(videolib.iter_frames_at(source_path, [result["points"][0][1]]))
    alignment = pd.align(il.gray(upload_frame), il.gray(source_frame))
    assert alignment["ok"]


def test_video_excerpt_uses_upload_duration_for_full_coverage(video_data, tmp_path):
    source_path = make_video(video_data / "long-source.avi", seconds=30)
    library.add_item(source_path, source_path.name, "long source", source_id="long-source",
                     strict=False, rebuild=True)

    excerpt_path = video_data / "first-ten-seconds.avi"
    source_frames = videolib.iter_frames_at(source_path, [float(t) for t in range(10)])
    writer = cv2.VideoWriter(str(excerpt_path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable")
    try:
        for frame in source_frames:
            writer.write(frame)
            writer.write(frame)
    finally:
        writer.release()

    samples, duration = videolib.sample_frames(excerpt_path)
    threshold = float(paths.config("gates")["video"]["trusted_threshold"])
    result = videolib.match_upload(samples, duration, threshold)

    assert duration == pytest.approx(10.0)
    assert result["matched"] and result["source_id"] == "long-source"
    assert result["coverage_of_upload"] == 1.0
    assert result["matched_ranges"] == [[0.0, duration]]
    assert result["points"][0][1] == pytest.approx(0.0)
    assert result["points"][-1][1] == pytest.approx(9.0)


def test_exact_video_copy_is_verified_without_calibrated_video_threshold(video_data, monkeypatch):
    source = make_video(video_data / "source.avi")
    library.add_item(source, source.name, "source", source_id="source", strict=False, rebuild=False)
    upload = video_data / "upload.avi"
    shutil.copyfile(source, upload)
    case = create_case(upload, upload.name, root=video_data / "cases")
    _, duration = videolib.sample_frames(upload)
    ingest_info = {
        "filename": upload.name, "sha256": sha256_file(case / "input" / upload.name),
        "door": "video", "duration_s": duration, "flags": {"has_audio": False, "has_speech": False},
        "probe_ok": True,
    }

    def fake_ingest(case_dir):
        paths.write_json(case_dir / "ingest.json", ingest_info)
        return ingest_info

    monkeypatch.setattr(runner, "ingest", fake_ingest)
    output = runner.run_case(case)

    assert output["verdict"] == "authentic"
    verdict = paths.load_json(case / "verdict.json")
    assert verdict["reason_code"] == "exact_source_identity"
    assert verdict["exact_identity"] is True
    assert verdict["stages_run"] == ["gate1"]


def test_video_alignment_marks_a_cut_as_a_timeline_jump(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_LIBRARY", str(tmp_path / "library"))
    case = tmp_path / "case"
    case.mkdir()
    points = [[0, 0], [1, 1], [2, 2], [3, 8], [4, 9], [5, 10]]
    paths.write_json(case / "chunk_map.json", points)
    ctx = Ctx(case_dir=case, ingest={"door": "video"})

    from modules.aligner.run import run
    result = run(ctx)

    alignment = paths.load_json(case / "alignment.json")
    assert result.status == "ok"
    assert alignment["ok"] and not alignment["consistent"]
    assert alignment["jumps"] and alignment["jumps"][0]["at_s"] == 3.0


def test_gate_preserves_both_sides_of_a_cut_for_alignment(tmp_path, monkeypatch):
    source_times = [0, 1, 2, 8, 9, 10, 11, 12]
    monkeypatch.setattr(
        videolib, "search_many",
        lambda _kind, vectors: [("source", float(t), 1.0) for t in source_times],
    )
    samples = [(float(t), np.zeros((24, 24, 3), dtype=np.uint8)) for t in range(8)]
    matched = videolib.match_upload(samples, duration_s=8.0, threshold=0.9)

    assert matched["matched"] and matched["coverage_of_upload"] == 1.0
    assert len(matched["points"]) == 8
    case = tmp_path / "case"
    case.mkdir()
    paths.write_json(case / "chunk_map.json", matched["points"])
    ctx = Ctx(case_dir=case, ingest={"door": "video"})

    from modules.aligner.run import run
    run(ctx)
    alignment = paths.load_json(case / "alignment.json")
    assert alignment["jumps"] and alignment["jumps"][0]["at_s"] == 3.0


def test_video_match_rounding_does_not_create_zero_length_final_gap(monkeypatch):
    duration = 15.6333333333
    samples = [(float(t), np.zeros((24, 24, 3), dtype=np.uint8)) for t in range(16)]
    monkeypatch.setattr(
        videolib, "search_many",
        lambda _kind, vectors: [("source", float(t), 1.0) for t in range(len(vectors))],
    )

    matched = videolib.match_upload(samples, duration, threshold=0.9)
    gate = Gate(
        library="trusted", matched=True, source_id=matched["source_id"],
        coverage_of_upload=matched["coverage_of_upload"],
        matched_ranges=matched["matched_ranges"], unmatched_ranges=matched["unmatched_ranges"],
    ).validate()

    assert gate.unmatched_ranges == []
    assert matched["coverage_of_upload"] == 1.0


def test_video_search_reports_a_missing_index(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_LIBRARY", str(tmp_path / "library"))
    with pytest.raises(FileNotFoundError, match="build-index"):
        videolib.search_many("trusted", np.zeros((1, 1024), dtype=np.float32))

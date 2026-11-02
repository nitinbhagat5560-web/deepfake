"""diff_sync  [path1]  channel: sync  doors: video lip-sync comparison

Compares facial mouth-region sharpness against upper face and measures lip motion / 
audio-visual sync consistency between the upload and the aligned trusted library source.

Catches re-dubbed lips, face re-enactments (e.g. Wav2Lip, SadTalker) even when
speech transcript matches.

The module never says 'edited'. It reports numbers; the engine compares them with config/genuine_ranges.json.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from modules._common import imagelib as il, videolib as vl
from sourcediff import paths
from sourcediff.contract import Evidence, ModuleResult
from sourcediff.module_api import genuine_range, in_scenario, run_module, skipped, stub_run

NAME = "diff_sync"


def _extract_face_and_mouth(frame: np.ndarray):
    """Detects face and partitions upper-face and mouth region."""
    h, w = frame.shape[:2]
    # Check Haar cascade first if available
    box = None
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        try:
            cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not cas.empty():
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cas.detectMultiScale(g, 1.1, 4, minSize=(60, 60))
                if len(faces) > 0:
                    box = max(faces, key=lambda b: b[2] * b[3])
        except Exception:
            box = None

    # Robust skin/face region fallback
    if box is None:
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 0.01 * h * w]
            if boxes:
                box = max(boxes, key=lambda b: b[2] * b[3])
        except Exception:
            box = None

    if box is None:
        return None, None

    bx, by, bw, bh = box
    bx = max(0, min(w - 1, bx))
    by = max(0, min(h - 1, by))
    bw = max(10, min(w - bx, bw))
    bh = max(10, min(h - by, bh))

    upper_face = frame[by : by + int(0.50 * bh), bx : bx + bw]
    mouth = frame[by + int(0.65 * bh) : by + bh, bx : bx + bw]
    return upper_face, mouth


def _sharpness(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _extract_audio_envelope(wav_path: Path, n_points: int) -> np.ndarray | None:
    try:
        from scipy.io import wavfile
        _, d = wavfile.read(str(wav_path))
        if d.ndim > 1:
            d = d[:, 0]
        data = d.astype(np.float32)
        if np.issubdtype(d.dtype, np.integer):
            if np.issubdtype(d.dtype, np.signedinteger):
                scale = max(1.0, float(np.iinfo(d.dtype).max))
            else:
                center = float(np.iinfo(d.dtype).max) / 2.0
                data = data - center
                scale = max(1.0, center)
        else:
            scale = 1.0
        data = np.abs(data / scale)
        if not len(data) or n_points <= 0:
            return None
        chunk_len = max(1, len(data) // max(1, n_points))
        env = [float(np.mean(data[i * chunk_len : (i + 1) * chunk_len])) for i in range(n_points)]
        return np.array(env, dtype=np.float32)
    except (OSError, ValueError):
        return None


def _av_sync_correlation(wav_path: Path, mouth_motions: list[float]) -> float | None:
    if len(mouth_motions) < 4:
        return None
    envelope = _extract_audio_envelope(wav_path, len(mouth_motions))
    if envelope is None or len(envelope) != len(mouth_motions):
        return None
    motion = np.asarray(mouth_motions, dtype=np.float32)
    if np.std(envelope) <= 1e-4 or np.std(motion) <= 1e-4:
        return None
    correlations = []
    for lag in range(-3, 4):
        if lag < 0:
            audio = envelope[-lag:]
            visual = motion[:len(audio)]
        elif lag > 0:
            audio = envelope[:-lag]
            visual = motion[lag:]
        else:
            audio = envelope
            visual = motion
        if len(audio) < 4 or np.std(audio) <= 1e-4 or np.std(visual) <= 1e-4:
            continue
        value = float(np.corrcoef(audio, visual)[0, 1])
        if np.isfinite(value):
            correlations.append(max(0.0, value))
    return max(correlations) if correlations else None


def run(ctx):
    if in_scenario(ctx):
        return stub_run(ctx, NAME)
    if ctx.door != "video":
        return skipped(NAME, f"unsupported sync door: {ctx.door}")

    flags = ctx.ingest.get("flags", {})
    if not flags.get("has_face"):
        return skipped(NAME, "video contains no detectable face")

    al = ctx.read_json("alignment.json")
    if not al.get("ok") or al.get("consistent") is False:
        return skipped(NAME, "no consistent video alignment to compare lip sync")

    ref = ctx.read_json("source_ref.json")
    source = vl.source_path("trusted", ref["source_id"])
    if not source.is_file():
        return skipped(NAME, f"source video file not found: {ref.get('source_id')}")

    timestamps, duration = vl.frame_timestamps(ctx.input_path)
    offset, speed = float(al["offset_s"]), float(al["speed"])
    source_timestamps = [speed * t + offset for t in timestamps]

    sharpness_by_second = {}
    mouth_diff_by_second = {}
    mouth_motions = []

    try:
        upload_frames = list(vl.iter_frames_at(ctx.input_path, timestamps))
        source_frames = list(vl.iter_frames_at(source, source_timestamps))
    except Exception as exc:
        return skipped(NAME, f"could not decode frames for sync diff: {exc}")

    prev_m_up = None
    paired_samples = 0
    for t, up, src in zip(timestamps, upload_frames, source_frames):
        up_upper, up_mouth = _extract_face_and_mouth(up)
        src_upper, src_mouth = _extract_face_and_mouth(src)

        if up_mouth is not None and src_mouth is not None and up_upper is not None and src_upper is not None:
            paired_samples += 1
            second = int(t)
            r_up = _sharpness(up_mouth) / max(1.0, _sharpness(up_upper))
            r_src = _sharpness(src_mouth) / max(1.0, _sharpness(src_upper))
            dev = abs(r_up - r_src)
            sharpness_by_second.setdefault(second, []).append(dev)

            # Resize source mouth to match upload mouth for spatial motion diff
            sm_resized = cv2.resize(src_mouth, (up_mouth.shape[1], up_mouth.shape[0]))
            diff = float(np.mean(np.abs(up_mouth.astype(np.float32) - sm_resized.astype(np.float32))))
            mouth_diff_by_second.setdefault(second, []).append(diff)

            if prev_m_up is not None and prev_m_up.shape == up_mouth.shape:
                motion = float(np.mean(np.abs(up_mouth.astype(np.float32) - prev_m_up.astype(np.float32))))
                mouth_motions.append(motion)
            else:
                mouth_motions.append(0.0)
            prev_m_up = up_mouth

    if paired_samples == 0:
        return skipped(NAME, "no paired mouth crops were compared")
    expected_seconds = {int(t) for t in timestamps}
    measured_seconds = set(sharpness_by_second)
    if measured_seconds != expected_seconds:
        missing = sorted(expected_seconds - measured_seconds)
        return skipped(NAME, f"mouth crops could not be paired at upload second(s): {missing}")

    # AV Sync Correlation against audio energy envelope
    if not flags.get("has_audio"):
        return skipped(NAME, "video has no audio track for AV correlation")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "up_audio.wav"
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = "ffmpeg"
        cmd = [ffmpeg, "-y", "-i", str(ctx.input_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if completed.returncode != 0 or not wav_path.is_file() or wav_path.stat().st_size <= 44:
            return skipped(NAME, "could not extract audio for AV correlation")
        sync_corr = _av_sync_correlation(wav_path, mouth_motions)
    if sync_corr is None:
        sync_artifact = {
            "ok": True,
            "per_second": [
                {"second": second, "mouth_sharpness_dev": round(max(values), 3),
                 "mouth_motion_diff": round(float(np.mean(mouth_diff_by_second[second])), 2)}
                for second, values in sorted(sharpness_by_second.items())
            ],
            "av_sync_correlation": None,
            "frames_checked": paired_samples,
            "av_sync_status": "not_measured",
            "reason": "audio envelope and mouth motion had insufficient or inconsistent variance",
        }
        ctx.write_json("sync_diff.json", sync_artifact)
        evidence = []
        for row in sync_artifact["per_second"]:
            start = float(row["second"])
            end = min(duration, start + 1.0)
            if end <= start:
                continue
            evidence.extend((
                Evidence(
                    module=NAME, door="video", family="source_diff", channel="sync",
                    metric="mouth_sharpness_dev", value=row["mouth_sharpness_dev"], unit="sharpness dev",
                    genuine_range=genuine_range(NAME, "mouth_sharpness_dev"),
                    timestamps=[[start, round(end, 3)]],
                    limits=["per-second mouth-region sharpness deviation; AV correlation unavailable"],
                ),
                Evidence(
                    module=NAME, door="video", family="source_diff", channel="sync",
                    metric="mouth_motion_diff", value=row["mouth_motion_diff"], unit="motion diff",
                    genuine_range=genuine_range(NAME, "mouth_motion_diff"),
                    timestamps=[[start, round(end, 3)]],
                    limits=["per-second mouth-region motion difference; AV correlation unavailable"],
                ),
            ))
        return ModuleResult(
            module=NAME, evidence=evidence, artifacts={"sync_diff": "sync_diff.json"},
            version="sync-0.3",
        )

    # Write summary artifact
    sharpness_rows = [
        {"second": second, "mouth_sharpness_dev": round(max(values), 3),
         "mouth_motion_diff": round(float(np.mean(mouth_diff_by_second[second])), 2)}
        for second, values in sorted(sharpness_by_second.items())
    ]
    sync_artifact = {
        "ok": True,
        "per_second": sharpness_rows,
        "av_sync_correlation": round(sync_corr, 3),
        "frames_checked": paired_samples,
    }
    ctx.write_json("sync_diff.json", sync_artifact)

    evidence = []
    for row in sharpness_rows:
        start = float(row["second"])
        end = min(duration, start + 1.0)
        if end <= start:
            continue
        evidence.extend((
            Evidence(
                module=NAME, door="video", family="source_diff", channel="sync",
                metric="mouth_sharpness_dev", value=row["mouth_sharpness_dev"], unit="sharpness dev",
                genuine_range=genuine_range(NAME, "mouth_sharpness_dev"),
                timestamps=[[start, round(end, 3)]],
                limits=["per-second mouth-region sharpness deviation against the aligned source"],
            ),
            Evidence(
                module=NAME, door="video", family="source_diff", channel="sync",
                metric="mouth_motion_diff", value=row["mouth_motion_diff"], unit="motion diff",
                genuine_range=genuine_range(NAME, "mouth_motion_diff"),
                timestamps=[[start, round(end, 3)]],
                limits=["per-second mouth-region difference against the aligned source"],
            ),
        ))
    evidence.append(Evidence(
        module=NAME, door="video", family="source_diff", channel="sync",
        metric="av_sync_correlation", value=round(sync_corr, 3), unit="correlation",
        genuine_range=genuine_range(NAME, "av_sync_correlation"),
        timestamps=[[0.0, round(duration, 3)]],
        limits=["audio-visual synchronization correlation across the analyzed clip"],
    ))

    return ModuleResult(
        module=NAME,
        evidence=evidence,
        artifacts={"sync_diff": "sync_diff.json"},
        version="sync-0.2",
    )


if __name__ == "__main__":
    run_module(NAME, run)

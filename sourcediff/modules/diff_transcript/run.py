"""Compare speech inside aligned videos using local transcription and word timestamps."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from sourcediff import paths
from sourcediff.contract import Evidence, ModuleResult
from sourcediff.module_api import in_scenario, run_module, skipped, stub_run

NAME = "diff_transcript"


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.casefold())


def _extract_wav(video_path: Path, out_wav: Path, start_s: float = 0.0,
                 duration_s: float | None = None) -> bool:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    cmd = [ffmpeg, "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    if duration_s is not None and duration_s > 0:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(out_wav)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return cp.returncode == 0 and out_wav.is_file() and out_wav.stat().st_size > 44
    except subprocess.TimeoutExpired:
        return False


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required for local transcript comparison") from exc
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8",
                            cpu_threads=max(1, min(4, os.cpu_count() or 1)))
    except Exception:
        if model_name == "tiny":
            raise
        return WhisperModel("tiny", device="cpu", compute_type="int8",
                            cpu_threads=max(1, min(4, os.cpu_count() or 1)))


def _transcribe(wav_path: Path) -> tuple[str, list[dict]]:
    model_name = os.environ.get("SOURCEDIFF_WHISPER_MODEL", "tiny")
    model = _load_whisper_model(model_name)
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True, vad_filter=True)
    words = []
    for segment in segments:
        for word in segment.words or ():
            text = word.word.strip()
            if text:
                words.append({"word": text, "start": float(word.start), "end": float(word.end)})
    return " ".join(word["word"] for word in words), words


def _align_words(reference: list[dict], hypothesis: list[dict]) -> list[tuple[str, int | None, int | None]]:
    """Return a minimum-edit alignment; each non-equal row is one WER insertion/deletion/substitution."""
    ref_words = [_normalize_word(word["word"]) for word in reference]
    hyp_words = [_normalize_word(word["word"]) for word in hypothesis]
    rows, cols = len(ref_words) + 1, len(hyp_words) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = distance[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1])
            distance[i][j] = min(substitution, distance[i - 1][j] + 1, distance[i][j - 1] + 1)

    aligned = []
    i, j = len(ref_words), len(hyp_words)
    while i or j:
        if i and j and ref_words[i - 1] == hyp_words[j - 1] and distance[i][j] == distance[i - 1][j - 1]:
            aligned.append(("equal", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            aligned.append(("replace", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            aligned.append(("delete", i - 1, None))
            i -= 1
        else:
            aligned.append(("insert", None, j - 1))
            j -= 1
    aligned.reverse()
    return aligned


def _word_error_rate(reference: list[dict], hypothesis: list[dict]) -> float:
    edits = sum(kind != "equal" for kind, _, _ in _align_words(reference, hypothesis))
    return edits / max(1, len(reference))


def _waveform_correlation(wav1: Path, wav2: Path) -> float | None:
    try:
        from scipy.io import wavfile
        _, d1 = wavfile.read(str(wav1))
        _, d2 = wavfile.read(str(wav2))
    except (OSError, ValueError):
        return None
    if d1.ndim > 1:
        d1 = d1[:, 0]
    if d2.ndim > 1:
        d2 = d2[:, 0]
    min_len = min(len(d1), len(d2))
    if min_len < 160:
        return None
    v1 = d1[:min_len].astype(np.float32)
    v2 = d2[:min_len].astype(np.float32)
    if np.std(v1) <= 1e-4 or np.std(v2) <= 1e-4:
        return None
    correlation = float(np.corrcoef(v1, v2)[0, 1])
    return correlation if np.isfinite(correlation) else None


def _per_second_wer(reference: list[dict], hypothesis: list[dict], alignment, speed: float) -> list[dict]:
    reference_counts: dict[int, int] = {}
    errors: dict[int, int] = {}
    for word in reference:
        second = max(0, int(float(word["start"]) / speed))
        reference_counts[second] = reference_counts.get(second, 0) + 1

    for kind, ref_index, hyp_index in alignment:
        if kind == "equal":
            continue
        if hyp_index is not None:
            word_time = float(hypothesis[hyp_index]["start"])
        elif ref_index is not None:
            word_time = float(reference[ref_index]["start"]) / speed
        else:
            continue
        second = max(0, int(word_time))
        errors[second] = errors.get(second, 0) + 1

    return [
        {"second": second, "word_error_rate": errors.get(second, 0) / max(1, count),
         "reference_words": count, "edit_operations": errors.get(second, 0)}
        for second, count in sorted(reference_counts.items())
    ]


def _difference_rows(reference: list[dict], hypothesis: list[dict], alignment) -> list[dict]:
    changes = []
    for kind, ref_index, hyp_index in alignment:
        if kind == "equal":
            continue
        source = reference[ref_index] if ref_index is not None else None
        upload = hypothesis[hyp_index] if hyp_index is not None else None
        changes.append({
            "type": kind,
            "original": source["word"] if source else "",
            "upload": upload["word"] if upload else "",
            "timestamps": [[round(float(upload["start"]), 3), round(float(upload["end"]), 3)]]
            if upload else [],
        })
    return changes


def run(ctx):
    if in_scenario(ctx):
        return stub_run(ctx, NAME)
    if ctx.door != "video":
        return skipped(NAME, f"unsupported transcript door: {ctx.door}")
    flags = ctx.ingest.get("flags", {})
    if not flags.get("has_audio"):
        return skipped(NAME, "video contains no audio track")

    al = ctx.read_json("alignment.json")
    if not al.get("ok") or al.get("consistent") is False:
        return skipped(NAME, "no consistent video alignment to compare speech")

    ref = ctx.read_json("source_ref.json")
    source_media = paths.library_dir() / ref["file"]
    if not source_media.is_file():
        return skipped(NAME, f"source video file not found: {ref.get('file')}")

    duration = float(ctx.ingest.get("duration_s") or 0.0)
    offset_s = float(al.get("offset_s", 0.0))
    speed = float(al.get("speed", 1.0))
    source_duration = duration * speed if duration > 0 else None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        up_wav = tmp_path / "upload.wav"
        src_wav = tmp_path / "source.wav"
        if not _extract_wav(ctx.input_path, up_wav):
            return skipped(NAME, "could not extract upload audio")
        if not _extract_wav(source_media, src_wav, start_s=offset_s, duration_s=source_duration):
            return skipped(NAME, "could not extract aligned source audio")
        try:
            t_up, up_words = _transcribe(up_wav)
            t_src, src_words = _transcribe(src_wav)
        except Exception as exc:
            return skipped(NAME, f"local transcription failed: {type(exc).__name__}: {exc}")
        corr = _waveform_correlation(up_wav, src_wav)

    if not src_words and not up_words:
        return skipped(NAME, "local transcription found no comparable words")
    if corr is None:
        return skipped(NAME, "audio correlation could not be computed")

    alignment = _align_words(src_words, up_words)
    wer = _word_error_rate(src_words, up_words)
    changes = _difference_rows(src_words, up_words, alignment)
    per_second = _per_second_wer(src_words, up_words, alignment, speed)
    diff_data = {
        "ok": True,
        "transcription_backend": "faster-whisper-local",
        "source_transcript": t_src,
        "upload_transcript": t_up,
        "source_words": src_words,
        "upload_words": up_words,
        "word_error_rate": round(wer, 4),
        "audio_correlation": round(corr, 4),
        "differences": changes,
        "per_second": per_second,
        "aligned_source_window": [round(offset_s, 3), round(offset_s + (source_duration or 0.0), 3)],
        "limitations": ["Transcript comparison checks recognized words only; it does not determine meaning "
                        "or detect every negation/semantic change."],
    }
    ctx.write_json("transcript_diff.json", diff_data)

    evidence = []
    evidence.append(Evidence(
        module=NAME, door="video", family="source_diff", channel="transcript",
        metric="word_error_rate", value=round(wer, 4), unit="WER",
        genuine_range=paths.config("genuine_ranges").get(f"{NAME}.word_error_rate", {}).get("range"),
        timestamps=[[0.0, round(duration, 3)]],
        limits=["whole-file word error rate from normalized, local, word-timestamped transcription"],
    ))
    for row in per_second:
        start = float(row["second"])
        end = min(duration, start + 1.0)
        if end <= start:
            continue
        evidence.append(Evidence(
            module=NAME, door="video", family="source_diff", channel="transcript",
            metric="word_error_rate", value=round(row["word_error_rate"], 4), unit="WER",
            genuine_range=paths.config("genuine_ranges").get(f"{NAME}.word_error_rate", {}).get("range"),
            timestamps=[[start, round(end, 3)]],
            limits=["per-second word error rate from local, word-timestamped transcription; no semantic analysis"],
        ))
    evidence.append(Evidence(
        module=NAME, door="video", family="source_diff", channel="transcript",
        metric="audio_correlation", value=round(corr, 4), unit="correlation",
        genuine_range=paths.config("genuine_ranges").get(f"{NAME}.audio_correlation", {}).get("range"),
        timestamps=[[0.0, round(duration, 3)]],
        limits=["waveform correlation over the aligned audio window"],
    ))

    return ModuleResult(
        module=NAME,
        evidence=evidence,
        artifacts={"transcript_diff": "transcript_diff.json"},
        version="faster-whisper-0.1",
    )


if __name__ == "__main__":
    run_module(NAME, run)

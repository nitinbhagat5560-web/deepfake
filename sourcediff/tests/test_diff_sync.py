from pathlib import Path
from types import SimpleNamespace

import numpy as np

from modules.diff_sync import run as sync


class FakeContext:
    door = "video"

    def __init__(self, path):
        self.case_dir = path
        self.input_path = path / "upload.mp4"
        self.ingest = {"flags": {"has_face": True, "has_audio": True}, "duration_s": 4.0}
        self.writes = {}

    def read_json(self, name):
        if name == "alignment.json":
            return {"ok": True, "consistent": True, "offset_s": 0.0, "speed": 1.0}
        if name == "source_ref.json":
            return {"source_id": "trusted", "file": "trusted.mp4"}
        raise AssertionError(name)

    def write_json(self, name, value):
        self.writes[name] = value

    def path(self, name):
        return self.case_dir / name


def test_sync_skips_when_no_paired_mouth_crops_exist(monkeypatch, tmp_path):
    source = tmp_path / "trusted.mp4"
    source.touch()
    ctx = FakeContext(tmp_path)
    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(4)]
    monkeypatch.setattr(sync, "in_scenario", lambda _ctx: False)
    monkeypatch.setattr(sync.vl, "source_path", lambda *_args: source)
    monkeypatch.setattr(sync.vl, "frame_timestamps", lambda *_args: ([0.0, 1.0, 2.0, 3.0], 4.0))
    monkeypatch.setattr(sync.vl, "iter_frames_at", lambda *_args: iter(frames))
    monkeypatch.setattr(sync, "_extract_face_and_mouth", lambda _frame: (None, None))

    result = sync.run(ctx)

    assert result.status == "skipped"
    assert "no paired mouth crops" in result.reason
    assert not ctx.writes


def test_av_correlation_is_missing_when_audio_or_motion_has_no_variation(monkeypatch, tmp_path):
    wav = Path(tmp_path / "audio.wav")
    monkeypatch.setattr(sync, "_extract_audio_envelope", lambda *_args: np.ones(4, dtype=np.float32))

    assert sync._av_sync_correlation(wav, [0.0, 1.0, 0.0, 1.0]) is None


def test_sync_emits_per_second_evidence_only_after_measuring_av_correlation(monkeypatch, tmp_path):
    source = tmp_path / "trusted.mp4"
    source.touch()
    ctx = FakeContext(tmp_path)
    upload_frames = [np.full((20, 20, 3), i * 20, dtype=np.uint8) for i in range(4)]
    source_frames = [np.full((20, 20, 3), i * 20 + 5, dtype=np.uint8) for i in range(4)]
    monkeypatch.setattr(sync, "in_scenario", lambda _ctx: False)
    monkeypatch.setattr(sync.vl, "source_path", lambda *_args: source)
    monkeypatch.setattr(sync.vl, "frame_timestamps", lambda *_args: ([0.0, 1.0, 2.0, 3.0], 4.0))
    monkeypatch.setattr(sync.vl, "iter_frames_at",
                        lambda path, _times: iter(upload_frames if path == ctx.input_path else source_frames))
    monkeypatch.setattr(sync, "_extract_face_and_mouth",
                        lambda frame: (frame[:10, :, :], frame[10:, :, :]))
    monkeypatch.setattr(sync, "_sharpness", lambda crop: float(np.mean(crop)) + 1.0)
    monkeypatch.setattr(sync, "_av_sync_correlation", lambda *_args: 0.75)
    monkeypatch.setattr(sync.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    real_temp_dir = sync.tempfile.TemporaryDirectory

    class TempDir(real_temp_dir):
        def __enter__(self):
            path = super().__enter__()
            (Path(path) / "up_audio.wav").write_bytes(b"RIFF" + b"\x00" * 50)
            return path

    monkeypatch.setattr(sync.tempfile, "TemporaryDirectory", TempDir)

    result = sync.run(ctx)

    assert result.status == "ok"
    assert len(ctx.writes["sync_diff.json"]["per_second"]) == 4
    assert len(result.evidence) == 9
    assert all(row.timestamps == [[float(i), float(i + 1)]] for i, row in enumerate(result.evidence[:8:2]))
    assert result.evidence[-1].metric == "av_sync_correlation"

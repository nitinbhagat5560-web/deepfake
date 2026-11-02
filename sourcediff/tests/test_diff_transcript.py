from types import SimpleNamespace

from modules.diff_transcript import run as transcript


def test_transcription_is_local_and_returns_word_timestamps(monkeypatch, tmp_path):
    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert path == str(tmp_path / "audio.wav")
            assert kwargs == {"word_timestamps": True, "vad_filter": True}
            words = [
                SimpleNamespace(word=" hello", start=0.1, end=0.4),
                SimpleNamespace(word=" world", start=0.5, end=0.9),
            ]
            return [SimpleNamespace(words=words)], SimpleNamespace(language="en")

    monkeypatch.setenv("SOURCEDIFF_WHISPER_MODEL", "tiny")
    monkeypatch.setattr(transcript, "_load_whisper_model", lambda model_name: FakeModel())

    text, words = transcript._transcribe(tmp_path / "audio.wav")

    assert text == "hello world"
    assert words == [
        {"word": "hello", "start": 0.1, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
    ]


def test_word_error_rate_uses_edit_distance_and_per_second_locations():
    reference = [{"word": "yes", "start": 0.2, "end": 0.4},
                 {"word": "we", "start": 1.1, "end": 1.3},
                 {"word": "can", "start": 1.4, "end": 1.6}]
    hypothesis = [{"word": "no", "start": 0.3, "end": 0.5},
                  {"word": "we", "start": 1.1, "end": 1.3},
                  {"word": "can", "start": 1.4, "end": 1.6},
                  {"word": "now", "start": 2.0, "end": 2.3}]

    alignment = transcript._align_words(reference, hypothesis)
    assert transcript._word_error_rate(reference, hypothesis) == 2 / 3
    assert transcript._difference_rows(reference, hypothesis, alignment) == [
        {"type": "replace", "original": "yes", "upload": "no", "timestamps": [[0.3, 0.5]]},
        {"type": "insert", "original": "", "upload": "now", "timestamps": [[2.0, 2.3]]},
    ]
    assert transcript._per_second_wer(reference, hypothesis, alignment, speed=1.0) == [
        {"second": 0, "word_error_rate": 1.0, "reference_words": 1, "edit_operations": 1},
        {"second": 1, "word_error_rate": 0.0, "reference_words": 2, "edit_operations": 0},
    ]


def test_transcript_configures_no_remote_recognition_service():
    import inspect

    source = inspect.getsource(transcript._transcribe)
    assert "recognize_google" not in source
    assert "word_timestamps=True" in source


def test_normalized_transcript_comparison_ignores_case_and_punctuation():
    reference = [{"word": "Hello!", "start": 0.1, "end": 0.4}]
    hypothesis = [{"word": "hello", "start": 0.1, "end": 0.4}]

    alignment = transcript._align_words(reference, hypothesis)
    assert alignment == [("equal", 0, 0)]
    assert transcript._word_error_rate(reference, hypothesis) == 0.0

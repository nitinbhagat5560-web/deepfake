from sourcediff.path2_explain import build_path2_explanation
from sourcediff.genai_reporter import report_path2


def test_face_explanation_reports_verified_metrics_without_claiming_calibration():
    result = {
        "model": "yermandy/GenD_PE_L",
        "mode": "Off",
        "metrics": {
            "num_frames": 1,
            "num_faces": 2,
            "avg_p_fake": 0.004205,
            "median_p_fake": 0.004100,
            "avg_p_ai": None,
        },
    }
    explanation = build_path2_explanation(result, "image")

    assert explanation["calibration_status"] == "uncalibrated"
    assert explanation["evidence"]["face_detections"] == 2
    assert explanation["evidence"]["average_face_output"] == 0.004205
    assert any("face" in item.lower() for item in explanation["observations"])
    assert any("not calibrated probabilities" in item for item in explanation["limitations"])


def test_no_face_and_smaug_explanations_are_distinct():
    no_face = build_path2_explanation({
        "model": "yermandy/GenD_CLIP_L_14",
        "mode": "Off",
        "metrics": {"num_frames": 1, "num_faces": 0},
    }, "image")
    smaug = build_path2_explanation({
        "model": "Smaug",
        "mode": "Smaug",
        "metrics": {"num_frames": 1, "num_faces": 0, "avg_p_ai": 0.001501},
    }, "image")

    assert "No face crop" in no_face["summary"]
    assert "complete-frame analysis" in smaug["summary"]
    assert smaug["evidence"]["average_complete_media_output"] == 0.001501
    assert no_face["decision_class"] == "inconclusive"
    assert no_face["calibration_status"] == "uncalibrated"


def test_reporter_can_be_disabled_without_losing_template_explanation(monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_DISABLE_GENERATIVE_REPORTER", "1")
    result = {
        "model": "Smaug",
        "mode": "Smaug",
        "metrics": {"num_frames": 1, "num_faces": 0, "avg_p_ai": 0.001501},
    }
    explanation = build_path2_explanation(result, "image")
    report = report_path2(explanation, result, "image", "path2_output.png")

    assert report["enabled"] is False
    assert report["status"] == "template_only"
    assert report["narrative"] is None

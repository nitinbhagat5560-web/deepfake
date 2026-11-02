from sourcediff import paths
from sourcediff.report import build_change_explanation


def test_sync_report_accepts_null_av_correlation(tmp_path):
    paths.write_json(tmp_path / "sync_diff.json", {
        "av_sync_correlation": None,
        "av_sync_status": "not_measured",
        "per_second": [
            {"second": 0, "mouth_sharpness_dev": 0.01, "mouth_motion_diff": 3.16},
            {"second": 1, "mouth_sharpness_dev": 1.026, "mouth_motion_diff": 5.67},
        ],
    })

    explanation = build_change_explanation(
        tmp_path,
        {"verdict": "video_incomplete"},
        {"door": "video"},
    )

    assert explanation["has_changes"] is False

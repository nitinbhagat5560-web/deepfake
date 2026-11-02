"""Evidence-first plain-language explanations for Path 2 detector results.

This module never creates or changes a detector verdict. It only translates
validated metrics returned by Path 2 and clearly marks model scores as
uncalibrated until deployment thresholds are established.
"""
from __future__ import annotations

from typing import Any
import json
from pathlib import Path


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_path2_explanation(result: dict, media_type: str) -> dict:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    model = str(result.get("model") or "Unknown model")
    mode = str(result.get("mode") or "Off")

    frames = _score(metrics.get("num_frames"))
    faces = _score(metrics.get("num_faces"))
    avg_fake = _score(metrics.get("avg_p_fake"))
    med_fake = _score(metrics.get("median_p_fake"))
    avg_ai = _score(metrics.get("avg_p_ai"))
    calibration_path = Path(__file__).resolve().parents[1] / "reports" / "path2_calibration.json"
    calibration = {}
    if calibration_path.is_file():
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            calibration = {}

    observations: list[str] = []
    if frames is not None:
        observations.append(
            f"The model analyzed {int(frames)} {'frame' if int(frames) == 1 else 'frames'}."
        )
    if faces is not None:
        observations.append(
            f"{int(faces)} face {'detection' if int(faces) == 1 else 'detections'} were produced."
            if int(faces) == 0
            else f"{int(faces)} face {'crop' if int(faces) == 1 else 'crops'} were scored."
        )
    if avg_fake is not None:
        observations.append(
            f"The average face detector output was {avg_fake:.6f}"
            + (f", with a median of {med_fake:.6f}." if med_fake is not None else ".")
        )
    if avg_ai is not None:
        observations.append(
            f"The average complete-{media_type} model output was {avg_ai:.6f}."
        )

    if mode == "Smaug":
        reason_text = (
            "Smaug was used for complete-frame analysis. Its output describes image-level "
            "AI-generation signals, not a localized edit made with a specific tool."
        )
    elif avg_fake is not None and faces and faces > 0:
        reason_text = (
            "The GenD backend scored detected face crops. Higher or lower aggregate output "
            "reflects model similarity patterns, but does not identify the exact changed pixels."
        )
    elif faces == 0:
        reason_text = (
            "No face crop was detected. This path produced no face-level evidence; any whole-frame "
            "output should be interpreted independently."
        )
    else:
        reason_text = (
            "The Path 2 backend returned aggregate model evidence. It does not yet provide a "
            "calibrated authenticity result."
        )

    image_calibration = (
        calibration.get("model") == "yermandy/GenD_PE_L"
        and media_type in ("image", "video")
        and int(calibration.get("samples") or 0) >= 30
    )
    if image_calibration:
        face_cfg = calibration.get("face_score") or {}
        frame_cfg = calibration.get("whole_frame_score") or {}
        face_threshold = _score(face_cfg.get("threshold_fake_if_greater_equal"))
        frame_threshold = _score(frame_cfg.get("threshold_fake_if_greater_equal"))
        if avg_fake is not None and face_threshold is not None:
            selected_name = "face-crop"
            selected_score = avg_fake
            selected_threshold = face_threshold
        elif avg_ai is not None and frame_threshold is not None:
            selected_name = "whole-frame"
            selected_score = avg_ai
            selected_threshold = frame_threshold
        else:
            selected_name = None
            selected_score = None
            selected_threshold = None
        calibrated_status = "calibrated on development images"
        limitations = [
            "This threshold is calibrated on bundled FF and CDFv2 image samples only; it is not a held-out guarantee.",
            *(["For video, the image-derived threshold is applied provisionally to the aggregate face/frame score and needs video-specific validation."] if media_type == "video" else []),
            "The annotated regions show where the model produced or localized evidence; they do not prove an exact edit.",
            "Lighting, compression, blur, makeup, camera processing, resolution, and dataset shift may affect the result.",
        ]
        if selected_score is not None and selected_threshold is not None:
            likely_manipulated = selected_score >= selected_threshold
            decision_class = "likely_manipulated" if likely_manipulated else "likely_real"
            score_label = "Likely manipulated" if likely_manipulated else "Likely real"
            headline = f"{score_label} — calibrated Path 2 development threshold"
            reason = (
                f"{selected_name} score {selected_score:.6f} "
                + (
                    f"is greater than or equal to the threshold {selected_threshold:.6f}, "
                    if likely_manipulated else
                    f"is below the threshold {selected_threshold:.6f}, "
                )
                + "so Path 2 gives this media a "
                + ("likely-manipulated" if likely_manipulated else "likely-real")
                + " tendency."
            )
            return {
                "calibration_status": (
                    calibrated_status if media_type == "image"
                    else "provisional calibrated image threshold for video"
                ),
                "decision_class": decision_class,
                "headline": headline,
                "final_verdict": score_label,
                "reason": reason,
                "summary": (
                    f"The {selected_name} score was {selected_score:.6f}; "
                    f"the development threshold is {selected_threshold:.6f}. "
                    "This indicates the model's calibrated image-side class tendency, not proof of authenticity or provenance."
                ),
                "observations": observations,
                "limitations": limitations,
                "evidence": {
                    "frames_analyzed": int(frames) if frames is not None else None,
                    "face_detections": int(faces) if faces is not None else None,
                    "average_face_output": avg_fake,
                    "median_face_output": med_fake,
                    "average_complete_media_output": avg_ai,
                    "selected_score": selected_score,
                    "selected_threshold": selected_threshold,
                    "selected_channel": selected_name,
                    "development_auc": (
                        face_cfg.get("auc") if selected_name == "face-crop"
                        else frame_cfg.get("auc")
                    ),
                    "development_accuracy": (
                        face_cfg.get("development_accuracy") if selected_name == "face-crop"
                        else frame_cfg.get("development_accuracy")
                    ),
                },
            }

    limitations = [
        "Path 2 is calibrated only on bundled static-image samples; video scores remain inconclusive.",
        "The annotated regions show where the model produced or localized evidence; they do not prove an exact edit.",
        "Lighting, compression, blur, makeup, camera processing, resolution, and model distribution shift may affect the result.",
    ]

    complete_score_missing = avg_fake is None and avg_ai is None
    insufficient_evidence = frames is None or frames < 1 or (
        media_type == "video" and faces == 0 and avg_ai is None
    )
    if complete_score_missing or insufficient_evidence or (faces == 0 and avg_fake is None and avg_ai is None):
        return {
            "calibration_status": "uncalibrated",
            "decision_class": "inconclusive",
            "headline": "Inconclusive experimental evidence",
            "summary": reason_text + " The result is inconclusive and must not be interpreted as fake or genuine.",
            "observations": observations,
            "limitations": limitations,
            "evidence": {
                "frames_analyzed": int(frames) if frames is not None else None,
                "face_detections": int(faces) if faces is not None else None,
                "average_face_output": avg_fake,
                "median_face_output": med_fake,
                "average_complete_media_output": avg_ai,
            },
        }

    return {
        "calibration_status": "uncalibrated",
        "decision_class": "experimental",
        "headline": "Experimental model evidence — no calibrated verdict",
        "summary": reason_text,
        "observations": observations,
        "limitations": limitations,
        "evidence": {
            "frames_analyzed": int(frames) if frames is not None else None,
            "face_detections": int(faces) if faces is not None else None,
            "average_face_output": avg_fake,
            "median_face_output": med_fake,
            "average_complete_media_output": avg_ai,
        },
    }

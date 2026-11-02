"""Optional grounded generative narration for SourceDiff evidence cards."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


_FORBIDDEN_CLAIMS = (
    "this is authentic",
    "this is genuine",
    "this is fake",
    "is authentic",
    "is genuine",
    "is fake",
    "the file is manipulated",
    "this file is manipulated",
    "is a deepfake",
    "is tampered",
    "likely authentic",
    "likely genuine",
    "likely fake",
    "highly likely",
)


def _numbers(text: str) -> list[float]:
    values = []
    for match in re.findall(r"(?<![\w.])(?:\d+(?:\.\d+)?)(?:%|s)?(?![\w.])", text):
        token = match[:-1] if match.endswith(("%", "s")) else match
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


def _number_is_supported(value: float, evidence_values: list[float]) -> bool:
    for expected in evidence_values:
        if value == expected:
            return True
        scale = max(1.0, abs(expected))
        if abs(value - expected) <= max(1e-6, scale * 1e-4):
            return True
    return False


def report_path2(explanation: dict, result: dict, media_type: str, artifact: str) -> dict:
    """Return a local LLM narrative only when its wording can be fact-grounded."""
    base = {
        "enabled": False,
        "model": os.getenv("SOURCEDIFF_REPORTER_MODEL", "llama3:latest"),
        "status": "template_only",
        "narrative": None,
    }
    if os.getenv("SOURCEDIFF_DISABLE_GENERATIVE_REPORTER", "").strip():
        return base

    evidence = explanation.get("evidence") or {}
    facts = {
        "media_type": media_type,
        "artifact": artifact,
        "detector_model": result.get("model") or "Unknown model",
        "analysis_mode": result.get("mode") or "Off",
        "calibration_status": explanation.get("calibration_status"),
        "decision_class": explanation.get("decision_class"),
        "evidence": evidence,
        "observations": explanation.get("observations") or [],
        "limitations": explanation.get("limitations") or [],
    }
    prompt = (
        "You are a forensic evidence reporter. Produce one JSON object only. "
        "Use this schema: {\"narrative\":\"...\"}. "
        "Explain WHAT was measured, WHERE it came from, HOW it was measured, and WHAT limits it has. "
        "Use only the supplied evidence. Do not infer manipulation, authenticity, a percentage, "
        "a probability, or a definitive verdict. Calibrated development scores may be described, "
        "but must not be called proof. "
        "Do not add numbers that are not in the evidence. "
        + json.dumps(facts, ensure_ascii=False)
    )
    try:
        host = os.getenv("SOURCEDIFF_OLLAMA_HOST", "http://127.0.0.1:11434")
        timeout_s = float(os.getenv("SOURCEDIFF_REPORTER_TIMEOUT", "60"))
        request = urllib.request.Request(
            host.rstrip("/") + "/api/generate",
            data=json.dumps({
                "model": base["model"],
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw = json.loads(payload.get("response") or "{}")
        narrative = str(raw.get("narrative") or "").strip()
        if not narrative:
            return base

        evidence_numbers: list[float] = []
        for value in evidence.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                evidence_numbers.extend(_numbers(f"{value}"))
        narrative_numbers = _numbers(narrative)
        if any(not _number_is_supported(value, evidence_numbers) for value in narrative_numbers):
            return base | {"status": "template_only_unsupported_numbers"}

        lowered = narrative.lower()
        if any(claim in lowered for claim in _FORBIDDEN_CLAIMS):
            return base | {"status": "template_only_verdict_language"}

        return {
            "enabled": True,
            "model": base["model"],
            "status": "generated",
            "narrative": narrative,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
            ValueError, OSError):
        return base | {"status": "template_only_reporter_unavailable"}

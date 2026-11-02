"""THE SHARED FORMAT. Freeze this file before two people build modules against it.

Every module (a folder under modules/) reads the case folder and writes ONE result file shaped like
ModuleResult below. The runner and the verdict engine only ever read that file, so modules never call
each other and a module can be rewritten without touching the rest.

  Evidence      one measured number: what was measured, its value and unit, where in the file,
                the genuine range it will be judged against, and plain-language limits.
  Gate          the answer of a library search: did we find the original, and how much of the UPLOAD
                does it cover (coverage is measured on the upload, not on the source).
  ModuleResult  status ("ok" | "skipped" | "error"), plus evidence rows and/or a gate.

A module NEVER decides a verdict. `supports` is the module's own opinion and the engine ignores it:
the engine re-derives "outside the genuine range" from config/genuine_ranges.json.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

DOORS = ("image", "video")
STATUSES = ("ok", "skipped", "error")
FAMILIES = ("source_diff",)                       # a later Path B adds: self_diff, detector, claim
CHANNELS = ("picture", "transcript", "sync", "audio", "timeline")
SUPPORTS = ("neutral", "synthetic", "genuine")
GATE_LIBRARIES = ("trusted", "fake")              # "fake" is reserved for a later Gate 2
ADMITTED_PROVENANCE = ("path1_proven", "human_confirmed")


class ContractError(ValueError):
    """A result does not follow the shared format."""


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _ranges(label: str, rs) -> None:
    if not isinstance(rs, list):
        raise ContractError(f"{label} must be a list of [start, end] pairs")
    for r in rs:
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and _num(r[0]) and _num(r[1]) and r[0] < r[1]):
            raise ContractError(f"{label}: bad range {r!r} (need [start, end] with start < end)")


def outside_range(value: float, rng) -> bool:
    """True when value lies outside the closed interval [lo, hi]."""
    lo, hi = rng
    return value < lo or value > hi


def _from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Evidence:
    module: str
    door: str
    family: str
    channel: str | None
    metric: str
    value: float
    unit: str = ""
    genuine_range: list | None = None          # shown for the reader; the engine uses config, not this
    timestamps: list = field(default_factory=list)   # [[start_s, end_s], ...] on the UPLOAD; [] = whole file
    supports: str = "neutral"                  # the module's opinion; IGNORED by the engine
    confidence: float = 0.0
    limits: list = field(default_factory=list)  # plain-language caveats shown on the card
    stub: bool = False                         # True = hand-written scenario evidence, not a real measurement

    def validate(self) -> "Evidence":
        if not self.module or not isinstance(self.module, str):
            raise ContractError("evidence.module is required")
        if self.door not in DOORS:
            raise ContractError(f"evidence.door must be one of {DOORS}, got {self.door!r}")
        if self.family not in FAMILIES:
            raise ContractError(f"evidence.family must be one of {FAMILIES}, got {self.family!r}")
        if self.family == "source_diff" and self.channel not in CHANNELS:
            raise ContractError(f"a source_diff row needs channel in {CHANNELS}, got {self.channel!r}")
        if not self.metric or not isinstance(self.metric, str):
            raise ContractError("evidence.metric is required")
        if not _num(self.value):
            raise ContractError(f"evidence.value must be a finite number, got {self.value!r}")
        if self.genuine_range is not None:
            gr = self.genuine_range
            if not (isinstance(gr, (list, tuple)) and len(gr) == 2 and _num(gr[0]) and _num(gr[1]) and gr[0] <= gr[1]):
                raise ContractError(f"evidence.genuine_range must be [lo, hi], got {gr!r}")
        _ranges("evidence.timestamps", self.timestamps)
        if self.supports not in SUPPORTS:
            raise ContractError(f"evidence.supports must be one of {SUPPORTS}")
        if not (_num(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ContractError("evidence.confidence must be between 0 and 1")
        if not (isinstance(self.limits, list) and all(isinstance(s, str) for s in self.limits)):
            raise ContractError("evidence.limits must be a list of strings")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return _from_dict(cls, d)


@dataclass
class Gate:
    library: str                                # "trusted"
    matched: bool
    source_id: str | None = None
    coverage_of_upload: float = 0.0             # fraction of the UPLOAD that matched the source (0..1)
    matched_ranges: list = field(default_factory=list)    # upload seconds that matched
    unmatched_ranges: list = field(default_factory=list)  # upload seconds that did not
    votes: int = 0
    threshold: float | None = None
    index_version: str = ""
    provenance: str | None = None               # only meaningful for a later Gate 2
    exact_match: bool = False                   # SHA-256 identity with a verified trusted source

    def validate(self) -> "Gate":
        if self.library not in GATE_LIBRARIES:
            raise ContractError(f"gate.library must be one of {GATE_LIBRARIES}")
        if self.matched and not self.source_id:
            raise ContractError("a matched gate must name its source_id")
        if not (_num(self.coverage_of_upload) and 0.0 <= self.coverage_of_upload <= 1.0):
            raise ContractError("gate.coverage_of_upload must be between 0 and 1")
        _ranges("gate.matched_ranges", self.matched_ranges)
        _ranges("gate.unmatched_ranges", self.unmatched_ranges)
        if not isinstance(self.exact_match, bool):
            raise ContractError("gate.exact_match must be a boolean")
        if self.exact_match and (not self.matched or self.coverage_of_upload != 1.0):
            raise ContractError("an exact match must be matched with full upload coverage")
        if self.library == "fake" and self.matched and self.provenance not in ADMITTED_PROVENANCE:
            raise ContractError(f"a matched fake needs provenance in {ADMITTED_PROVENANCE}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Gate":
        return _from_dict(cls, d)


@dataclass
class ModuleResult:
    module: str
    status: str = "ok"                          # ok | skipped | error. Anything but ok means "did not run"
    reason: str = ""
    stub: bool = False
    version: str = ""
    evidence: list = field(default_factory=list)
    gate: Gate | None = None
    artifacts: dict = field(default_factory=dict)   # {"map": "diff_picture_map.png"}: files inside the case folder
    input_sha256: str = ""                      # stamped by run_module(); a result for another file never counts
    runtime_s: float = 0.0

    def validate(self) -> "ModuleResult":
        if not self.module:
            raise ContractError("result.module is required")
        if self.status not in STATUSES:
            raise ContractError(f"result.status must be one of {STATUSES}")
        if self.status != "ok" and (self.evidence or self.gate):
            raise ContractError("only an 'ok' result may carry evidence or a gate")
        for e in self.evidence:
            e.validate()
        if self.gate is not None:
            self.gate.validate()
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gate"] = self.gate.to_dict() if self.gate else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModuleResult":
        d = dict(d)
        d["evidence"] = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        g = d.get("gate")
        d["gate"] = Gate.from_dict(g) if g else None
        return _from_dict(cls, d)

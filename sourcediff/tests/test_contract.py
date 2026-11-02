"""The shared format: what it accepts and what it refuses."""
import math

import pytest

from sourcediff.contract import (ADMITTED_PROVENANCE, ContractError, Evidence, Gate, ModuleResult, outside_range)


def ev(**kw):
    base = dict(module="diff_picture", door="image", family="source_diff", channel="picture",
                metric="block_diff_max", value=1.5)
    base.update(kw)
    return Evidence(**base)


def test_good_evidence_validates_and_round_trips():
    e = ev(unit="grey levels", genuine_range=[0, 2.6], timestamps=[[1, 2]], limits=["x"], confidence=0.5).validate()
    assert Evidence.from_dict(e.to_dict()) == e


@pytest.mark.parametrize("bad", [dict(value=True), dict(value=float("nan")), dict(value=math.inf), dict(value="1.5"),
                                 dict(door="film"), dict(family="detector"), dict(channel="smell"), dict(channel=None),
                                 dict(confidence=1.5), dict(confidence=-0.1), dict(supports="maybe"),
                                 dict(timestamps=[[3, 2]]), dict(timestamps=[[1]]), dict(timestamps=[1, 2]),
                                 dict(limits=[1]), dict(genuine_range=[2, 1]), dict(genuine_range=[1]),
                                 dict(metric=""), dict(module="")])
def test_bad_evidence_is_refused(bad):
    with pytest.raises(ContractError):
        ev(**bad).validate()


def test_gate_rules():
    Gate(library="trusted", matched=True, source_id="s", coverage_of_upload=1.0).validate()
    Gate(library="trusted", matched=False).validate()
    with pytest.raises(ContractError):
        Gate(library="trusted", matched=True).validate()                       # matched but no source
    with pytest.raises(ContractError):
        Gate(library="trusted", matched=False, coverage_of_upload=1.2).validate()
    with pytest.raises(ContractError):
        Gate(library="museum", matched=False).validate()
    with pytest.raises(ContractError):
        Gate(library="trusted", matched=False, unmatched_ranges=[[5, 5]]).validate()
    with pytest.raises(ContractError):
        Gate(library="trusted", matched=False, exact_match=True).validate()
    with pytest.raises(ContractError):                                          # a matched fake must say how it got in
        Gate(library="fake", matched=True, source_id="f", provenance="vibes").validate()
    Gate(library="fake", matched=True, source_id="f", provenance=ADMITTED_PROVENANCE[0]).validate()


def test_only_ok_results_carry_evidence():
    ModuleResult(module="m", status="ok", evidence=[ev()]).validate()
    for status in ("skipped", "error"):
        with pytest.raises(ContractError):
            ModuleResult(module="m", status=status, evidence=[ev()]).validate()
        with pytest.raises(ContractError):
            ModuleResult(module="m", status=status, gate=Gate(library="trusted", matched=False)).validate()
    with pytest.raises(ContractError):
        ModuleResult(module="m", status="maybe").validate()


def test_module_result_round_trip_and_unknown_keys():
    r = ModuleResult(module="m", evidence=[ev()], gate=Gate(library="trusted", matched=True, source_id="s",
                                                           coverage_of_upload=0.5, matched_ranges=[[0, 5]]),
                     artifacts={"map": "a.png"}, input_sha256="x", runtime_s=1.0).validate()
    d = r.to_dict()
    d["future_field"] = "ignored"
    back = ModuleResult.from_dict(d)
    assert back.evidence[0] == r.evidence[0] and back.gate == r.gate and back.artifacts == r.artifacts


def test_outside_range_edges_are_inside():
    assert not outside_range(0, [0, 1]) and not outside_range(1, [0, 1])
    assert outside_range(-0.001, [0, 1]) and outside_range(1.001, [0, 1])

"""The Golden Rule, attacked with random cases.

For each random case the verdict engine's answer is compared with a second, deliberately plain
implementation of the rule (the "oracle" below). They are written separately on purpose: if one is changed
and the other is not, this test fails.

  1. Image 'authentic' only when all Path 1 conditions hold; non-exact video uses its own safe outcome labels.
  2. Everything else is never 'authentic'.
  3. Junk from modules that are not verdict roles, results computed on another file and unreadable results
     never change the answer.
"""
import random

import pytest
from helpers import (CHANNEL_MODULES, GATE_MODULE, GOOD_ALIGNMENT, SHA, aligner_result, evidence_result, gate_result,
                     inside, make_case, outside, range_of)

from sourcediff import paths, verdict
from sourcediff.contract import Evidence, ModuleResult

N_CASES = 500
DOORS = ["image", "video"]
CH_ORDER = ["alignment", "picture", "transcript", "sync"]
CH_VERDICT = {"alignment": "edited_spliced", "picture": "edited_picture", "transcript": "edited_audio", "sync": "lips_resynthesized"}
DOOR_CHANNELS = {"image": ["alignment", "picture"], "video": ["alignment", "picture", "transcript", "sync"]}
NEEDS = {"picture": [], "transcript": ["has_audio"], "sync": ["has_face", "has_audio"]}
CH_DOORS = {"picture": ("image", "video"), "transcript": ("video",), "sync": ("video",)}


def make_random_case(rng, root):
    door = rng.choice(DOORS)
    flags = {"has_face": rng.random() < 0.6, "has_audio": rng.random() < 0.5}
    duration = None if door == "image" else rng.choice([30.0, 60.0, 120.0])
    # 45% of cases start as a perfect Authentic case with at most one fault; the rest are fully random
    good = rng.random() < 0.45
    mut = rng.choice([None] * 3 + ["gate", "chan_missing", "chan_outside", "align_jump", "coverage", "probe"]) if good else None
    victim = rng.choice(CH_ORDER[1:])
    probe_ok = (mut != "probe") if good else rng.random() > 0.05
    spec = {"door": door, "flags": flags, "duration": duration, "probe_ok": probe_ok}

    results = []
    # ---- gate 1
    g_state = ("nomatch" if mut == "gate" else "match") if good else \
        rng.choice(["missing", "error", "skipped", "nomatch", "match", "match", "match", "match", "stale"])
    g_cov, g_unmatched = 1.0, []
    if g_state in ("match", "stale"):
        g_cov = (0.85 if mut == "coverage" else 1.0) if good else rng.choice([1.0, 1.0, 1.0, 0.97, 0.85, 0.5])
        if door != "image" and g_cov < 1.0:
            d = duration
            g_unmatched = [(d / 2, d / 2 + 5)] if good else \
                rng.choice([[], [(0, 0.5)], [(d - 0.5, d)], [(d / 2, d / 2 + 0.5)], [(0, 3.0)], [(d / 2, d)]])
        g = gate_result(door, True, g_cov, g_unmatched, [(0, duration)] if duration else (),
                        sha=SHA if g_state == "match" else "b" * 64)
        results.append(g)
    elif g_state == "nomatch":
        results.append(gate_result(door, matched=False))
    elif g_state in ("error", "skipped"):
        results.append(gate_result(door, status=g_state))
    spec["gate"] = {"state": g_state, "coverage": g_cov, "unmatched": g_unmatched}

    # ---- aligner + alignment.json
    a_state = "ok" if good else rng.choice(["ok", "ok", "ok", "error", "missing"])
    if a_state != "missing":
        results.append(aligner_result(status=a_state))
    jump = {"ok": True, "consistent": True, "jumps": [{"at_s": 5, "size_s": 2}]}
    al = (jump if mut == "align_jump" else GOOD_ALIGNMENT) if good else \
        rng.choice([GOOD_ALIGNMENT, GOOD_ALIGNMENT, GOOD_ALIGNMENT, {"ok": True, "consistent": False, "jumps": []},
                    jump, {"ok": False}, None])
    spec["aligner"], spec["alignment"] = a_state, al

    # ---- channels
    spec["channels"] = {}
    raw = {}
    for ch in CH_ORDER[1:]:
        mod, metrics = CHANNEL_MODULES[ch]
        st = ("missing" if (mut == "chan_missing" and ch == victim) else "ok") if good else \
            rng.choice(["missing", "error", "skipped", "ok", "ok", "ok", "ok", "stale"])
        rows = {}
        if st in ("ok", "stale"):
            for m in rng.sample(metrics, k=rng.randint(1, len(metrics))):
                bad = (mut == "chan_outside" and ch == victim) if good else rng.random() < 0.25
                rows[m] = outside(mod, m) if bad else inside(mod, m)
            if not good and rng.random() < 0.1:
                rows["made_up_metric"] = 1.0                       # no configured range: cannot be judged
            results.append(evidence_result(ch, door if door in CH_DOORS[ch] else "video", rows,
                                           sha=SHA if st == "ok" else "b" * 64))
        elif st in ("error", "skipped"):
            results.append(evidence_result(ch, door if door in CH_DOORS[ch] else "video", status=st))
        elif st == "missing" and rng.random() < 0.2:
            raw[mod] = "{ not json"
        spec["channels"][ch] = {"state": st, "rows": rows, "raw": mod in raw}

    # ---- noise that must never matter
    if rng.random() < 0.5:
        results.append(evidence_result("picture", door, {"block_diff_max": 999.0}, module="evil_module"))
    if rng.random() < 0.3:
        results.append(evidence_result("picture", door, {"block_diff_max": 999.0}, module="b1_detector"))
    case = make_case(root, door, flags, duration, results, al, probe_ok=probe_ok, raw_results=raw)
    return spec, case


def oracle(spec):
    """The rule, written as plainly as possible."""
    door, flags, dur = spec["door"], spec["flags"], spec["duration"]
    if not spec["probe_ok"]:
        return "cant_verify"
    g = spec["gate"]
    if g["state"] != "match":
        return "cant_verify"
    # coverage of the upload: at least 90% matched and unmatched pieces are slivers (<= 1 s) at an edge
    covered = g["coverage"] >= 0.9 and all(
        (b - a) <= 1.0 and (a <= 1.0 or (dur is not None and b >= dur - 1.0)) for a, b in g["unmatched"])
    if not covered and door == "image":
        return "not_authentic_partial"
    states = {}
    for ch in DOOR_CHANNELS[door]:
        if ch == "alignment":
            al = spec["alignment"]
            if spec["aligner"] != "ok" or al is None or not al.get("ok"):
                states[ch] = "not_run"
            elif al.get("consistent") is False or al.get("jumps"):
                states[ch] = "differs"
            else:
                states[ch] = "agrees"
            continue
        applicable = door in CH_DOORS[ch] and all(flags.get(n) for n in NEEDS[ch])
        if not applicable:
            continue
        c = spec["channels"][ch]
        if c["state"] != "ok" or not c["rows"]:
            states[ch] = "not_run"
            continue
        mod = CHANNEL_MODULES[ch][0]
        judged = []
        for metric, value in c["rows"].items():
            if metric == "made_up_metric":
                judged.append(None)
            else:
                lo, hi = range_of(mod, metric)
                judged.append(value < lo or value > hi)
        if any(j is None for j in judged):
            states[ch] = "not_run"
        else:
            states[ch] = "differs" if any(judged) else "agrees"
    differing = [c for c in CH_ORDER if states.get(c) == "differs"]
    if differing:
        return "video_edited" if door == "video" else CH_VERDICT[differing[0]]
    if door == "video" and not covered:
        return "video_incomplete"
    if any(s == "not_run" for s in states.values()):
        return "video_incomplete" if door == "video" else "cant_verify"
    return "video_matched_no_edit" if door == "video" else "authentic"


def test_engine_agrees_with_the_oracle_on_random_cases(cfg, tmp_path):
    rng = random.Random(20261019)
    seen = {}
    for i in range(N_CASES):
        spec, case = make_random_case(rng, tmp_path)
        got = verdict.decide(case)["verdict"]
        want = oracle(spec)
        assert got == want, f"case {i}: engine said {got}, oracle said {want}\n{spec}"
        seen[got] = seen.get(got, 0) + 1
    # the generator must actually exercise the interesting outcomes, or the test proves nothing
    assert seen.get("authentic", 0) >= 15, seen
    for v in ("cant_verify", "not_authentic_partial", "edited_picture", "edited_spliced",
              "video_incomplete", "video_edited", "video_matched_no_edit"):
        assert seen.get(v, 0) >= 5, seen


def test_authentic_never_without_all_its_conditions(cfg, tmp_path):
    """One-sided check on the same cases, written against the raw facts and not against the oracle."""
    rng = random.Random(77)
    n_auth = 0
    for _ in range(N_CASES):
        spec, case = make_random_case(rng, tmp_path)
        v = verdict.decide(case)
        if v["verdict"] != "authentic":
            continue
        n_auth += 1
        assert spec["probe_ok"] and spec["gate"]["state"] == "match"
        assert spec["gate"]["coverage"] >= 0.9
        assert spec["aligner"] == "ok" and spec["alignment"] and spec["alignment"]["ok"] and not spec["alignment"].get("jumps")
        assert v["gate1"]["matched"] and v["gate1_full_coverage"]
        for ch, st in v["channels"].items():
            assert st in ("agrees", "na"), (ch, st)
        for r in v["evidence"]:
            if r["counted"]:
                assert r["outside"] is False
    assert n_auth > 0


@pytest.mark.parametrize("bad_status", ["error", "skipped", "missing"])
def test_a_module_that_dies_never_helps(cfg, tmp_path, bad_status):
    """Take a perfect Authentic case and break each module in turn: the answer may only get more cautious."""
    door, flags, dur = "video", {}, 60.0
    perfect = [gate_result(door, matched_ranges=[(0, dur)]), aligner_result()] + \
              [evidence_result("picture", door)]
    assert verdict.decide(make_case(tmp_path, door, flags, dur, perfect, GOOD_ALIGNMENT))["verdict"] == \
        "video_matched_no_edit"
    for victim in [r.module for r in perfect]:
        results = [r for r in perfect if r.module != victim]
        if bad_status != "missing":
            kind = {"g1_visual_match": gate_result(door, status=bad_status)}.get(victim) or \
                   (aligner_result(status=bad_status) if victim == "aligner" else
                    ModuleResult(module=victim, status=bad_status, reason="test", input_sha256=SHA))
            results.append(kind)
        v = verdict.decide(make_case(tmp_path, door, flags, dur, results, GOOD_ALIGNMENT))
        assert v["verdict"] not in ("authentic", "video_matched_no_edit"), f"{victim} {bad_status}"

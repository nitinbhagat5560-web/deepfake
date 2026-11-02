"""Measure the image door's numbers instead of guessing them.

    python tools/calibrate_image.py                  # measure and print, change nothing
    python tools/calibrate_image.py --write          # also write config/genuine_ranges.json and config/gates.json

It uses the SAME code the pipeline uses (picturediff.align and picturediff.compare), so what is measured
is what runs. About a minute on two cores.

WHAT IT MEASURES
  1. GATE THRESHOLD    similarity of genuine re-uploads to their own original (positives) versus similarity of
                       other pictures to the library (hard negatives). The threshold goes in the gap.
  2. GENUINE RANGE     block_diff_max, detail_dev_max and block_chroma_diff_max on genuine re-uploads
                       (JPEG quality x shrink) of the CALIBRATION pictures. The range's upper edge =
                       the largest genuine value seen x margin.
  3. HONEST TEST       the range is then applied to HELD-OUT pictures that the calibration never saw:
                       how many genuine re-uploads are wrongly called edited, and how many edits slip through
                       and would be called Authentic. N is always printed.

MARGIN  The range's upper edge is the largest genuine value seen, times a margin (default 1.0 = no slack).
        Slack lowers false "edited" calls on genuine copies, but every bit of it lets more edits through, and an
        edit that passes is the dangerous error (a false "Authentic"), while a genuine copy called edited is the safe
        one. The table printed below shows the price of slack. The default was chosen on that principle AFTER seeing
        the held-out table, so the held-out numbers are no longer a clean test: for a headline claim, use new pictures.

Reading the result: "0 of N edits passed" does NOT mean the true miss rate is zero. At 95% confidence it
only bounds it below about 3/N (the rule of three). Say "0 of N", say N, say the bound.

These are REHEARSAL numbers: sample photos from scikit-image, one kind of re-upload (shrink + JPEG), a
stand-in embedder. At the hackathon, measure again on your real media.
"""
import argparse
import json
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2
import numpy as np

import _variants as V
from modules._common import imagelib as il, picturediff as pd
from sourcediff import paths

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)

POOL = ["astronaut", "camera", "coins", "chelsea", "coffee", "rocket", "moon", "immunohistochemistry",
        "hubble_deep_field", "retina", "cell", "page", "clock", "text", "microaneurysms"]
LIBRARY = ["astronaut", "camera", "coins", "chelsea", "coffee", "rocket", "moon", "immunohistochemistry"]
HOLDOUT = ["astronaut", "coffee", "retina", "clock", "coins"]       # never used to set a range
QUALITIES = [30, 45, 60, 75, 90]
SCALES = [0.5, 0.65, 0.8, 1.0]
EDIT_REUPLOAD = (60, 0.8)                                          # edits arrive re-uploaded like everything else
MARGINS = [1.0, 1.25, 1.5, 2.0]
DEFAULT_MARGIN = 1.0
METRICS = ("block_diff_max", "detail_dev_max", "block_chroma_diff_max")


def rule_of_three(n: int) -> float:
    """If 0 of n happened, the true rate is below about 3/n with 95% confidence."""
    return min(1.0, 3.0 / n) if n else 1.0


def measure(job):
    name, kind, up_bgr, src_bgr = job
    gate = float(il.embed_array(up_bgr) @ il.embed_array(src_bgr))
    ug, sg = il.gray(up_bgr), il.gray(src_bgr)
    al = pd.align(ug, sg)
    row = {"name": name, "kind": kind, "gate": gate, "aligned": bool(al["ok"]), "compared": False}
    if al["ok"]:
        # Colour compare so Lab chroma is measured the same way the pipeline runs it.
        res = pd.compare(up_bgr, src_bgr, al["H"])
        if res["ok"]:
            row.update(compared=True, ecc_cc=res["ecc_cc"],
                       **{m: res[m] for m in METRICS})
    return row


def jobs(imgs):
    rng = np.random.default_rng(7)
    order = list(imgs)
    for i, n in enumerate(order):
        img = imgs[n]
        for q in QUALITIES:
            for s in SCALES:
                yield (n, "genuine", V.reupload(img, q, s), img)
        donor = imgs[order[(i + 3) % len(order)]]
        for kind, e in V.random_edits(img, donor, rng):
            yield (n, kind, V.reupload(e, *EDIT_REUPLOAD), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the measured numbers into config/")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="upper range edge = largest genuine value seen x this margin (default 1.0, see MARGIN below)")
    args = ap.parse_args()

    imgs = {n: V.sample(n) for n in POOL}
    print(f"pictures: {len(POOL)} | library: {len(LIBRARY)} | held out: {HOLDOUT}")

    # ---------------------------------------------------------------- 1. gate threshold
    lib_vec = {n: il.embed_array(imgs[n]) for n in LIBRARY}
    pos = [float(il.embed_array(V.reupload(imgs[n], q, s)) @ lib_vec[n])
           for n in LIBRARY for q in QUALITIES for s in SCALES]
    neg = [max(float(il.embed_array(imgs[n]) @ v) for v in lib_vec.values()) for n in POOL if n not in LIBRARY]
    neg += [float(lib_vec[a] @ lib_vec[b]) for a in LIBRARY for b in LIBRARY if a != b]
    pos_min, neg_max = min(pos), max(neg)
    separable = pos_min > neg_max
    thr = round((pos_min + neg_max) / 2, 4) if separable else None
    print(f"\n[gate]  genuine re-uploads n={len(pos)}: lowest similarity {pos_min:.4f}")
    print(f"[gate]  hard negatives     n={len(neg)}: highest similarity {neg_max:.4f}")
    print("[gate]  " + (f"separable, threshold = {thr}" if separable else "NOT SEPARABLE: do not trust this embedder"))

    # ---------------------------------------------------------------- 2. measure every variant
    work = list(jobs(imgs))
    print(f"\nmeasuring {len(work)} pictures on {os.cpu_count()} core(s) ...", flush=True)
    with Pool(os.cpu_count()) as p:
        rows = p.map(measure, work, chunksize=4)
    gen = [r for r in rows if r["kind"] == "genuine"]
    edits = [r for r in rows if r["kind"] != "genuine"]
    cal_gen = [r for r in gen if r["name"] not in HOLDOUT and r["compared"]]
    hold_gen = [r for r in gen if r["name"] in HOLDOUT]
    hold_edits = [r for r in edits if r["name"] in HOLDOUT]
    print(f"genuine re-uploads: {len(gen)} | could not be aligned or compared: {sum(1 for r in gen if not r['compared'])}"
          f" (these pictures have too little texture; the engine answers Can't Verify for them, never Authentic)")

    maxima = {m: max(r[m] for r in cal_gen) for m in METRICS}
    print(f"\n[range] calibration genuine re-uploads compared: n={len(cal_gen)}")
    for m in METRICS:
        v = np.array([r[m] for r in cal_gen])
        print(f"[range] {m:15s} median {np.median(v):6.2f} | p99 {np.percentile(v, 99):6.2f} | max {v.max():6.2f}")

    def bounds(margin):
        return {m: maxima[m] * margin for m in METRICS}

    def flagged(r, b):
        return r["compared"] and any(r[m] > b[m] for m in METRICS)

    def outcome(r, b):
        """What the whole pipeline would do with this upload."""
        if thr is not None and r["gate"] < thr:
            return "no_match"             # Gate 1 loses it: Can't Verify
        if not r["aligned"] or not r["compared"]:
            return "not_compared"         # Can't Verify
        return "caught" if flagged(r, b) else "passed"

    # ---------------------------------------------------------------- 3. honest test on held-out pictures
    print(f"\n[test]  held-out pictures, margin sensitivity (genuine called edited / edits that pass as Authentic):")
    print(f"        {'margin':>6} | {'range top: diff / detail / chroma':>32} | genuine wrongly flagged | edits passed as Authentic")
    table = {}
    for mg in MARGINS:
        b = bounds(mg)
        gf = sum(1 for r in hold_gen if r["compared"] and flagged(r, b))
        gn = sum(1 for r in hold_gen if r["compared"])
        ps = sum(1 for r in hold_edits if outcome(r, b) == "passed")
        table[mg] = {"genuine_flagged": gf, "genuine_compared": gn, "edits_passed": ps, "edits_n": len(hold_edits)}
        print(f"        {mg:>6} | {b['block_diff_max']:>7.2f} / {b['detail_dev_max']:<6.2f} / "
              f"{b['block_chroma_diff_max']:<6.2f} | {gf:>3} of {gn:<3}              | {ps} of {len(hold_edits)}")

    b = bounds(args.margin)
    kinds = sorted({r["kind"] for r in hold_edits})
    print(f"\n[test]  chosen margin {args.margin}: held-out edits by type (caught / stopped before the picture check / passed):")
    by_kind = {}
    for k in kinds:
        rr = [r for r in hold_edits if r["kind"] == k]
        c = {o: sum(1 for r in rr if outcome(r, b) == o) for o in ("caught", "no_match", "not_compared", "passed")}
        by_kind[k] = {**c, "n": len(rr)}
        early = c["no_match"] + c["not_compared"]
        print(f"          {k:10s} caught {c['caught']:>2} | stopped early {early:>2} | PASSED {c['passed']:>2}   (of {len(rr)})")
    passed = sum(v["passed"] for v in by_kind.values())
    caught = sum(v["caught"] for v in by_kind.values())
    early = sum(v["no_match"] + v["not_compared"] for v in by_kind.values())
    n = len(hold_edits)
    gf = table.get(args.margin, {}).get("genuine_flagged")
    if gf is None:
        gf = sum(1 for r in hold_gen if r["compared"] and flagged(r, b))
    print(f"\n[pitch] on THIS held-out set: {passed} of {n} edited files would be called Authentic "
          f"({caught} caught by the picture check, {early} stopped earlier as Can't Verify).")
    if passed == 0:
        print(f"[pitch] zero misses out of {n} still only bounds the true rate below about {rule_of_three(n):.1%} at 95%.")
    else:
        print(f"[pitch] that is {passed / n:.1%} of this test set. Do NOT claim '0 false-Authentic' for the picture check.")
    print(f"[pitch] genuine re-uploads wrongly called edited: {gf} of {sum(1 for r in hold_gen if r['compared'])} "
          f"(held-out, compared).")

    report = {
        "rehearsal": True, "opencv": cv2.__version__, "margin": args.margin, "holdout": HOLDOUT,
        "gate": {"pos_n": len(pos), "pos_min": pos_min, "neg_n": len(neg), "neg_max": neg_max, "threshold": thr},
        "range": {"n": len(cal_gen), "max_seen": maxima, "upper_edge": b},
        "heldout": {"genuine_n": len(hold_gen), "genuine_compared": sum(1 for r in hold_gen if r["compared"]),
                    "genuine_flagged": gf, "edits_n": n, "edits_caught": caught, "edits_stopped_early": early,
                    "edits_passed_as_authentic": passed, "by_kind": by_kind, "margin_table": {str(k): v for k, v in table.items()}},
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "calibration_image.json").write_text(json.dumps(report, indent=2))
    print(f"\nraw numbers: {out / 'calibration_image.json'}")

    if args.write:
        if not separable:
            raise SystemExit("not writing: the gate is not separable")
        gr = paths.config("genuine_ranges")
        note = (f"MEASURED by tools/calibrate_image.py on {len(POOL) - len(HOLDOUT)} pictures x "
                f"{len(QUALITIES) * len(SCALES)} re-uploads (JPEG {min(QUALITIES)}-{max(QUALITIES)}, shrink "
                f"{min(SCALES)}-{max(SCALES)}); upper edge = largest genuine value x {args.margin}. "
                "Rehearsal numbers: redo on real data.")
        units = {"block_diff_max": "grey levels", "detail_dev_max": "|log ratio|",
                 "block_chroma_diff_max": "Lab a*b* distance"}
        for m in METRICS:
            key = f"diff_picture.{m}"
            gr.setdefault(key, {})
            gr[key].update(range=[0.0, round(b[m], 3)], unit=units[m], calibrated=True,
                           n=len(cal_gen), note=note)
        paths.write_json(paths.config_dir() / "genuine_ranges.json", gr)
        g = paths.config("gates")
        g["image"].update(trusted_threshold=thr, calibrated=True,
                          calibrated_embedder=il.embedder_for("image"),
                          n_positive=len(pos), n_negative=len(neg))
        paths.write_json(paths.config_dir() / "gates.json", g)
        print("wrote config/genuine_ranges.json and config/gates.json")


if __name__ == "__main__":
    main()

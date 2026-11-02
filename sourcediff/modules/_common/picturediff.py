"""Image alignment and picture comparison. Needs numpy and opencv.

align()    which homography maps the upload onto the source? ORB feature points + RANSAC
           at a normalized working resolution, so small re-uploads retain enough features.
           Survives resize / rotation / crop. Accuracy is about a pixel, which is NOT enough to compare pixels.
compare()  the part that took the work. Findings from measuring 660 genuine re-uploads and edits:

  1. The big enemy of "genuine re-upload" is registration error, not JPEG damage. Plain region-SSIM on the
     ORB alignment flagged half-size re-uploads as edited (worst-block score down to -0.04).
  2. So: bring the SOURCE down to the upload's resolution (shrinking the source is what a platform did; never
     enlarge the upload), refine the alignment with ECC (sub-pixel), blur both a little (sigma 3 px at <=384 px)
     so leftover edge shifts and JPEG noise vanish, then compare blocks.
  3. Three numbers come out, all "worst block":
       block_diff_max          mean absolute grey-level difference in an 8x8 grid  (pasted/changed content)
       detail_dev_max          |log ratio| of local gradient energy in a 12x12 grid (blurred/smoothed regions)
       block_chroma_diff_max   mean Lab a*/b* distance in an 8x8 grid (recolour that keeps brightness)

Alignment and ECC stay on grey. Colour is carried through the same warps so an iso-luminant recolour
(e.g. hair dye that matches the old luminance) still moves a metric. Grey-only inputs skip chroma.

The module never says 'edited'. It reports numbers; the verdict engine compares them with
config/genuine_ranges.json, which tools/calibrate_image.py measured.
"""
from __future__ import annotations

import cv2
import numpy as np

WORK = 384            # working size (longest side, px)
ALIGN_WORK = 768      # normalize feature detection across uploads with different resolutions
SIGMA = 3.0           # blur before the block difference
DIFF_GRID = 8
DETAIL_GRID = 12
DETAIL_C = 50.0       # keeps flat blocks from dominating the log ratio
MIN_VALID = 0.5       # at least half of the upload must overlap the source
MIN_BLOCK_VALID = 0.9
MIN_ECC_ALIGN = 0.9


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------------ coarse alignment
def align(up_gray: np.ndarray, src_gray: np.ndarray) -> dict:
    def feature_view(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        factor = ALIGN_WORK / float(max(h, w))
        if factor > 1.0:
            view = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        elif factor < 1.0:
            view = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        else:
            view = img
        sx, sy = view.shape[1] / float(w), view.shape[0] / float(h)
        to_original = np.diag([1.0 / sx, 1.0 / sy, 1.0])
        return view, to_original

    up_view, up_to_original = feature_view(up_gray)
    src_view, src_to_original = feature_view(src_gray)
    if np.isclose(up_gray.shape[1] / up_gray.shape[0],
                  src_gray.shape[1] / src_gray.shape[0], rtol=0.005):
        target_shape = up_view.shape[:2]
        if src_view.shape[:2] != target_shape:
            src_view = cv2.resize(src_view, (target_shape[1], target_shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            src_to_original = np.diag([src_gray.shape[1] / float(target_shape[1]),
                                       src_gray.shape[0] / float(target_shape[0]), 1.0])
    orb = cv2.ORB_create(2500)
    k1, d1 = orb.detectAndCompute(up_view, None)
    k2, d2 = orb.detectAndCompute(src_view, None)
    out = {"ok": False, "method": "orb_ransac", "matches": 0, "inliers": 0, "inlier_ratio": 0.0,
           "scale": None, "H": None, "reason": ""}

    def use_ecc_fallback(reason: str, initial: np.ndarray | None = None) -> dict:
        if up_view.shape[:2] != src_view.shape[:2]:
            out["reason"] = reason
            return out
        a = up_view.astype(np.float32) / 255.0
        b = src_view.astype(np.float32) / 255.0
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        starts = [initial.astype(np.float32)] if initial is not None else []
        starts.append(np.eye(3, dtype=np.float32))
        for start in starts:
            try:
                cc, warp = cv2.findTransformECC(a, b, start, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            except cv2.error:
                continue
            if cc < MIN_ECC_ALIGN or not np.isfinite(warp).all():
                continue
            H = src_to_original @ warp @ np.linalg.inv(up_to_original)
            det = float(np.linalg.det(H[:2, :2]))
            scale = float(np.sqrt(abs(det))) if abs(det) > 0 else 0.0
            sane = 0.1 < scale < 10 and abs(H[2, 0]) < 1e-2 and abs(H[2, 1]) < 1e-2
            if sane:
                out.update(ok=True, method="orb_ecc", scale=round(scale, 4), H=H.tolist(), reason="",
                           ecc_cc=round(float(cc), 4))
                return out
        out["reason"] = reason
        return out

    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return use_ecc_fallback("too few features to align")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    out["matches"] = len(good)
    if len(good) < 8:
        return use_ecc_fallback("too few good matches")
    a = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    b = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_view, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    if H_view is None or mask is None or not np.isfinite(H_view).all():
        return use_ecc_fallback("no homography")
    H = src_to_original @ H_view @ np.linalg.inv(up_to_original)
    inl = int(mask.sum())
    det = float(np.linalg.det(H[:2, :2]))
    scale = float(np.sqrt(abs(det))) if abs(det) > 0 else 0.0
    out.update(inliers=inl, inlier_ratio=round(inl / len(good), 3), scale=round(scale, 4), H=H.tolist())
    sane = 0.1 < scale < 10 and abs(H[2, 0]) < 1e-2 and abs(H[2, 1]) < 1e-2
    out["ok"] = bool(inl >= 25 and inl / len(good) >= 0.25 and sane)
    if not out["ok"]:
        reason = f"alignment not trusted (inliers={inl}, ratio={inl / len(good):.2f}, scale={scale:.2f})"
        fallback = use_ecc_fallback(reason, H_view)
        if fallback["ok"]:
            return fallback
        out["reason"] = reason
    return out


# ------------------------------------------------------------------ comparison
def _working(up: np.ndarray, H: np.ndarray, work: int):
    uh, uw = up.shape[:2]
    w = min(1.0, work / float(max(uh, uw)))
    upw = cv2.resize(up, None, fx=w, fy=w, interpolation=cv2.INTER_AREA) if w < 1 else up
    return upw, H @ np.diag([1 / w, 1 / w, 1.0])


def _source_in_upload_frame(upw: np.ndarray, src: np.ndarray, Hw: np.ndarray):
    """Resample the source into the upload's pixel grid (shrinking the source, never enlarging the upload)."""
    uh, uw = upw.shape[:2]
    s = float(np.sqrt(abs(np.linalg.det(Hw[:2, :2]))))      # source pixels per working pixel
    if s > 1.0:
        sm = cv2.resize(src, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
        M = np.diag([1 / s, 1 / s, 1.0]) @ Hw
    else:
        sm, M = src, Hw
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    sp = cv2.warpPerspective(sm, M, (uw, uh), flags=flags)
    mask_src = np.full(sm.shape[:2], 255, np.uint8)
    valid = cv2.warpPerspective(mask_src, M, (uw, uh),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 250
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return sp, valid


def _ecc(upw: np.ndarray, spw: np.ndarray, valid: np.ndarray):
    """Sub-pixel refinement. Returns (warp, correlation) or (None, 0.0) if it did not converge."""
    a = upw.astype(np.float32) / 255.0
    b = spw.astype(np.float32) / 255.0
    warp = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_HOMOGRAPHY, crit, valid.astype(np.uint8), 5)
    except cv2.error:
        return None, 0.0
    if not np.isfinite(warp).all():
        return None, 0.0
    return warp, float(cc)


def _blocks(arr: np.ndarray, valid: np.ndarray, grid: int, fn):
    """[(gy, gx, (x0, y0, x1, y1), value)] for every block that is at least 90% inside the valid area."""
    h, w = arr.shape
    out = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1, x0, x1 = gy * h // grid, (gy + 1) * h // grid, gx * w // grid, (gx + 1) * w // grid
            m = valid[y0:y1, x0:x1]
            if m.size == 0 or m.mean() < MIN_BLOCK_VALID:
                continue
            out.append((gy, gx, (x0, y0, x1, y1), float(fn(arr[y0:y1, x0:x1][m]))))
    return out


def _energy(x: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(x, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return gx * gx + gy * gy


def _chroma_map(up_bgr: np.ndarray, src_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel Lab a*/b* distance after a light blur (same sigma as grey block diff)."""
    def ab(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = cv2.GaussianBlur(lab[..., 1], (0, 0), SIGMA)
        b = cv2.GaussianBlur(lab[..., 2], (0, 0), SIGMA)
        return a, b

    ua, ub = ab(up_bgr)
    sa, sb = ab(src_bgr)
    return np.sqrt((ua - sa) ** 2 + (ub - sb) ** 2)


def compare(up: np.ndarray, src: np.ndarray, H, work: int = WORK) -> dict:
    """Compare an aligned upload with its source. Accepts grey or BGR; chroma needs both in colour.
    Returns numbers plus the arrays needed to draw them."""
    H = np.asarray(H, dtype=np.float64)
    colour = up.ndim == 3 and src.ndim == 3 and up.shape[2] == 3 and src.shape[2] == 3
    up_g, src_g = _gray(up), _gray(src)
    upw_g, Hw = _working(up_g, H, work)
    spw_g, valid = _source_in_upload_frame(upw_g, src_g, Hw)
    if valid.mean() < MIN_VALID:
        return {"ok": False, "reason": "upload overlaps too little of the source"}
    warp, cc = _ecc(upw_g, spw_g, valid)
    if warp is None:
        return {"ok": False, "reason": "fine alignment (ECC) did not converge"}
    shape = (upw_g.shape[1], upw_g.shape[0])
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    sp2_g = cv2.warpPerspective(spw_g, warp, shape, flags=flags)
    v2 = cv2.warpPerspective(valid.astype(np.uint8) * 255, warp, shape, flags=flags) > 250
    v2 = cv2.erode(v2.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    a = cv2.GaussianBlur(upw_g.astype(np.float32), (0, 0), SIGMA)
    b = cv2.GaussianBlur(sp2_g.astype(np.float32), (0, 0), SIGMA)
    diff_blocks = _blocks(np.abs(a - b), v2, DIFF_GRID, np.mean)

    ea, eb = _energy(upw_g.astype(np.float32)), _energy(sp2_g.astype(np.float32))
    ba, bb = _blocks(ea, v2, DETAIL_GRID, np.mean), _blocks(eb, v2, DETAIL_GRID, np.mean)
    detail_blocks = [(ga[0], ga[1], ga[2], abs(float(np.log((ga[3] + DETAIL_C) / (gb[3] + DETAIL_C)))))
                     for ga, gb in zip(ba, bb)]
    if not diff_blocks or not detail_blocks:
        return {"ok": False, "reason": "too little overlap to compare blocks"}

    out = {"ok": True,
           "block_diff_max": max(b_[3] for b_ in diff_blocks),
           "detail_dev_max": max(b_[3] for b_ in detail_blocks),
           "diff_blocks": diff_blocks, "detail_blocks": detail_blocks,
           "chroma_blocks": [], "block_chroma_diff_max": 0.0,
           "ecc_cc": cc, "size": list(shape),
           "upload_w": upw_g, "source_w": sp2_g, "valid": v2}

    if colour:
        upw_c, _ = _working(up, H, work)
        spw_c, _ = _source_in_upload_frame(upw_c, src, Hw)
        sp2_c = cv2.warpPerspective(spw_c, warp, shape, flags=flags)
        chroma_blocks = _blocks(_chroma_map(upw_c, sp2_c), v2, DIFF_GRID, np.mean)
        if not chroma_blocks:
            return {"ok": False, "reason": "too little overlap to compare colour blocks"}
        out.update(block_chroma_diff_max=max(b_[3] for b_ in chroma_blocks),
                   chroma_blocks=chroma_blocks, upload_bgr=upw_c, source_bgr=sp2_c)
    return out


def overlay(res: dict, diff_limit: float | None, detail_limit: float | None,
            chroma_limit: float | None = None) -> np.ndarray:
    """Upload (left) beside the aligned original (right). Red = grey block above range,
    orange = detail above range, magenta = chroma above range."""
    if "upload_bgr" in res:
        left, right = res["upload_bgr"].copy(), res["source_bgr"].copy()
    else:
        left = cv2.cvtColor(res["upload_w"], cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(res["source_w"], cv2.COLOR_GRAY2BGR)
    for img in (left, right):
        for _, _, (x0, y0, x1, y1), val in res["detail_blocks"]:
            if detail_limit is not None and val > detail_limit:
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 140, 255), 1)
        for _, _, (x0, y0, x1, y1), val in res.get("chroma_blocks", []):
            if chroma_limit is not None and val > chroma_limit:
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 255), 2)
        for _, _, (x0, y0, x1, y1), val in res["diff_blocks"]:
            if diff_limit is not None and val > diff_limit:
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
    bar = np.full((22, left.shape[1] * 2, 3), 255, np.uint8)
    cv2.putText(bar, "UPLOAD", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, "ORIGINAL (aligned, same size)", (left.shape[1] + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, np.hstack([left, right])])


def explain_diff_regions(res: dict, r_diff: tuple | None, r_detail: tuple | None,
                         r_chroma: tuple | None = None) -> dict:
    """Analyze differing blocks and return structured, human-understandable explanation."""
    diff_lim = r_diff[1] if r_diff else 2.63
    detail_lim = r_detail[1] if r_detail else 2.94
    chroma_lim = r_chroma[1] if r_chroma else 2.81

    w, h = res.get("size", (WORK, WORK))
    diff_bad = [b for b in res.get("diff_blocks", []) if b[3] > diff_lim]
    detail_bad = [b for b in res.get("detail_blocks", []) if b[3] > detail_lim]
    chroma_bad = [b for b in res.get("chroma_blocks", []) if b[3] > chroma_lim]

    has_diff = len(diff_bad) > 0
    has_detail = len(detail_bad) > 0
    has_chroma = len(chroma_bad) > 0

    if not (has_diff or has_detail or has_chroma):
        return {
            "has_changes": False,
            "headline": "No visual modifications detected",
            "narrative": "Every analyzed region of the image is consistent with the genuine library original within calibrated tolerances.",
            "narrative_hi": "तस्वीर का हर क्षेत्र मूल स्रोत से मेल खाता है। कोई बदलाव नहीं पाया गया।",
            "items": [],
            "edit_types": [],
        }

    edit_types = []
    if has_diff:
        edit_types.append("Content Altered")
    if has_detail:
        edit_types.append("Texture Smoothed / Blurred")
    if has_chroma:
        edit_types.append("Color / Chroma Modified")

    def region_name(gx, gy, grid_w, grid_h):
        v_desc = "Top" if gy < grid_h / 3 else ("Center" if gy < 2 * grid_h / 3 else "Bottom")
        h_desc = "Left" if gx < grid_w / 3 else ("Center" if gx < 2 * grid_w / 3 else "Right")
        if v_desc == "Center" and h_desc == "Center":
            return "Center"
        if v_desc == "Center":
            return f"{h_desc} Side"
        if h_desc == "Center":
            return f"{v_desc} Area"
        return f"{v_desc}-{h_desc}"

    all_bad_boxes = []
    region_counts: dict[str, int] = {}
    for gy, gx, (x0, y0, x1, y1), val in diff_bad:
        rname = region_name(gx, gy, DIFF_GRID, DIFF_GRID)
        region_counts[rname] = region_counts.get(rname, 0) + 1
        all_bad_boxes.append((x0, y0, x1, y1))

    for gy, gx, (x0, y0, x1, y1), val in chroma_bad:
        rname = region_name(gx, gy, DIFF_GRID, DIFF_GRID)
        region_counts[rname] = region_counts.get(rname, 0) + 1
        all_bad_boxes.append((x0, y0, x1, y1))

    for gy, gx, (x0, y0, x1, y1), val in detail_bad:
        rname = region_name(gx, gy, DETAIL_GRID, DETAIL_GRID)
        region_counts[rname] = region_counts.get(rname, 0) + 1
        all_bad_boxes.append((x0, y0, x1, y1))

    if all_bad_boxes:
        min_x = min(b[0] for b in all_bad_boxes)
        min_y = min(b[1] for b in all_bad_boxes)
        max_x = max(b[2] for b in all_bad_boxes)
        max_y = max(b[3] for b in all_bad_boxes)
        affected_area_pct = round(((max_x - min_x) * (max_y - min_y)) / float(w * h) * 100, 1)
        bounding_box = [int(min_x), int(min_y), int(max_x), int(max_y)]
    else:
        bounding_box = [0, 0, w, h]
        affected_area_pct = 0.0

    top_regions = sorted(region_counts.keys(), key=lambda r: region_counts[r], reverse=True)
    location_summary = ", ".join(top_regions[:3]) if top_regions else "Localized area"

    items = []
    sentences = []
    if has_diff:
        max_v = max(b[3] for b in diff_bad)
        msg = f"Content modification (pasted elements, text overlay, or altered pixels) was detected in the {location_summary} area (pixel difference reached {max_v:.1f}, normal limit {diff_lim:.2f})."
        sentences.append(msg)
        items.append({
            "where": location_summary,
            "what": "Content modification / pasted elements",
            "details": f"Grey-level difference: {max_v:.1f} (genuine range [0.0, {diff_lim:.2f}]) across {len(diff_bad)} blocks",
        })
    if has_detail:
        max_v = max(b[3] for b in detail_bad)
        msg = f"Texture smoothing or blurring was applied in the {location_summary} area (detail gradient deviation of {max_v:.2f} vs genuine limit {detail_lim:.2f})."
        sentences.append(msg)
        items.append({
            "where": location_summary,
            "what": "Texture smoothing / blurring",
            "details": f"Detail gradient ratio: {max_v:.2f} (genuine range [0.0, {detail_lim:.2f}]) across {len(detail_bad)} blocks",
        })
    if has_chroma:
        max_v = max(b[3] for b in chroma_bad)
        msg = f"Color/chroma alterations were identified (color distance {max_v:.1f} vs genuine limit {chroma_lim:.2f}) while overall luminance remained similar."
        sentences.append(msg)
        items.append({
            "where": location_summary,
            "what": "Color alteration / recoloring",
            "details": f"Chroma distance: {max_v:.1f} (genuine range [0.0, {chroma_lim:.2f}]) across {len(chroma_bad)} blocks",
        })

    sentences.append(
        f"The altered area covers approximately {affected_area_pct}% of the image."
    )
    narrative = " ".join(sentences)
    narrative_hi = f"तस्वीर में बदलाव पाया गया: {location_summary} क्षेत्र में बदलाव हुआ है (लगभग {affected_area_pct}% भाग प्रभावित)।"

    return {
        "has_changes": True,
        "headline": f"Modifications detected in {location_summary}",
        "location_summary": location_summary,
        "affected_area_pct": affected_area_pct,
        "bounding_box": bounding_box,
        "edit_types": edit_types,
        "items": items,
        "narrative": narrative,
        "narrative_hi": narrative_hi,
        "bad_blocks_count": len(diff_bad) + len(detail_bad) + len(chroma_bad),
    }

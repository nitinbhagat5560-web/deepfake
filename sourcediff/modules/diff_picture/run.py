"""diff_picture  [path1]  channel: picture  doors: image and sampled-frame video comparison

IMAGE: compare the upload with the matched source (see modules/_common/picturediff.py for why it is done
this way) and report three numbers, all "worst block":
  block_diff_max          grey-level difference (pasted / changed content)
  detail_dev_max          change in local detail (blurred / smoothed regions)
  block_chroma_diff_max   Lab a*/b* distance (recolour that keeps brightness)
The blocks above the genuine range are boxed in diff_picture_map.png, which shows the upload beside the
aligned original (red = grey, orange = detail, magenta = chroma).

VIDEO: compare each 1-second upload sample to the source frame predicted by the fitted timeline. Any
frame that cannot be decoded, aligned, or compared makes this channel not run rather than silently
omitting evidence. Video metrics use calibrated H.264/H.265 re-encode ranges.

The module never says 'edited'. It reports numbers; the engine compares them with config/genuine_ranges.json.
"""
import cv2
import numpy as np

from sourcediff import paths
from sourcediff.contract import Evidence, ModuleResult
from sourcediff.module_api import genuine_range, in_scenario, run_module, skipped, stub_run

NAME = "diff_picture"


def run(ctx):
    if in_scenario(ctx):
        return stub_run(ctx, NAME)
    if ctx.door == "video":
        from modules._common import imagelib as il, picturediff as pd, videolib as vl

        al = ctx.read_json("alignment.json")
        if not al.get("ok") or al.get("consistent") is False:
            return skipped(NAME, "no consistent video alignment to compare")
        ref = ctx.read_json("source_ref.json")
        source = vl.source_path("trusted", ref["source_id"])
        timestamps, duration = vl.frame_timestamps(ctx.input_path)
        offset, speed = float(al["offset_s"]), float(al["speed"])
        source_timestamps = [speed * timestamp + offset for timestamp in timestamps]
        for timestamp, source_timestamp in zip(timestamps, source_timestamps):
            if source_timestamp < 0:
                return skipped(NAME, f"aligned source time is before the source at upload {timestamp:.3f}s")
        maxima = {}
        candidate_comparisons = {}
        per_second_comparisons = []
        worst_grey = None
        try:
            upload_frames = vl.iter_frames_at(ctx.input_path, timestamps)
            source_frames = vl.iter_frames_at(source, source_timestamps)
            for timestamp, source_timestamp, upload, original in zip(
                    timestamps, source_timestamps, upload_frames, source_frames):
                frame_alignment = pd.align(il.gray(upload), il.gray(original))
                if not frame_alignment["ok"]:
                    return skipped(NAME, f"frame at {timestamp:.3f}s could not be aligned: "
                                         f"{frame_alignment['reason']}")
                comparison = pd.compare(upload, original, frame_alignment["H"])
                if not comparison["ok"]:
                    return skipped(NAME, f"frame at {timestamp:.3f}s could not be compared: "
                                         f"{comparison['reason']}")
                per_second_comparisons.append((timestamp, comparison))
                for metric in ("block_diff_max", "detail_dev_max", "block_chroma_diff_max"):
                    current = maxima.get(metric)
                    if current is None or comparison[metric] > current[0]:
                        maxima[metric] = (comparison[metric], timestamp)
                        candidate_comparisons[timestamp] = comparison
                retained_timestamps = {item[1] for item in maxima.values()}
                candidate_comparisons = {
                    ts: item for ts, item in candidate_comparisons.items() if ts in retained_timestamps
                }
                if worst_grey is None or comparison["block_diff_max"] > worst_grey[0]:
                    worst_grey = (comparison["block_diff_max"], timestamp, comparison)
        except ValueError as exc:
            return skipped(NAME, str(exc))
        if not maxima or worst_grey is None:
            return skipped(NAME, "video contains no comparable frames")
        if len(per_second_comparisons) != len(timestamps):
            return skipped(NAME, "one or more sampled upload/source frame pairs could not be decoded")

        candidate_metrics = {}
        for metric, (_, timestamp) in maxima.items():
            candidate_metrics.setdefault(timestamp, []).append(metric)
        timeline_frames = [
            (
                timestamp,
                speed * timestamp + offset,
                ", ".join(metrics),
                candidate_comparisons[timestamp]["upload_bgr"],
                candidate_comparisons[timestamp]["source_bgr"],
            )
            for timestamp, metrics in sorted(candidate_metrics.items())
        ]
        metric_specs = (
            ("block_diff_max", "video_block_diff_max", "grey levels"),
            ("detail_dev_max", "video_detail_dev_max", "|log ratio|"),
            ("block_chroma_diff_max", "video_block_chroma_diff_max", "Lab a*b* distance"),
        )
        rows = []
        for timestamp, comparison in per_second_comparisons:
            end = min(duration, timestamp + vl.SAMPLE_INTERVAL_S)
            for result_key, metric, unit in metric_specs:
                rows.append(Evidence(
                    module=NAME, door="video", family="source_diff", channel="picture",
                    metric=metric, value=round(comparison[result_key], 3), unit=unit,
                    genuine_range=genuine_range(NAME, metric),
                    timestamps=[[round(timestamp, 3), round(end, 3)]],
                    limits=["measured on this 1-second sampled frame",
                            "video-specific calibrated genuine range",
                            "spatial comparison uses per-frame ORB and ECC alignment"],
                ))
        _, timestamp, worst = worst_grey
        image = pd.overlay(
            worst,
            genuine_range(NAME, "video_block_diff_max")[1]
            if genuine_range(NAME, "video_block_diff_max") else None,
            genuine_range(NAME, "video_detail_dev_max")[1]
            if genuine_range(NAME, "video_detail_dev_max") else None,
            genuine_range(NAME, "video_block_chroma_diff_max")[1]
            if genuine_range(NAME, "video_block_chroma_diff_max") else None,
        )
        cv2.putText(image, f"worst grey difference at upload {timestamp:.2f}s",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(ctx.path("diff_picture_map.png")), image):
            raise RuntimeError("could not write video picture comparison map")
        tile_width, tile_height, label_height, row_gap = 460, 260, 28, 12
        sheet = np.full(
            (len(timeline_frames) * (tile_height + label_height + row_gap), tile_width * 2, 3),
            248, dtype="uint8",
        )
        for row, (upload_time, source_time, metrics, upload_frame, source_frame) in enumerate(timeline_frames):
            y = row * (tile_height + label_height + row_gap)
            for column, (label, frame) in enumerate((
                (f"REVIEW CANDIDATE  {upload_time:.2f}s (max: {metrics})", upload_frame),
                (f"TRUSTED SOURCE  {source_time:.2f}s", source_frame),
            )):
                x = column * tile_width
                cv2.putText(sheet, label, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (25, 30, 38), 1, cv2.LINE_AA)
                h, w = frame.shape[:2]
                scale = min((tile_width - 16) / w, tile_height / h)
                resized = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                top = y + label_height + (tile_height - resized.shape[0]) // 2
                left = x + (tile_width - resized.shape[1]) // 2
                sheet[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        if not cv2.imwrite(str(ctx.path("video_timeline.png")), sheet):
            raise RuntimeError("could not write timestamped video comparison")

        r_vdiff = genuine_range(NAME, "video_block_diff_max")
        vdiff_lim = r_vdiff[1] if r_vdiff else 3.85
        deviant_seconds = [
            round(t, 2) for t, c in per_second_comparisons if c["block_diff_max"] > vdiff_lim
        ]
        video_explanation = {
            "has_changes": bool(deviant_seconds),
            "headline": (f"Visual frame differences detected across {len(deviant_seconds)} sampled second(s)"
                         if deviant_seconds else "Video frames match trusted original within range"),
            "worst_timestamp_s": round(timestamp, 2),
            "worst_grey_diff": round(worst["block_diff_max"], 2),
            "deviant_seconds": deviant_seconds,
            "narrative": (
                f"Visual frame divergence was detected at {len(deviant_seconds)} second(s) "
                f"(peak grey-level difference {worst['block_diff_max']:.2f} at upload time {timestamp:.2f}s). "
                "Representative frames are shown in the timeline comparison."
                if deviant_seconds else "Sampled video frames align and visually match the trusted source."
            ),
            "narrative_hi": (
                f"वीडियो फ़्रेम में अंतर पाया गया (अपलोड समय {timestamp:.2f}s पर सबसे बड़ा अंतर)।"
                if deviant_seconds else "वीडियो फ़्रेम मूल स्रोत से मेल खाते हैं।"
            ),
        }
        ctx.write_json("diff_detail.json", video_explanation)

        return ModuleResult(
            module=NAME, evidence=rows,
            artifacts={"map": "diff_picture_map.png", "timeline": "video_timeline.png",
                       "detail": "diff_detail.json"},
            version="video-0.1",
        )
    if ctx.door != "image":
        return stub_run(ctx, NAME)
    from modules._common import imagelib as il, picturediff as pd

    al = ctx.read_json("alignment.json")
    if not al.get("ok"):
        return skipped(NAME, "no usable alignment: " + al.get("reason", ""))
    ref = ctx.read_json("source_ref.json")
    # Keep BGR so Lab chroma can catch iso-luminant recolours; aligner already ran on grey.
    src = il.read_bgr(paths.library_dir() / ref["file"])
    up = il.read_bgr(ctx.input_path)
    res = pd.compare(up, src, al["H"])
    if not res["ok"]:
        return skipped(NAME, res["reason"])

    r_diff = genuine_range(NAME, "block_diff_max")
    r_detail = genuine_range(NAME, "detail_dev_max")
    r_chroma = genuine_range(NAME, "block_chroma_diff_max")
    cv2.imwrite(str(ctx.path("diff_picture_map.png")),
                pd.overlay(res,
                           r_diff[1] if r_diff else None,
                           r_detail[1] if r_detail else None,
                           r_chroma[1] if r_chroma else None))

    exp = pd.explain_diff_regions(res, r_diff, r_detail, r_chroma)
    ctx.write_json("diff_detail.json", exp)

    shared = [f"compared at {max(res['size'])} px after alignment refined to sub-pixel (ECC cc={res['ecc_cc']:.3f})",
              "boxes in diff_picture_map.png mark blocks above the genuine range"]
    rows = [
        Evidence(module=NAME, door="image", family="source_diff", channel="picture",
                 metric="block_diff_max", value=round(res["block_diff_max"], 3), unit="grey levels",
                 genuine_range=r_diff, limits=shared + ["worst of an 8x8 grid of blocks"]),
        Evidence(module=NAME, door="image", family="source_diff", channel="picture",
                 metric="detail_dev_max", value=round(res["detail_dev_max"], 3), unit="|log ratio|",
                 genuine_range=r_detail, limits=shared + ["worst of a 12x12 grid; flags blurred or smoothed regions"]),
        Evidence(module=NAME, door="image", family="source_diff", channel="picture",
                 metric="block_chroma_diff_max", value=round(res["block_chroma_diff_max"], 3), unit="Lab a*b* distance",
                 genuine_range=r_chroma, limits=shared + ["worst of an 8x8 grid; flags recolour that keeps brightness"]),
    ]
    return ModuleResult(module=NAME, evidence=rows,
                        artifacts={"map": "diff_picture_map.png", "detail": "diff_detail.json"},
                        version="rehearsal-0.3")


if __name__ == "__main__":
    run_module(NAME, run)

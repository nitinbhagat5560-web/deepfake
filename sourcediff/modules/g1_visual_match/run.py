"""g1_visual_match  [gate1]  doors: image and sampled-frame video matching

IMAGE: looks the upload up in library/trusted/image with FAISS and the configured local embedder.
       (SHA-256 is NOT used here: a re-upload has different bytes. FAISS compares "what it looks like".)

VIDEO: sample 1 frame/s and search a FAISS index of trusted-video frames using the configured embedder.
Require consecutive hits to the same source with increasing source timestamps. Each embedder needs its own
threshold calibration on genuine video copies and hard negatives.

Emits a Gate. Also writes gate1_detail.json (scores of the top hits) and source_ref.json (which library
file the aligner and diff modules compare against), plus chunk_map.json for video alignment.
"""
from sourcediff import paths
from sourcediff.contract import Gate, ModuleResult
from sourcediff import library
from sourcediff.module_api import in_scenario, run_module, stub_run

NAME = "g1_visual_match"


def run(ctx):
    if in_scenario(ctx):
        return stub_run(ctx, NAME)
    if ctx.door == "video":
        from modules._common import imagelib as il, videolib as vl

        exact = library.find_exact("trusted", "video", ctx.sha256)
        if exact:
            meta, media = exact
            duration = float(ctx.ingest.get("duration_s") or 0)
            ctx.write_json("source_ref.json", {
                "library": "trusted", "source_id": meta["source_id"],
                "file": str(media.relative_to(paths.library_dir())),
            })
            detail = {
                "embedder": "sha256_exact_identity", "exact_match": True,
                "source_id": meta["source_id"],
                "source_file_sha256_verified": True,
            }
            ctx.write_json("gate1_detail.json", detail)
            return ModuleResult(
                module=NAME,
                gate=Gate(library="trusted", matched=True, source_id=meta["source_id"],
                          coverage_of_upload=1.0,
                          matched_ranges=[[0.0, duration]] if duration > 0 else [],
                          votes=1, index_version="sha256_exact_identity", exact_match=True),
                artifacts={"detail": "gate1_detail.json", "source_ref": "source_ref.json"},
                version="exact-sha256-1",
            )

        timestamps, duration = vl.frame_timestamps(ctx.input_path)
        samples = [(timestamp, il.embed_array(frame, vl.embedder()))
                   for timestamp, frame in zip(timestamps, vl.iter_frames_at(ctx.input_path, timestamps))]
        video_gate = paths.config("gates")["video"]
        threshold = float(video_gate["trusted_threshold"])
        match = vl.match_embeddings(samples, duration, threshold)
        calibrated = (
            bool(video_gate.get("calibrated"))
            and video_gate.get("calibrated_embedder", il.TINY_EMBEDDER) == vl.embedder()
        )
        accepted = match["matched"] and calibrated
        detail = {
            "embedder": vl.embedder(),
            "sample_interval_s": vl.SAMPLE_INTERVAL_S,
            "threshold": threshold,
            "threshold_calibrated": calibrated,
            "candidate_match": match["matched"],
            "usable_match": accepted,
            "index": vl.index_version("trusted"),
            "sample_count": len(samples),
            "votes": match["votes"],
            "top": match["sample_hits"],
        }
        ctx.write_json("gate1_detail.json", detail)
        artifacts = {"detail": "gate1_detail.json"}
        if accepted:
            media = vl.source_path("trusted", match["source_id"])
            ctx.write_json("source_ref.json", {
                "library": "trusted", "source_id": match["source_id"],
                "file": str(media.relative_to(paths.library_dir())),
            })
            ctx.write_json("chunk_map.json", match["points"])
            artifacts.update(source_ref="source_ref.json", chunk_map="chunk_map.json")
        gate = Gate(
            library="trusted", matched=accepted, source_id=match["source_id"] if accepted else None,
            coverage_of_upload=match["coverage_of_upload"] if accepted else 0.0,
            matched_ranges=match["matched_ranges"] if accepted else [],
            unmatched_ranges=match["unmatched_ranges"] if accepted
            else ([[0.0, round(duration, 3)]] if duration > 0 else []),
            votes=match["votes"], threshold=threshold, index_version=vl.index_version("trusted"),
        )
        return ModuleResult(module=NAME, gate=gate, artifacts=artifacts, version="video-0.1")
    if ctx.door != "image":
        return stub_run(ctx, NAME)

    exact = library.find_exact("trusted", "image", ctx.sha256)
    if exact:
        meta, media = exact
        ctx.write_json("source_ref.json", {
            "library": "trusted", "source_id": meta["source_id"],
            "file": str(media.relative_to(paths.library_dir())),
        })
        ctx.write_json("gate1_detail.json", {
            "embedder": "sha256_exact_identity", "exact_match": True,
            "source_id": meta["source_id"], "source_file_sha256_verified": True,
        })
        return ModuleResult(
            module=NAME,
            gate=Gate(library="trusted", matched=True, source_id=meta["source_id"],
                      coverage_of_upload=1.0, votes=1,
                      index_version="sha256_exact_identity", exact_match=True),
            artifacts={"detail": "gate1_detail.json", "source_ref": "source_ref.json"},
            version="exact-sha256-1",
        )

    from modules._common import imagelib as il, picturediff as pd

    image_gate = paths.config("gates")["image"]
    embedder = il.embedder_for("image")
    up_bgr = il.read_bgr(ctx.input_path)
    hits = il.search_views("trusted", up_bgr, k=5)
    thr = float(image_gate["trusted_threshold"])
    calibrated = (
        bool(image_gate.get("calibrated"))
        and image_gate.get("calibrated_embedder", il.TINY_EMBEDDER) == embedder
    )

    matched_source_id = None
    match_method = None
    align_detail = None

    for sid, score in hits:
        if score >= thr and calibrated:
            matched_source_id = sid
            match_method = "thumbnail_similarity"
            break
        if score >= 0.35:
            try:
                src_path = il.source_path("trusted", sid)
                src_bgr = il.read_bgr(src_path)
                al = pd.align(il.gray(up_bgr), il.gray(src_bgr))
                if al.get("ok"):
                    matched_source_id = sid
                    match_method = "geometric_alignment"
                    align_detail = {
                        "inliers": al.get("inliers"),
                        "inlier_ratio": al.get("inlier_ratio"),
                        "scale": al.get("scale"),
                        "method": al.get("method"),
                    }
                    break
            except Exception:
                continue

    matched = bool(matched_source_id)
    candidate_match = matched
    detail = {
        "embedder": embedder, "threshold": thr, "threshold_calibrated": calibrated,
        "candidate_match": candidate_match, "match_method": match_method,
        "index": il.index_version("trusted"),
        "top": [{"source_id": s, "score": round(v, 4)} for s, v in hits],
    }
    if align_detail:
        detail["align_verification"] = align_detail
    ctx.write_json("gate1_detail.json", detail)
    arts = {"detail": "gate1_detail.json"}
    if matched:
        f = il.source_path("trusted", matched_source_id)
        ctx.write_json("source_ref.json", {"library": "trusted", "source_id": matched_source_id,
                                           "file": str(f.relative_to(paths.library_dir()))})
        arts["source_ref"] = "source_ref.json"
    gate = Gate(library="trusted", matched=matched, source_id=matched_source_id if matched else None,
                coverage_of_upload=1.0 if matched else 0.0, votes=1 if matched else 0,
                threshold=thr, index_version=il.index_version("trusted"))
    return ModuleResult(module=NAME, gate=gate, artifacts=arts, version="rehearsal-0.3")


if __name__ == "__main__":
    run_module(NAME, run)

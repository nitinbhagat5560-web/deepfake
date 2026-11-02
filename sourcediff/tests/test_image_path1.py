"""Path 1 for the image door, on real pictures. Uses the demo library that tools/seed_demo.py builds."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import _variants as V
from modules._common import imagelib as il, picturediff as pd
from sourcediff import paths, runner
from sourcediff.ingest import create_case

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "demo" / "media"
LIBRARY = ["astronaut", "camera", "coins", "chelsea", "coffee", "rocket", "moon", "immunohistochemistry"]

CASES = [  # file, verdict, reason code
    ("genuine_reupload_astronaut.jpg", "authentic", "full_coverage_all_agree"),
    ("genuine_reupload_coffee.jpg", "authentic", "full_coverage_all_agree"),
    ("genuine_reupload_chelsea.jpg", "authentic", "full_coverage_all_agree"),
    ("edited_astronaut_patch.jpg", "edited_picture", "channel_differs"),
    ("edited_coffee_text.jpg", "edited_picture", "channel_differs"),
    ("edited_chelsea_copymove.jpg", "edited_picture", "channel_differs"),
    ("unrelated_retina.jpg", "cant_verify", "no_original"),
    ("known_limit_crop_astronaut.jpg", "authentic", "full_coverage_all_agree"),
]


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_DATA", str(tmp_path / "data"))
    return tmp_path


def run(path, root):
    return runner.run_case(create_case(path, Path(path).name, root=root / "cases"))


@pytest.mark.parametrize("fn,verdict,code", CASES)
def test_demo_upload(data, fn, verdict, code):
    out = run(MEDIA / fn, data)
    assert (out["verdict"], out["reason_code"]) == (verdict, code)
    case = Path(out["case_dir"])
    v = paths.load_json(case / "verdict.json")
    assert not v["stub_used"] and not any("not calibrated" in w for w in v["warnings"])
    assert (case / "card.md").exists() and (case / "card.json").exists()
    if verdict == "edited_picture":
        assert v["differing_channels"] == ["picture"] and (case / "diff_picture_map.png").exists()
    if verdict == "authentic":
        assert v["channels"] == {"alignment": "agrees", "picture": "agrees"}
        assert (case / "diff_picture_map.png").exists()
    if code == "no_original":
        assert v["stages_run"] == ["gate1"]                      # nothing to diff against
        assert "No original found. This does not mean the media is genuine." in v["notes"]
        card = paths.load_json(case / "card.json")
        assert card["title"] == "Can't Verify"


def test_every_library_picture_re_uploaded_is_never_called_edited(data, tmp_path):
    """False alarms on genuine copies: a library picture, shrunk and recompressed, must be Authentic or Can't Verify."""
    verdicts = []
    for name in LIBRARY:
        img = V.reupload(V.sample(name), 50, 0.7)
        f = tmp_path / f"{name}.jpg"
        cv2.imwrite(str(f), img)
        verdicts.append((name, run(f, data)["verdict"]))
    assert all(v in ("authentic", "cant_verify") for _, v in verdicts), verdicts
    assert sum(v == "authentic" for _, v in verdicts) >= 6, verdicts


@pytest.mark.parametrize("name", ["hubble_deep_field", "retina", "cell", "page", "clock", "text", "microaneurysms",
                                  "brick", "gravel", "grass"])
def test_pictures_not_in_the_library_are_never_authentic(data, tmp_path, name):
    f = tmp_path / f"{name}.jpg"
    cv2.imwrite(str(f), V.reupload(V.sample(name), 70, 0.8))
    assert run(f, data)["verdict"] == "cant_verify"


def test_a_picture_with_nothing_to_align_on_is_cant_verify_never_authentic(data, tmp_path, monkeypatch):
    """Library picture matched at Gate 1 but the aligner cannot fit it: the picture check cannot run."""
    f = tmp_path / "flat.png"
    cv2.imwrite(str(f), np.full((200, 300, 3), 128, np.uint8))
    out = run(f, data)
    assert out["verdict"] == "cant_verify"


def test_compare_identical_is_tiny_and_a_pasted_patch_is_big():
    src = il.gray(V.sample("astronaut"))
    H = np.eye(3)
    same = pd.compare(src, src, H)
    assert same["ok"] and same["block_diff_max"] < 0.5 and same["detail_dev_max"] < 0.5
    edited = V.paste_patch(V.sample("astronaut"), V.sample("chelsea"), 300, 200, 120)
    res = pd.compare(il.gray(edited), src, H)
    assert res["ok"] and res["block_diff_max"] > 10


def test_iso_luminant_recolour_is_caught_by_chroma_not_grey():
    """Hair-dye style: shift Lab a*/b*, keep L*. Grey metrics stay small; chroma must move."""
    src = V.sample("chelsea")  # colour photo (cat)
    edited = V.recolour_region(src, 80, 40, 120)
    H = np.eye(3)
    grey = pd.compare(il.gray(edited), il.gray(src), H)
    colour = pd.compare(edited, src, H)
    assert grey["ok"] and colour["ok"]
    # Same brightness → grey block diff stays near a genuine re-upload; chroma must not.
    lim = paths.config("genuine_ranges")
    assert grey["block_diff_max"] <= lim["diff_picture.block_diff_max"]["range"][1] * 2
    assert colour["block_chroma_diff_max"] > lim["diff_picture.block_chroma_diff_max"]["range"][1]
    assert colour["block_chroma_diff_max"] > 5.0


def test_align_refuses_flat_pictures():
    flat = np.full((200, 300), 128, np.uint8)
    al = pd.align(flat, flat)
    assert al["ok"] is False and al["reason"]


def test_align_recovers_a_shrunk_copy_and_compare_accepts_it():
    src = il.gray(V.sample("camera"))
    up = il.gray(V.reupload(V.sample("camera"), 60, 0.5))
    al = pd.align(up, src)
    assert al["ok"] and 1.8 < al["scale"] < 2.2
    res = pd.compare(up, src, al["H"])
    lim = paths.config("genuine_ranges")
    assert res["ok"]
    assert res["block_diff_max"] <= lim["diff_picture.block_diff_max"]["range"][1]


@pytest.mark.parametrize("name", ["camera", "rocket", "moon"])
def test_align_recovers_a_small_compressed_copy_and_compare_accepts_it(name):
    src = V.sample(name)
    up = V.reupload(src, 70, 0.25)
    al = pd.align(il.gray(up), il.gray(src))
    assert al["ok"] and 3.8 < al["scale"] < 4.2

    res = pd.compare(up, src, al["H"])
    lim = paths.config("genuine_ranges")
    assert res["ok"]
    for metric in ("block_diff_max", "detail_dev_max", "block_chroma_diff_max"):
        assert res[metric] <= lim[f"diff_picture.{metric}"]["range"][1]


def test_gate_scores_separate_copies_from_strangers():
    """The measured gap that the Gate 1 threshold sits in."""
    thr = paths.config("gates")["image"]["trusted_threshold"]
    src = V.sample("rocket")
    copy_score = float(il.embed_array(V.reupload(src, 40, 0.6)) @ il.embed_array(src))
    stranger = max(float(il.embed_array(V.sample("retina")) @ il.embed(paths.library_entry_dir("trusted", "image", n) / "original.png"))
                   for n in LIBRARY)
    assert copy_score > thr > stranger


def test_sscd_backend_requires_a_checkpoint_and_never_falls_back(monkeypatch, tmp_path):
    from modules._common import sscd

    monkeypatch.setenv("SOURCEDIFF_SSCD_MODEL", str(tmp_path / "missing-model.pt"))
    with pytest.raises(FileNotFoundError, match="download_sscd.py"):
        sscd.embed_array(V.sample("camera"))


def test_sscd_provider_dimension_is_supported_by_indexes():
    assert il.embedding_dimension("sscd_disc_mixup") == 512

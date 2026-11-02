"""The library tool: supported media goes to image/ or video/ by file extension."""
import cv2
import numpy as np
import pytest

import library_cli as cli
from sourcediff import paths


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_LIBRARY", str(tmp_path / "library"))
    monkeypatch.setenv("SOURCEDIFF_DATA", str(tmp_path / "data"))
    return tmp_path / "library"


def jpg(path, seed=0):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
    return path


def video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (96, 72))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable")
    rng = np.random.default_rng(7)
    for _ in range(4):
        writer.write(rng.integers(0, 255, (72, 96, 3), dtype=np.uint8))
    writer.release()
    return path


def test_init_creates_only_image_and_video_subfolders(lib):
    cli.main(["init"])
    for kind in ("trusted", "fake"):
        for door in ("image", "video"):
            assert (lib / kind / door).is_dir()
    assert (lib / "index").is_dir()


def test_add_trusted_routes_by_extension_and_writes_meta(lib, tmp_path):
    files = {"photo.jpg": "image", "clip.mp4": "video", "pic.PNG": "image"}
    for name, door in files.items():
        src = tmp_path / name
        src.write_bytes(b"x")
        d = cli.add_trusted(src, f"id-{name.lower().replace('.', '-')}", "Title", "Pub", "http://u", "CC0")
        assert d.parent.name == door and d.parent.parent.name == "trusted"
        meta = paths.load_json(d / "meta.json")
        assert meta["door"] == door and meta["publisher"] == "Pub" and meta["license"] == "CC0" and meta["url"] == "http://u"
        assert (d / meta["file"]).exists()


def test_standalone_audio_is_rejected(lib, tmp_path):
    src = tmp_path / "voice.wav"
    src.write_bytes(b"audio bytes")
    with pytest.raises(Exception):
        cli.add_trusted(src, "voice", "Voice")


def test_duplicate_id_is_refused(lib, tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"x")
    cli.add_trusted(src, "same", "t")
    with pytest.raises(SystemExit):
        cli.add_trusted(src, "same", "t")


def test_unsupported_extension_is_refused(lib, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("hi")
    with pytest.raises(Exception):
        cli.add_trusted(src, "n", "t")


def test_add_folder_bulk_skips_unsupported_and_makes_ids_unique(lib, tmp_path):
    inc = tmp_path / "incoming"
    inc.mkdir()
    (inc / "Press Clip One.mp4").write_bytes(b"v")
    jpg(inc / "Photo A.jpg")
    jpg(inc / "Photo A.png")                                  # same stem, same door: needs a unique id
    (inc / "notes.txt").write_text("skip me")
    (inc / "subfolder").mkdir()
    added, skipped = cli.add_folder(inc, publisher="P", license="L")
    assert sorted(added) == ["image/photo-a", "image/photo-a-2", "video/press-clip-one"]
    assert skipped == ["notes.txt"]


def test_build_index_makes_images_and_videos_searchable(lib, tmp_path, capsys):
    inc = tmp_path / "incoming"
    inc.mkdir()
    jpg(inc / "one.jpg", 1)
    jpg(inc / "two.jpg", 2)
    video(inc / "clip.avi")
    cli.add_folder(inc)
    cli.build_index()
    out = capsys.readouterr().out
    assert "2 image(s) indexed" in out and "4 frame(s) indexed" in out
    from modules._common import imagelib as il, videolib as vl
    hits = il.search("trusted", il.embed(inc / "one.jpg"), k=2)
    assert hits[0][0] == "one" and hits[0][1] > 0.99
    samples, _ = vl.sample_frames(inc / "clip.avi")
    assert vl.search("trusted", il.embed_array(samples[0][1]))[0] == "clip"


def test_build_index_supports_sscd_dimensions_for_images_and_video(lib, tmp_path, monkeypatch):
    import json
    import shutil

    import faiss
    from modules._common import imagelib as il, videolib as vl

    config_dir = tmp_path / "config-sscd"
    shutil.copytree(paths.config_dir(), config_dir)
    gates_file = config_dir / "gates.json"
    gates = json.loads(gates_file.read_text(encoding="utf-8"))
    gates["image"]["embedder"] = "sscd_disc_mixup"
    gates["video"]["embedder"] = "sscd_disc_mixup"
    gates_file.write_text(json.dumps(gates), encoding="utf-8")
    monkeypatch.setenv("SOURCEDIFF_CONFIG", str(config_dir))

    inc = tmp_path / "sscd-inputs"
    inc.mkdir()
    jpg(inc / "photo.jpg")
    video(inc / "clip.avi")
    cli.add_folder(inc)
    monkeypatch.setattr(
        il, "embed_array",
        lambda _img, _embedder=None: np.ones(512, dtype=np.float32) / np.sqrt(512),
    )

    cli.build_index()

    image_index = faiss.read_index(str(il.index_files("trusted")[0]))
    video_index = faiss.read_index(str(vl.index_files("trusted")[0]))
    assert image_index.d == video_index.d == 512
    assert paths.load_json(il.index_files("trusted")[1])["embedder"] == "sscd_disc_mixup"
    assert paths.load_json(vl.index_files("trusted")[1])["embedder"] == "sscd_disc_mixup"
    assert il.search("trusted", np.ones(512, dtype=np.float32) / np.sqrt(512))[0][0] == "photo"
    assert vl.search("trusted", np.ones(512, dtype=np.float32) / np.sqrt(512))[0] == "clip"


def test_migrate_moves_a_flat_library_and_is_safe_to_repeat(lib, tmp_path):
    flat = lib / "trusted" / "old-one"
    flat.mkdir(parents=True)
    jpg(flat / "original.jpg")
    paths.write_json(flat / "meta.json", {"source_id": "old-one", "file": "original.jpg", "door": "image"})
    vflat = lib / "trusted" / "old-vid"
    vflat.mkdir()
    (vflat / "original.mp4").write_bytes(b"v")
    paths.write_json(vflat / "meta.json", {"source_id": "old-vid", "file": "original.mp4", "door": "video"})
    moved = cli.migrate_layout()
    assert len(moved) == 2
    assert (lib / "trusted" / "image" / "old-one" / "original.jpg").exists()
    assert (lib / "trusted" / "video" / "old-vid" / "original.mp4").exists()
    assert cli.migrate_layout() == []


def test_list_prints_each_media_folder(lib, tmp_path, capsys):
    src = tmp_path / "a.jpg"
    jpg(src)
    cli.add_trusted(src, "alpha", "t")
    cli.main(["list"])
    out = capsys.readouterr().out
    assert "trusted/image (1): alpha" in out and "trusted/video (0)" in out

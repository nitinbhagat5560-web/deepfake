import shutil

import cv2
import numpy as np
import pytest

import _variants as V
from modules._common import imagelib, videolib
from sourcediff import library, paths


@pytest.fixture
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_LIBRARY", str(tmp_path / "library"))
    config_dir = tmp_path / "config"
    shutil.copytree(paths.config_dir(), config_dir)
    monkeypatch.setenv("SOURCEDIFF_CONFIG", str(config_dir))
    return tmp_path


def test_adding_images_appends_to_the_existing_index(isolated_library, tmp_path, monkeypatch):
    images = [V.sample("camera"), V.sample("astronaut")]
    sources = []
    for i, image in enumerate(images):
        source = tmp_path / f"source-{i}.jpg"
        assert cv2.imwrite(str(source), image)
        sources.append(source)

    full_build = imagelib.build_index
    builds = []

    def track_build(kind):
        builds.append(kind)
        return full_build(kind)

    monkeypatch.setattr(imagelib, "build_index", track_build)
    library.add_item(sources[0], sources[0].name, "First", source_id="first",
                     strict=False, rebuild=True)
    library.add_item(sources[1], sources[1].name, "Second", source_id="second",
                     strict=False, rebuild=True)

    metadata = paths.load_json(imagelib.index_files("trusted")[1])
    assert builds == ["trusted"]
    assert metadata["ids"] == ["first", "second"]
    assert imagelib.search("trusted", imagelib.embed(sources[1]), k=1)[0][0] == "second"


def test_adding_videos_appends_to_the_existing_index(isolated_library, tmp_path, monkeypatch):
    sources = []
    for i in range(2):
        source = tmp_path / f"source-{i}.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (320, 240))
        if not writer.isOpened():
            writer.release()
            pytest.skip("OpenCV MJPG video writer is unavailable")
        frame = np.random.default_rng(i).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
        writer.write(frame)
        writer.release()
        sources.append(source)

    full_build = videolib.build_index
    builds = []

    def track_build(kind="trusted"):
        builds.append(kind)
        return full_build(kind)

    monkeypatch.setattr(videolib, "build_index", track_build)
    library.add_item(sources[0], sources[0].name, "First", source_id="first",
                     strict=False, rebuild=True)
    library.add_item(sources[1], sources[1].name, "Second", source_id="second",
                     strict=False, rebuild=True)

    metadata = paths.load_json(videolib.index_files("trusted")[1])
    assert builds == ["trusted"]
    assert {ref["source_id"] for ref in metadata["ids"]} == {"first", "second"}
    query = imagelib.embed_array(next(videolib.iter_frames_at(sources[1], [0.0])), videolib.embedder())
    assert videolib.search_many("trusted", query.reshape(1, -1))[0][0] == "second"

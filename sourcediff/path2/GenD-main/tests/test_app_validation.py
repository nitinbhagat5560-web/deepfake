import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gend_app_run", ROOT / "app" / "run.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class FakeDetector:
    def infer_faces(self, *args, **kwargs):
        return [(np.array([10, 20, 30, 40]), 0.25)]

    def infer_whole_frame(self, *args, **kwargs):
        return 0.73

    def annotate_frame(self, frame, faces, avg_fake=None, backend=None, p_ai_score=None):
        return frame.copy()


class FakeDetectorNoFaces:
    def infer_faces(self, *args, **kwargs):
        return []

    def infer_whole_frame(self, *args, **kwargs):
        return 0.61

    def annotate_frame(self, frame, faces, avg_fake=None, backend=None, p_ai_score=None):
        return frame.copy()


class UtilityTests(unittest.TestCase):
    def test_model_priority_and_default(self):
        self.assertEqual(app.HF_MODELS[0], "yermandy/GenD_PE_L")
        self.assertEqual(app.HF_MODELS[-1], "yermandy/GenD_DINOv3_L")

    def test_rgb_conversion_formats(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(app.rgb_to_bgr(rgb).shape, (8, 8, 3))
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        self.assertEqual(app.rgb_to_bgr(rgba).shape, (8, 8, 3))
        gray = np.zeros((8, 8), dtype=np.uint8)
        self.assertEqual(app.rgb_to_bgr(gray).shape, (8, 8, 3))
        floats = np.zeros((8, 8, 3), dtype=np.float32)
        self.assertEqual(app.rgb_to_bgr(floats).shape, (8, 8, 3))

    def test_ui_builds_with_model_priority(self):
        self.assertIsNotNone(app.build_ui())

    def test_collect_and_classify_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "sample.png"
            video = root / "sample.mp4"
            image.write_bytes(b"not-used")
            video.write_bytes(b"not-used")
            upload = SimpleNamespace(name=str(image))
            paths = app.collect_inputs([upload, str(video)], None)
            self.assertEqual(set(paths), {str(image), str(video)})
            self.assertTrue(app.is_image(str(image)))
            self.assertTrue(app.is_video(str(video)))

    def test_invalid_run_settings(self):
        generator = app.run_inference("yermandy/GenD_CLIP_L_14", None, 0.5, 1, -1, 1.3, -1, 1, "Off", progress=lambda *args, **kwargs: None)
        first = next(generator)
        status = first[1]
        self.assertIn("Invalid settings", status)
        self.assertIn("No input files", status)


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_process_image_with_faces_and_whole_frame(self):
        source = self.out_dir / "source.png"
        cv2.imwrite(str(source), np.full((64, 64, 3), 127, dtype=np.uint8))
        processor = app.MediaProcessor(FakeDetector())
        out, metrics = processor.process_image(
            str(source), None, None, None, torch.float32, 1.3, None,
            self.out_dir, max_faces=1, whole_frame="GenD"
        )
        self.assertTrue(Path(out).is_file())
        self.assertEqual(metrics["status"], "Face and whole-frame analyzed")
        self.assertEqual(metrics["num_faces"], 1.0)
        self.assertEqual(metrics["avg_p_ai"], 0.73)

    def test_process_image_no_faces(self):
        source = self.out_dir / "empty.png"
        cv2.imwrite(str(source), np.full((64, 64, 3), 127, dtype=np.uint8))
        processor = app.MediaProcessor(FakeDetectorNoFaces())
        out, metrics = processor.process_image(
            str(source), None, None, None, torch.float32, 1.3, None,
            self.out_dir, max_faces=1, whole_frame="GenD"
        )
        self.assertTrue(Path(out).is_file())
        self.assertEqual(metrics["status"], "Whole-frame analyzed; no face detected")
        self.assertEqual(metrics["num_faces"], 0.0)

    def test_process_video(self):
        source = self.out_dir / "source.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
        for value in (50, 100, 150):
            writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
        writer.release()
        self.assertTrue(source.is_file() and source.stat().st_size > 0)

        processor = app.MediaProcessor(FakeDetector())
        out, metrics = processor.process_video(
            str(source), None, None, None, torch.float32, 1.3, None,
            self.out_dir, stride=1, max_frames=2, max_faces=1, whole_frame=False
        )
        self.assertTrue(Path(out).is_file())
        self.assertEqual(metrics["num_frames"], 2.0)
        self.assertEqual(metrics["num_faces"], 2.0)
        self.assertEqual(metrics["status"], "Analyzed")


if __name__ == "__main__":
    unittest.main(verbosity=2)

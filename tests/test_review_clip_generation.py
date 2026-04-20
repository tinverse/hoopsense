import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


cv2_stub = types.ModuleType("cv2")
sys.modules.setdefault("cv2", cv2_stub)

from tools.data import clip_slicer, sample_rohan_clips


class _FakeCapture:
    def __init__(self, fps=30.0, frame_count=3600, opened=True):
        self._fps = fps
        self._frame_count = frame_count
        self._opened = opened

    def isOpened(self):
        return self._opened

    def get(self, prop):
        if prop == cv2_stub.CAP_PROP_FPS:
            return self._fps
        if prop == cv2_stub.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        return 0

    def release(self):
        return None


class ReviewClipGenerationTest(unittest.TestCase):
    def setUp(self):
        cv2_stub.CAP_PROP_FPS = 5
        cv2_stub.CAP_PROP_FRAME_COUNT = 7

    def test_sample_rohan_clips_defaults_to_minute_scale_duration(self):
        self.assertEqual(sample_rohan_clips.DEFAULT_REVIEW_CLIP_DURATION_S, 60.0)

    def test_get_single_clip_uses_requested_duration_in_ffmpeg_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake")
            output_dir = Path(tmpdir) / "out"
            fake_capture = _FakeCapture(fps=30.0, frame_count=3600)
            with mock.patch.object(sample_rohan_clips.cv2, "VideoCapture", return_value=fake_capture, create=True):
                with mock.patch.object(sample_rohan_clips.random, "randint", return_value=0):
                    with mock.patch.object(sample_rohan_clips.subprocess, "run") as run_mock:
                        ok = sample_rohan_clips.get_single_clip(video_path, output_dir, 0, duration=75.0)
            self.assertTrue(ok)
            ffmpeg_cmd = run_mock.call_args.args[0]
            self.assertEqual(ffmpeg_cmd[0], "ffmpeg")
            self.assertEqual(ffmpeg_cmd[5], "75.0")

    def test_clip_slicer_defaults_to_minute_scale_segments(self):
        self.assertEqual(clip_slicer.DEFAULT_REVIEW_SEGMENT_DURATION_S, 60.0)

    def test_clip_slicer_uses_segment_duration_to_step_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake")
            fake_capture = _FakeCapture(fps=30.0, frame_count=5400)
            with mock.patch.object(clip_slicer.cv2, "VideoCapture", return_value=fake_capture, create=True):
                with mock.patch("builtins.print") as print_mock:
                    clip_slicer.slice_video(video_path, Path(tmpdir) / "out", "youth", segment_duration=60.0)
            lines = [call.args[0] for call in print_mock.call_args_list if "Generated" in call.args[0]]
            self.assertEqual(len(lines), 3)
            self.assertIn("Frame 0 to 1800", lines[0])
            self.assertIn("Frame 1800 to 3600", lines[1])
            self.assertIn("Frame 3600 to 5400", lines[2])


if __name__ == "__main__":
    unittest.main()

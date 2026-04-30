import unittest

import numpy as np

from unittest import mock

from tools.review.labeller.render_perception_overlay_video import (
    build_detection_label,
    build_docker_ffmpeg_command,
    frame_summary,
    mask_from_grid,
    resolve_docker_ffmpeg_command,
)


class RenderPerceptionOverlayVideoTest(unittest.TestCase):
    def test_build_detection_label_matches_overlay_semantics(self):
        detection = {
            "track_id": 7,
            "identity_track_id": 11,
            "identity_jersey_number": "24",
            "identity_jersey_number_confidence": 0.83,
            "uniform_bucket": "light",
            "active_player_candidate": True,
            "confidence": 0.91,
            "on_court_score": 0.88,
            "active_player_score": 0.77,
            "motion_speed_px": 5.4,
        }
        label = build_detection_label(detection)
        self.assertIn("P7/I11", label)
        self.assertIn("#24(83)", label)
        self.assertIn("LIGHT ACTIVE", label)
        self.assertIn("91%", label)
        self.assertIn("C88", label)
        self.assertIn("A77", label)
        self.assertIn("M5.4", label)

    def test_frame_summary_includes_counts_and_ball_state(self):
        frame = {
            "frame_idx": 12,
            "t_ms": 500.0,
            "live_play_label": "live",
            "live_play_score": 0.9,
            "discontinuity_label": "continuous",
            "discontinuity_score": 0.0,
            "continuity_segment_id": 2,
            "detections": [
                {"on_court_candidate": True, "active_player_candidate": True},
                {"on_court_candidate": True, "active_player_candidate": False},
                {"on_court_candidate": False, "active_player_candidate": False},
            ],
            "ball_state": {"state": "observed", "confidence": 0.62},
        }
        summary = frame_summary(frame)
        self.assertIn("f12", summary)
        self.assertIn("det=3 on=2 act=1 raw=1", summary)
        self.assertIn("live=live 0.90", summary)
        self.assertIn("seg=2", summary)
        self.assertIn("ball observed 62%", summary)

    def test_mask_from_grid_scales_to_frame_shape(self):
        mask = mask_from_grid([[1, 0], [0, 1]], (20, 30, 3))
        self.assertEqual(mask.shape, (20, 30))
        self.assertEqual(int(mask[0, 0]), 1)
        self.assertEqual(int(mask[-1, -1]), 1)
        self.assertEqual(int(mask[0, -1]), 0)

    def test_build_docker_ffmpeg_command_mounts_input_and_output_dirs(self):
        cmd = build_docker_ffmpeg_command('/tmp/raw/input.mp4', '/tmp/out/output.mp4', image='demo:image')
        self.assertEqual(cmd[:3], ['docker', 'run', '--rm'])
        self.assertIn('/tmp/raw:/input', cmd)
        self.assertIn('/tmp/out:/output', cmd)
        self.assertIn('demo:image', cmd)
        self.assertIn('/input/input.mp4', cmd[-1])
        self.assertIn('/output/output.mp4', cmd[-1])

    @mock.patch('tools.review.labeller.render_perception_overlay_video.subprocess.run')
    @mock.patch('tools.review.labeller.render_perception_overlay_video.shutil.which', return_value='/usr/bin/docker')
    def test_resolve_docker_ffmpeg_command_requires_available_image(self, _which, run_mock):
        run_mock.return_value = mock.Mock(returncode=0)
        cmd = resolve_docker_ffmpeg_command('/tmp/raw/input.mp4', '/tmp/out/output.mp4', image='demo:image')
        self.assertIsNotNone(cmd)
        self.assertEqual(run_mock.call_args.args[0], ['/usr/bin/docker', 'image', 'inspect', 'demo:image'])

    @mock.patch('tools.review.labeller.render_perception_overlay_video.subprocess.run')
    @mock.patch('tools.review.labeller.render_perception_overlay_video.shutil.which', return_value='/usr/bin/docker')
    def test_resolve_docker_ffmpeg_command_returns_none_when_image_missing(self, _which, run_mock):
        run_mock.return_value = mock.Mock(returncode=1)
        cmd = resolve_docker_ffmpeg_command('/tmp/raw/input.mp4', '/tmp/out/output.mp4', image='demo:image')
        self.assertIsNone(cmd)


if __name__ == '__main__':
    unittest.main()

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


LABELLER_APP_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "review" / "labeller" / "app.py"
)


cv2_stub = types.ModuleType("cv2")
sys.modules.setdefault("cv2", cv2_stub)


def load_labeller_module():
    spec = importlib.util.spec_from_file_location("labeller_app", LABELLER_APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LabellerOverlayTest(unittest.TestCase):
    def test_clip_list_only_returns_clips_with_perception_artifacts(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            module.CLIPS_DIR = tmp / "clips"
            module.PERCEPTION_DIR = tmp / "artifacts"
            (module.CLIPS_DIR / "youth").mkdir(parents=True)
            module.PERCEPTION_DIR.mkdir(parents=True)
            (module.CLIPS_DIR / "youth" / "with_overlay.mp4").write_bytes(b"")
            (module.CLIPS_DIR / "youth" / "without_overlay.mp4").write_bytes(b"")
            (module.PERCEPTION_DIR / "with_overlay.perception.json").write_text(json.dumps({
                "clip_id": "with_overlay",
                "frames": [],
            }))
            client = module.app.test_client()
            response = client.get("/api/clips")
            payload = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual([clip["id"] for clip in payload], ["with_overlay"])

    def test_calibration_primitives_payload_exposes_line_and_arc_families(self):
        module = load_labeller_module()
        client = module.app.test_client()
        response = client.get("/api/landmarks")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["left_baseline"]["family"], "baseline")
        self.assertEqual(payload["left_three_point_arc"]["kind"], "arc")

    def test_missing_overlay_returns_disabled_payload(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            module.PERCEPTION_DIR = Path(tmpdir)
            client = module.app.test_client()
            response = client.get("/api/perception/no_such_clip")
            payload = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertFalse(payload["enabled"])
            self.assertEqual(payload["status"], "missing")

    def test_existing_overlay_payload_is_returned(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            module.PERCEPTION_DIR = Path(tmpdir)
            artifact = {
                "schema_version": "1.0.0",
                "clip_id": "demo_clip",
                "video_path": "youth/demo_clip.mp4",
                "video": {"fps": 24.0, "frame_count": 3, "width": 1280, "height": 720},
                "model": {"name": "yolov8n-pose.pt", "task": "pose_track", "device": "cuda:0"},
                "frames": [{"frame_idx": 0, "t_ms": 0, "calibrated": False, "detections": []}],
            }
            (module.PERCEPTION_DIR / "demo_clip.perception.json").write_text(json.dumps(artifact))
            client = module.app.test_client()
            response = client.get("/api/perception/demo_clip")
            payload = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["enabled"])
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["clip_id"], "demo_clip")
            self.assertEqual(payload["frames"][0]["frame_idx"], 0)

    def test_mismatched_overlay_clip_id_is_rejected(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            module.PERCEPTION_DIR = Path(tmpdir)
            artifact = {
                "schema_version": "1.0.0",
                "clip_id": "other_clip",
                "frames": [],
            }
            (module.PERCEPTION_DIR / "demo_clip.perception.json").write_text(json.dumps(artifact))
            client = module.app.test_client()
            response = client.get("/api/perception/demo_clip")
            payload = response.get_json()
            self.assertEqual(response.status_code, 409)
            self.assertFalse(payload["enabled"])
            self.assertEqual(payload["status"], "clip_id_mismatch")

    def test_perception_feedback_is_appended(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            module.PERCEPTION_DIR = Path(tmpdir)
            module.PERCEPTION_FEEDBACK_FILE = Path(tmpdir) / "perception_feedback.jsonl"
            artifact = {
                "schema_version": "1.0.0",
                "clip_id": "demo_clip",
                "frames": [{
                    "frame_idx": 12,
                    "detections": [{"track_id": 7}],
                }],
            }
            (module.PERCEPTION_DIR / "demo_clip.perception.json").write_text(json.dumps(artifact))
            client = module.app.test_client()
            response = client.post(
                "/api/perception_feedback",
                json={
                    "clip_id": "demo_clip",
                    "frame_idx": 12,
                    "t_ms": 500,
                    "issue_type": "false_positive",
                    "track_id": "7",
                    "note": "sideline player",
                    "timestamp": "2026-03-13T12:00:00Z",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "success")
            lines = module.PERCEPTION_FEEDBACK_FILE.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            saved = json.loads(lines[0])
            self.assertEqual(saved["issue_type"], "false_positive")
            self.assertEqual(saved["track_id"], "7")

    def test_general_note_feedback_is_appended_without_track(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            module.PERCEPTION_DIR = Path(tmpdir)
            module.PERCEPTION_FEEDBACK_FILE = Path(tmpdir) / "perception_feedback.jsonl"
            artifact = {
                "schema_version": "1.0.0",
                "clip_id": "demo_clip",
                "frames": [{
                    "frame_idx": 0,
                    "detections": [{"track_id": 6}],
                }],
            }
            (module.PERCEPTION_DIR / "demo_clip.perception.json").write_text(json.dumps(artifact))
            client = module.app.test_client()
            response = client.post(
                "/api/perception_feedback",
                json={
                    "clip_id": "demo_clip",
                    "frame_idx": 0,
                    "t_ms": 0,
                    "issue_type": "general_note",
                    "note": "P6 is a seated spectator in frame 0. Missed cluster right of P3 through frame 16.",
                    "timestamp": "2026-03-30T15:00:00Z",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "success")
            lines = module.PERCEPTION_FEEDBACK_FILE.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            saved = json.loads(lines[0])
            self.assertEqual(saved["issue_type"], "general_note")
            self.assertTrue("track_id" not in saved or saved["track_id"] is None)

    def test_invalid_feedback_issue_type_is_rejected(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            module.PERCEPTION_DIR = Path(tmpdir)
            module.PERCEPTION_FEEDBACK_FILE = Path(tmpdir) / "perception_feedback.jsonl"
            artifact = {
                "schema_version": "1.0.0",
                "clip_id": "demo_clip",
                "frames": [],
            }
            (module.PERCEPTION_DIR / "demo_clip.perception.json").write_text(json.dumps(artifact))
            client = module.app.test_client()
            response = client.post(
                "/api/perception_feedback",
                json={
                    "clip_id": "demo_clip",
                    "frame_idx": 12,
                    "t_ms": 500,
                    "issue_type": "made_up_issue",
                    "timestamp": "2026-03-13T12:00:00Z",
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["status"], "error")

    def test_missing_feedback_track_id_is_rejected(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            module.PERCEPTION_DIR = Path(tmpdir)
            module.PERCEPTION_FEEDBACK_FILE = Path(tmpdir) / "perception_feedback.jsonl"
            artifact = {
                "schema_version": "1.0.0",
                "clip_id": "demo_clip",
                "frames": [{
                    "frame_idx": 12,
                    "detections": [{"track_id": 7}],
                }],
            }
            (module.PERCEPTION_DIR / "demo_clip.perception.json").write_text(json.dumps(artifact))
            client = module.app.test_client()
            response = client.post(
                "/api/perception_feedback",
                json={
                    "clip_id": "demo_clip",
                    "frame_idx": 12,
                    "t_ms": 500,
                    "issue_type": "track_error",
                    "track_id": "999",
                    "timestamp": "2026-03-13T12:00:00Z",
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["reason"], "track_id_missing")

    def test_partial_court_primitive_calibration_is_saved_without_corners(self):
        module = load_labeller_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            module.CLIPS_DIR = tmp / "clips"
            module.CALIBRATION_FILE = tmp / "camera_calibration.json"
            clip_dir = module.CLIPS_DIR / "youth"
            clip_dir.mkdir(parents=True)
            (clip_dir / "demo_clip.mp4").write_bytes(b"")

            class FakeCapture:
                def __init__(self, *_args, **_kwargs):
                    pass
                def get(self, prop):
                    if prop == module.cv2.CAP_PROP_FPS:
                        return 30.0
                    if prop == module.cv2.CAP_PROP_FRAME_COUNT:
                        return 3
                    return 0
                def release(self):
                    return None

            module.cv2.VideoCapture = FakeCapture
            module.cv2.CAP_PROP_FPS = 5
            module.cv2.CAP_PROP_FRAME_COUNT = 7
            module.cv2.findHomography = lambda img, world: (module.np.eye(3), None)
            module.track_landmarks = lambda _path, start_idx, initial_pts: {start_idx: initial_pts}

            client = module.app.test_client()
            response = client.post(
                "/api/calibrate",
                json={
                    "id": "demo_clip",
                    "path": "youth/demo_clip.mp4",
                    "points": [
                        {"point_key": "a", "primitive_id": "far_sideline", "x": 10, "y": 20, "t_ms": 0, "sample_order": 0},
                        {"point_key": "b", "primitive_id": "far_sideline", "x": 20, "y": 20, "t_ms": 0, "sample_order": 1},
                        {"point_key": "c", "primitive_id": "left_baseline", "x": 10, "y": 40, "t_ms": 0, "sample_order": 2},
                        {"point_key": "d", "primitive_id": "left_three_point_arc", "x": 40, "y": 30, "t_ms": 0, "sample_order": 3},
                        {"point_key": "e", "primitive_id": "left_three_point_arc", "x": 42, "y": 32, "t_ms": 0, "sample_order": 4},
                    ],
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["status"], "success")
            self.assertIn("sideline", payload["primitive_families"])
            self.assertIn("baseline", payload["primitive_families"])
            saved = json.loads(module.CALIBRATION_FILE.read_text())
            self.assertEqual(saved["demo_clip"]["type"], "temporal_aggregation_partial_court")
            self.assertEqual(saved["demo_clip"]["solver"], "primitive_sample_correspondence_v1")
            self.assertIn("three_point_arc", saved["demo_clip"]["primitive_families"])


if __name__ == "__main__":
    unittest.main()

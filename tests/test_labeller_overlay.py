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


if __name__ == "__main__":
    unittest.main()

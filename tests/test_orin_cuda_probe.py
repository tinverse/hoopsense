import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_orin_cuda_probe


class OrinCudaProbeTest(unittest.TestCase):
    def test_build_report_fails_when_cuda_checks_fail(self):
        report = run_orin_cuda_probe.build_report(
            {
                "python_executable": "/tmp/python",
                "platform": "Linux",
                "machine": "x86_64",
                "torch_version": "2.0",
                "torch_file": "/tmp/site-packages/torch/__init__.py",
                "torch_ok": True,
                "numpy_ok": True,
                "cuda_available": False,
                "cuda_device_name": None,
                "cuda_tensor_ok": False,
                "errors": [],
            },
            {
                "device_type": "cpu",
                "action_brain_ok": False,
                "action_brain_device": None,
                "action_brain_latency_ms": None,
                "pose_association_ok": None,
                "dsl_rule_ok": None,
                "rust_bridge_ok": None,
            },
        )
        self.assertEqual(report["status"], "fail")

    def test_main_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts"
            md_path = Path(tmpdir) / "ORIN_VALIDATION_REPORT.md"
            env_result = {
                "python_executable": "/tmp/python",
                "platform": "Linux",
                "machine": "aarch64",
                "torch_version": "2.0",
                "torch_file": "/tmp/site-packages/torch/__init__.py",
                "torch_ok": True,
                "numpy_ok": True,
                "cuda_available": True,
                "cuda_device_name": "Jetson GPU",
                "cuda_tensor_ok": True,
                "errors": [],
            }
            probe_result = {
                "device_type": "cuda",
                "cuda_tensor_ok": True,
                "action_brain_ok": True,
                "action_brain_device": "cuda",
                "action_brain_latency_ms": 3.25,
                "pose_association_ok": True,
                "dsl_rule_ok": True,
                "rust_bridge_ok": True,
                "rust_bridge_stderr": None,
            }
            with mock.patch.object(run_orin_cuda_probe, "OUTPUT_DIR", output_dir), \
                 mock.patch.object(run_orin_cuda_probe, "JSON_PATH", output_dir / "latest.json"), \
                 mock.patch.object(run_orin_cuda_probe, "MD_PATH", md_path), \
                 mock.patch.object(run_orin_cuda_probe, "validate_environment", return_value=env_result), \
                 mock.patch.object(run_orin_cuda_probe, "run_rigorous_probe", return_value=probe_result):
                rc = run_orin_cuda_probe.main()
            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / "latest.json").exists())
            self.assertTrue(md_path.exists())
            saved = json.loads((output_dir / "latest.json").read_text())
            self.assertEqual(saved["status"], "pass")
            self.assertIn("Overall Status", md_path.read_text())


if __name__ == "__main__":
    unittest.main()

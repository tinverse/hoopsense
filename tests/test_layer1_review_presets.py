import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.review.labeller import layer1_review_presets as presets


class Layer1ReviewPresetTest(unittest.TestCase):
    def test_plain_generation_scripts_default_to_grounding_dino(self):
        native_script = (presets.REPO_ROOT / "scripts/generate_layer1_annotations.sh").read_text()
        docker_script = (presets.REPO_ROOT / "scripts/generate_layer1_annotations_docker.sh").read_text()
        for script in (native_script, docker_script):
            self.assertIn("HOOPSENSE_LAYER1_GROUNDING_DINO:-1", script)
            self.assertIn("--bootstrap-foreground-backend grounding_dino", script)
            self.assertIn("basketball hoop", script)
            self.assertIn("has_bootstrap_backend", script)

    def test_load_grounded_sam3_review_preset(self):
        preset = presets.get_preset("grounded_sam3_review")
        args = preset["generator_args"]
        self.assertEqual(preset["runner"], "docker")
        self.assertEqual(args["player_tracker_backend"], "default")
        self.assertEqual(args["bootstrap_foreground_backend"], "grounding_dino")
        self.assertEqual(args["player_recovery_backend"], "sam3")
        self.assertEqual(args["ball_bootstrap_backend"], "sam3")
        self.assertIn("basketball", args["bootstrap_foreground_prompt"])

    def test_build_generation_command_uses_docker_and_expected_flags(self):
        preset = presets.get_preset("grounded_sam3_review")
        command = presets.build_generation_command(
            presets.REPO_ROOT / "data/raw_clips/youth/demo.mp4",
            preset,
            runner="docker",
            output_path=presets.REPO_ROOT / "tmp_runs/demo.perception.json",
        )
        command_text = " ".join(command)
        self.assertIn("run_orin_layer1_docker.sh", command[0])
        self.assertIn("--player-tracker-backend default", command_text)
        self.assertIn("--bootstrap-foreground-backend grounding_dino", command_text)
        self.assertIn("--player-recovery-backend sam3", command_text)
        self.assertIn("--ball-bootstrap-backend sam3", command_text)
        self.assertIn("--output", command)
        self.assertEqual(command[-2:], ["--device", "cuda:0"])

    def test_run_preset_dry_run_writes_manifest_without_executing_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            output = Path(tmpdir) / "artifact.json"
            args = type("Args", (), {})()
            args.clip_path = "data/raw_clips/youth/demo.mp4"
            args.preset = "grounded_sam3_review"
            args.runner = "docker"
            args.output = str(output)
            args.run_dir = str(run_dir)
            args.dry_run = True
            args.extra_generator_args = []

            result = presets.run_preset(args)

            self.assertEqual(result, 0)
            manifest_path = run_dir / "run_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["kind"], "layer1_review_generation_run_manifest_v1")
            self.assertEqual(manifest["preset_name"], "grounded_sam3_review")
            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["runner"], "docker")
            self.assertEqual(manifest["output_path"], str(output.resolve()))
            self.assertIn("--bootstrap-foreground-backend", manifest["command"])

    def test_cli_consumes_preset_options_after_clip_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(presets.REPO_ROOT / "tools/review/labeller/layer1_review_presets.py"),
                    "data/raw_clips/youth/demo.mp4",
                    "--preset",
                    "grounded_sam3_review",
                    "--dry-run",
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=presets.REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertIn("Wrote run manifest", result.stdout)
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertTrue(manifest["dry_run"])
            self.assertNotIn("--preset", manifest["command"])
            self.assertNotIn("--dry-run", manifest["command"])



if __name__ == "__main__":
    unittest.main()

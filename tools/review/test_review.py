import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from review_change import build_review_prompt
from review_change import run_agent_review
from infra.gemini_project import GeminiProjectClient

class TestReviewInfrastructure(unittest.TestCase):
    def test_git_diff_dependency(self):
        """Ensures the script handles environments without git or staged changes."""
        try:
            diff = subprocess.check_output(["git", "diff", "--cached"]).decode("utf-8")
            self.assertIsInstance(diff, str)
        except subprocess.CalledProcessError:
            self.fail("review_change.py must be run within a git repository.")

    def test_prompt_generation(self):
        """Verifies that the agent prompt contains the necessary review criteria."""
        prompt = build_review_prompt("diff --git a/foo b/foo")
        self.assertIn("Findings first", prompt)
        self.assertIn("Design-First", prompt)
        self.assertIn("DIFF:", prompt)

    @patch("infra.gemini_project.subprocess.run")
    def test_run_agent_review_uses_persisted_project_session(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            state_dir = project_root / ".gemini"
            state_dir.mkdir()
            (state_dir / "project_session.json").write_text(
                json.dumps({"session_id": "session-123"}) + "\n"
            )

            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps(
                {"session_id": "session-123", "response": "Approved"}
            )
            mock_run.return_value.stderr = ""

            client = GeminiProjectClient(project_root=project_root)
            review = run_agent_review("diff --git a/foo b/foo", client)

            self.assertEqual(review, "Approved")
            command = mock_run.call_args.args[0]
            self.assertIn("--resume", command)
            self.assertIn("session-123", command)

if __name__ == "__main__":
    unittest.main()

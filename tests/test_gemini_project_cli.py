import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.infra.gemini_project import GeminiProjectClient
from tools.infra.gemini_project_cli import build_consult_prompt


class TestGeminiProjectCli(unittest.TestCase):
    def test_build_consult_prompt_mentions_topic(self):
        prompt = build_consult_prompt("architecture", "Review the tracking module boundary.")
        self.assertIn("Topic: architecture", prompt)
        self.assertIn("boundaries, contracts, components", prompt)
        self.assertIn("Review the tracking module boundary.", prompt)

    @patch("tools.infra.gemini_project.subprocess.run")
    def test_client_persists_project_session_id(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps(
                {"session_id": "session-abc", "response": "ok"}
            )
            mock_run.return_value.stderr = ""

            client = GeminiProjectClient(project_root=project_root)
            payload = client.ask("Bootstrap the session", resume=False)

            self.assertEqual(payload["session_id"], "session-abc")
            self.assertEqual(client.load_session_id(), "session-abc")


if __name__ == "__main__":
    unittest.main()

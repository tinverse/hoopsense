import unittest
from unittest.mock import patch

from tools.infra.gemini_collab_mcp import GeminiCollabConfig
from tools.infra.gemini_collab_mcp import GeminiCollabServer


class TestGeminiCollabMcp(unittest.TestCase):
    def setUp(self):
        self.server = GeminiCollabServer(
            GeminiCollabConfig(
                gemini_command="gemini",
                default_cwd="/data/projects/hoopsense",
            )
        )

    def test_initialize_reports_tool_capability(self):
        response = self.server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )

        self.assertEqual(response["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", response["result"]["capabilities"])
        self.assertEqual(response["result"]["serverInfo"]["name"], "hoopsense-gemini-collab")

    def test_tools_list_exposes_ask_gemini(self):
        response = self.server.handle_message(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )

        tool = response["result"]["tools"][0]
        self.assertEqual(tool["name"], "ask_gemini")
        self.assertIn("prompt", tool["inputSchema"]["properties"])

    @patch("tools.infra.gemini_collab_mcp.GeminiProjectClient.ask")
    def test_tools_call_uses_project_session_with_collaboration_prompt(self, mock_ask):
        mock_ask.return_value = {
            "session_id": "session-1",
            "response": "ok",
            "stats": {"tokens": 123},
        }

        response = self.server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "ask_gemini",
                    "arguments": {
                        "topic": "architecture",
                        "prompt": "Summarize the repo",
                        "model": "gemini-2.5-pro",
                        "output_format": "json",
                    },
                },
            }
        )

        mock_ask.assert_called_once()
        prompt_arg = mock_ask.call_args.args[0]
        self.assertIn("Senior Software Architect", prompt_arg)
        self.assertIn("Topic: architecture", prompt_arg)
        self.assertIn("Summarize the repo", prompt_arg)
        self.assertEqual(response["result"]["structuredContent"]["response"], "ok")
        self.assertEqual(response["result"]["structuredContent"]["session_id"], "session-1")
        self.assertNotIn("isError", response["result"])

    def test_tools_call_requires_prompt(self):
        response = self.server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "ask_gemini", "arguments": {}},
            }
        )

        self.assertTrue(response["result"]["isError"])
        self.assertIn("Missing required field: prompt", response["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()

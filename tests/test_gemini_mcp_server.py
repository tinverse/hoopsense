import unittest
from unittest.mock import patch

from tools.infra.gemini_mcp_server import GeminiBridgeConfig
from tools.infra.gemini_mcp_server import GeminiBridgeServer


class TestGeminiMcpServer(unittest.TestCase):
    def setUp(self):
        self.server = GeminiBridgeServer(GeminiBridgeConfig(gemini_command="gemini"))

    def test_initialize_reports_tool_capability(self):
        response = self.server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )

        self.assertEqual(response["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", response["result"]["capabilities"])
        self.assertEqual(response["result"]["serverInfo"]["name"], "hoopsense-gemini-bridge")

    def test_tools_list_exposes_ask_gemini(self):
        response = self.server.handle_message(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )

        tool = response["result"]["tools"][0]
        self.assertEqual(tool["name"], "ask_gemini")
        self.assertIn("prompt", tool["inputSchema"]["properties"])

    @patch("tools.infra.gemini_mcp_server.subprocess.run")
    def test_tools_call_shells_out_to_gemini(self, mock_run):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"answer":"ok"}'
        mock_run.return_value.stderr = ""

        response = self.server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "ask_gemini",
                    "arguments": {
                        "prompt": "Summarize the repo",
                        "model": "gemini-2.5-pro",
                        "approval_mode": "plan",
                        "output_format": "json",
                    },
                },
            }
        )

        mock_run.assert_called_once()
        command = mock_run.call_args.kwargs["args"] if "args" in mock_run.call_args.kwargs else mock_run.call_args.args[0]
        self.assertEqual(
            command,
            [
                "gemini",
                "--model",
                "gemini-2.5-pro",
                "--approval-mode",
                "plan",
                "--output-format",
                "json",
                "--prompt",
                "Summarize the repo",
            ],
        )
        self.assertEqual(response["result"]["structuredContent"]["stdout"], '{"answer":"ok"}')
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

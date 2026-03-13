import unittest
from unittest.mock import Mock

from tools.infra.gemini_mcp import GeminiMCPServer


class TestGeminiCollabMCP(unittest.TestCase):
    def test_tools_list_exposes_ask_gemini(self):
        server = object.__new__(GeminiMCPServer)
        response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(response["result"]["tools"][0]["name"], "ask_gemini")

    def test_tool_call_returns_structured_response(self):
        server = object.__new__(GeminiMCPServer)
        server.client = Mock()
        server.project_root = "."
        server.client.ask.return_value = {
            "response": "review ok",
            "session_id": "session-1",
            "stats": {"tokens": 10},
        }

        response = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "ask_gemini",
                    "arguments": {
                        "topic": "review",
                        "prompt": "Review the current diff.",
                    },
                },
            }
        )

        self.assertFalse(response["result"].get("isError", False))
        structured = response["result"]["structuredContent"]
        self.assertEqual(structured["response"], "review ok")
        self.assertEqual(structured["session_id"], "session-1")
        self.assertEqual(structured["topic"], "review")


if __name__ == "__main__":
    unittest.main()

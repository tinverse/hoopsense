import json
import unittest
from unittest.mock import patch

from tools.infra.chat_with_gemini import chat


class TestChatWithGemini(unittest.TestCase):
    @patch("tools.infra.chat_with_gemini.subprocess.Popen")
    def test_chat_invokes_bridge(self, mock_popen):
        process = mock_popen.return_value
        process.communicate.return_value = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": "bridge ok"}]
                    },
                }
            ),
            "",
        )

        rc = chat("general", "Reply with one line.")

        self.assertEqual(rc, 0)
        args = mock_popen.call_args.args[0]
        self.assertIn("tools/infra/gemini_mcp.py", args)


if __name__ == "__main__":
    unittest.main()

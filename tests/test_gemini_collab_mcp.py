import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.infra.codex_mcp import CodexBridgeClient
from tools.infra.codex_mcp import CodexBridgeHost


class TestGeminiCollabMCP(unittest.TestCase):
    def test_json_rpc_structure(self):
        """Verify client correctly formats JSON-RPC requests."""
        with patch("pathlib.Path.write_text") as mock_write:
            with patch("pathlib.Path.exists", return_value=False):
                client = CodexBridgeClient(Path("/tmp"))
                try:
                    client.call("test_method", {"param": 1}, timeout=0.1)
                except TimeoutError:
                    pass

                # Check call
                args, _ = mock_write.call_args
                req = json.loads(args[0])
                self.assertEqual(req["jsonrpc"], "2.0")
                self.assertEqual(req["method"], "test_method")

    def test_host_handler_routing(self):
        """Verify host routes messages to registered handlers."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text") as mock_read:
                with patch("pathlib.Path.write_text") as mock_write:
                    mock_read.return_value = json.dumps({
                        "jsonrpc": "2.0",
                        "id": "123",
                        "method": "hello",
                        "params": {"name": "world"}
                    })

                    host = CodexBridgeHost(Path("/tmp"))
                    host.register_handler("hello", lambda p: f"hi {p['name']}")
                    host.process_once()

                    # Check response
                    args, _ = mock_write.call_args
                    res = json.loads(args[0])
                    self.assertEqual(res["result"], "hi world")

import json
import tempfile
import unittest
from pathlib import Path

from tools.infra.codex_mcp import CodexMCPClient
from tools.infra.codex_mcp import CodexMCPHost


class TestCodexMCP(unittest.TestCase):
    def test_host_processes_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            host = CodexMCPHost(root)
            host.register_handler("hello", lambda params: {"message": f"hi {params['name']}"})

            bridge_dir = root / ".codex_bridge"
            bridge_dir.mkdir(parents=True, exist_ok=True)
            (bridge_dir / "request.json").write_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "req-1",
                        "method": "hello",
                        "params": {"name": "world"},
                    }
                )
            )

            self.assertTrue(host.process_once())
            response = json.loads((bridge_dir / "response.json").read_text())
            self.assertEqual(response["result"]["message"], "hi world")

    def test_client_times_out_without_host(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = CodexMCPClient(Path(tmpdir))
            with self.assertRaises(TimeoutError):
                client.call("hello", {"name": "world"}, timeout=0.1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Client to chat with Gemini via the local collaboration MCP bridge."""

from __future__ import annotations

import json
import subprocess
import sys


def chat(topic: str, prompt: str) -> int:
    proc = subprocess.Popen(
        [sys.executable, "tools/infra/gemini_mcp.py", "--default-cwd", "."],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ask_gemini",
            "arguments": {
                "topic": topic,
                "prompt": prompt,
            },
        },
    }
    stdout, stderr = proc.communicate(json.dumps(message))
    if stderr:
        print(stderr)
    if not stdout:
        print("[ERROR] No response from Gemini bridge.")
        return 1

    response = json.loads(stdout)
    if "error" in response:
        print(f"[ERROR] {response['error']['message']}")
        return 1

    print(response["result"]["content"][0]["text"])
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) >= 2:
        return chat(argv[0], " ".join(argv[1:]))
    return chat("general", "Reply with one short sentence confirming the Gemini collaboration bridge works.")


if __name__ == "__main__":
    raise SystemExit(main())

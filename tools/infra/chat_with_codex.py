#!/usr/bin/env python3
"""Expert-to-Expert Collaboration Client for Codex MCP."""

import subprocess
import json
import sys


def collaborate(topic, prompt):
    print(f"[COLLABORATE] Engaging Peer Architect ({topic})...")

    # Initialize MCP Server process
    proc = subprocess.Popen(
        [sys.executable, "tools/infra/codex_mcp.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ask_codex",
            "arguments": {
                "topic": topic,
                "prompt": prompt
            }
        }
    }

    stdout, stderr = proc.communicate(input=json.dumps(msg))

    if stderr:
        print(f"[ERROR] {stderr}")
        return

    try:
        response = json.loads(stdout)
        content = response["result"]["content"][0]["text"]
        print("\n=== PEER ARCHITECT FEEDBACK ===\n")
        print(content)
        print("\n===============================\n")
    except Exception as e:
        print(f"[ERROR] Failed to parse response: {e}")
        print(f"Raw Output: {stdout}")


if __name__ == "__main__":
    peer_review_request = """
I am implementing the 'Hybrid Attachment Architecture' for Schema V4.
Current Plan:
1. Treat ball as a virtual child of the wrist bone during possession (FK-driven).
2. Inherit world-space velocity at detachment (Release frame).
3. Use a parabolic trajectory solver for the In-Flight state.

Critical Technical Inquiry:
How should we handle 'Ball Handoff' transitions (e.g., Crossover or Catch)
to minimize discontinuity in the feature tensor?
Specifically, should we use a 'Soft Handoff' (IK-based smoothing) or
trust the MoCap's anatomical truth to bridge the gap?
    """
    collaborate("architecture", peer_review_request)

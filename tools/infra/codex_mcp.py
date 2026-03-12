#!/usr/bin/env python3
"""Local line-based MCP-like bridge to the project session."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project infra is in path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gemini_project import GeminiProjectClient, repo_root_from

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "hoopsense-codex-mcp", "version": "0.1.0"}

class CodexMCPServer:
    def __init__(self):
        self.project_root = repo_root_from(Path.cwd())
        self.client = GeminiProjectClient(project_root=self.project_root)

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        if method == "initialize":
            return self._success(message["id"], {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            })
        if method == "tools/list":
            return self._success(message["id"], {"tools": [self._tool_schema()]})
        if method == "tools/call":
            return self._handle_tool_call(message)
        return None

    def _tool_schema(self):
        return {
            "name": "ask_codex",
            "description": "Consult the project-scoped engineering session about architecture, design, or implementation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["requirements", "usecases", "architecture", "design", "implementation", "review", "general"],
                        "description": "The domain of the inquiry."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The specific question or request for Codex."
                    }
                },
                "required": ["topic", "prompt"]
            }
        }

    def _handle_tool_call(self, message: dict[str, Any]) -> dict[str, Any]:
        params = message.get("params", {})
        args = params.get("arguments", {})
        topic = args.get("topic", "general")
        prompt = args.get("prompt")
        
        # Establishing Expert Peer Context
        collaboration_prompt = f"""
Act as a Senior Software Architect and Expert Engineer. 
We are collaborating on the HoopSense project. 
Topic: {topic}

Provide an expert-level critique or design guidance. 
Focus on:
- Architectural integrity and component boundaries.
- Performance implications for edge AI (Orin).
- Technical debt and future-proofing.
- Rigorous verification strategies.

Request from your colleague:
{prompt}
""".strip()
        
        response = self.client.ask(collaboration_prompt)
        
        return self._success(message["id"], {
            "content": [{"type": "text", "text": response["response"]}]
        })

    def _success(self, id, result):
        return {"jsonrpc": "2.0", "id": id, "result": result}

def main():
    server = CodexMCPServer()
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = server.handle_message(message)
            if response:
                print(json.dumps(response), flush=True)
        except Exception as exc:
            error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(exc)},
            }
            print(json.dumps(error), flush=True)

if __name__ == "__main__":
    main()

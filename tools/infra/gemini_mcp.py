#!/usr/bin/env python3
"""Project-scoped Gemini collaboration MCP bridge."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gemini_project import GeminiProjectClient  # noqa: E402
from gemini_project import repo_root_from  # noqa: E402

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "hoopsense-gemini-mcp", "version": "0.3.0"}


@dataclass
class GeminiMCPConfig:
    gemini_command: str = "gemini"
    default_cwd: str | None = None


class GeminiMCPServer:
    def __init__(self, config: GeminiMCPConfig | None = None) -> None:
        self.config = config or GeminiMCPConfig()
        if self.config.default_cwd:
            default_root = Path(self.config.default_cwd).resolve()
        else:
            default_root = repo_root_from(Path.cwd())
        self.project_root = default_root
        self.client = GeminiProjectClient(
            project_root=self.project_root,
            gemini_command=self.config.gemini_command,
        )

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        if method == "initialize":
            return self._success(
                message["id"],
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return self._success(message["id"], {})
        if method == "tools/list":
            return self._success(message["id"], {"tools": [self._tool_schema()]})
        if method == "tools/call":
            return self._handle_tool_call(message)
        if "id" in message:
            return self._error(message["id"], -32601, f"Method not found: {method}")
        return None

    def _handle_tool_call(self, message: dict[str, Any]) -> dict[str, Any]:
        params = message.get("params", {})
        name = params.get("name")
        if name != "ask_gemini":
            return self._success(
                message["id"],
                {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True,
                },
            )

        arguments = params.get("arguments") or {}
        prompt = arguments.get("prompt")
        if not prompt:
            return self._success(
                message["id"],
                {
                    "content": [{"type": "text", "text": "Missing required field: prompt"}],
                    "isError": True,
                },
            )

        result = self._invoke_gemini(arguments)
        payload = {
            "content": [{"type": "text", "text": json.dumps(result, indent=2, sort_keys=True)}],
            "structuredContent": result,
        }
        if result["returncode"] != 0:
            payload["isError"] = True
        return self._success(message["id"], payload)

    def _invoke_gemini(self, arguments: dict[str, Any]) -> dict[str, Any]:
        topic = arguments.get("topic", "general")
        prompt = self._build_collaboration_prompt(topic, arguments["prompt"])
        response = self.client.ask(
            prompt,
            model=arguments.get("model"),
            output_format=arguments.get("output_format", "json"),
        )
        return {
            "topic": topic,
            "cwd": str(self.project_root),
            "returncode": 0,
            "response": response.get("response", ""),
            "session_id": response.get("session_id"),
            "stats": response.get("stats"),
        }

    @staticmethod
    def _build_collaboration_prompt(topic: str, prompt: str) -> str:
        return f"""
Act as a Senior Software Architect and Expert Engineer collaborating on the HoopSense project.

Topic: {topic}

Provide high-signal, technically rigorous guidance.
Focus on:
- architectural integrity and component boundaries
- performance and deployment implications
- technical debt and future-proofing
- concrete verification strategy

Request from your collaborator:
{prompt}
""".strip()

    @staticmethod
    def _tool_schema() -> dict[str, Any]:
        return {
            "name": "ask_gemini",
            "title": "Ask Gemini Collaborator",
            "description": "Consult the project-scoped Gemini collaborator.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": [
                            "requirements",
                            "usecases",
                            "architecture",
                            "design",
                            "implementation",
                            "review",
                            "general",
                        ],
                        "description": "The domain of the inquiry.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The specific request.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional Gemini model override.",
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Gemini CLI output format.",
                        "enum": ["text", "json", "stream-json"],
                        "default": "json",
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _success(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    @staticmethod
    def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _read_message(stream: Any) -> dict[str, Any] | None:
    content_length = None
    while True:
        header = stream.readline()
        if not header:
            return None
        if header in (b"\r\n", b"\n"):
            break
        name, _, value = header.decode("utf-8").partition(":")
        if name.lower() == "content-length":
            content_length = int(value.strip())

    if content_length is None:
        raise ValueError("Missing Content-Length header")
    body = stream.read(content_length)
    if len(body) != content_length:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(stream: Any, message: dict[str, Any]) -> None:
    body = json.dumps(message).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    stream.write(body)
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expose the Gemini collaborator.")
    parser.add_argument("--gemini-command", default="gemini")
    parser.add_argument("--default-cwd")
    args = parser.parse_args(argv)

    server = GeminiMCPServer(
        GeminiMCPConfig(
            gemini_command=args.gemini_command,
            default_cwd=args.default_cwd,
        )
    )

    while True:
        try:
            message = _read_message(sys.stdin.buffer)
        except ValueError as exc:
            error = GeminiMCPServer._error(None, -32700, str(exc))
            _write_message(sys.stdout.buffer, error)
            continue

        if message is None:
            return 0

        response = server.handle_message(message)
        if response is not None:
            _write_message(sys.stdout.buffer, response)


if __name__ == "__main__":
    raise SystemExit(main())

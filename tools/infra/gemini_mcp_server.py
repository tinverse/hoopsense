#!/usr/bin/env python3
"""Minimal stdio MCP server that exposes Gemini CLI as a tool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "hoopsense-gemini-bridge", "version": "0.1.0"}
TEXT_JSON = "application/json"


@dataclass
class GeminiBridgeConfig:
    gemini_command: str = "gemini"
    default_cwd: str | None = None


class GeminiBridgeServer:
    def __init__(self, config: GeminiBridgeConfig | None = None) -> None:
        self.config = config or GeminiBridgeConfig()

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
        command = [self.config.gemini_command]
        model = arguments.get("model")
        if model:
            command.extend(["--model", model])

        approval_mode = arguments.get("approval_mode")
        if approval_mode:
            command.extend(["--approval-mode", approval_mode])

        output_format = arguments.get("output_format", "json")
        command.extend(["--output-format", output_format, "--prompt", arguments["prompt"]])

        include_directories = arguments.get("include_directories") or []
        for directory in include_directories:
            command.extend(["--include-directories", directory])

        if arguments.get("sandbox") is not None:
            command.extend(["--sandbox", str(bool(arguments["sandbox"])).lower()])

        cwd = arguments.get("cwd") or self.config.default_cwd
        resolved_cwd = str(Path(cwd).resolve()) if cwd else None

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=resolved_cwd,
        )
        return {
            "command": command,
            "cwd": resolved_cwd,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    @staticmethod
    def _tool_schema() -> dict[str, Any]:
        return {
            "name": "ask_gemini",
            "title": "Ask Gemini CLI",
            "description": "Send a one-shot prompt to the local Gemini CLI and return stdout/stderr.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to Gemini in headless mode.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional Gemini model override.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory for the Gemini invocation.",
                    },
                    "approval_mode": {
                        "type": "string",
                        "description": "Gemini approval mode, such as default, auto_edit, yolo, or plan.",
                    },
                    "sandbox": {
                        "type": "boolean",
                        "description": "Whether to ask Gemini CLI to run in sandbox mode.",
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Gemini CLI output format.",
                        "enum": ["text", "json", "stream-json"],
                        "default": "json",
                    },
                    "include_directories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra directories Gemini should include in its workspace.",
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
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": code, "message": message},
        }


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
    parser = argparse.ArgumentParser(description="Expose Gemini CLI through a minimal stdio MCP server.")
    parser.add_argument("--gemini-command", default="gemini")
    parser.add_argument("--default-cwd")
    args = parser.parse_args(argv)

    server = GeminiBridgeServer(
        GeminiBridgeConfig(
            gemini_command=args.gemini_command,
            default_cwd=args.default_cwd,
        )
    )

    while True:
        try:
            message = _read_message(sys.stdin.buffer)
        except ValueError as exc:
            error = GeminiBridgeServer._error(None, -32700, str(exc))
            _write_message(sys.stdout.buffer, error)
            continue

        if message is None:
            return 0

        response = server.handle_message(message)
        if response is not None:
            _write_message(sys.stdout.buffer, response)


if __name__ == "__main__":
    raise SystemExit(main())

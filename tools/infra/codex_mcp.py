#!/usr/bin/env python3
"""Canonical repo-local Codex collaboration transport.

This is a filesystem-backed mailbox transport for structured collaboration with
Codex-like agents in the repo. It does not attach to the live Codex session by
itself; instead, it provides a stable request/response protocol and host loop
for local tools or external agents to use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

BRIDGE_DIRNAME = ".codex_bridge"
REQUEST_FILENAME = "request.json"
RESPONSE_FILENAME = "response.json"


class CodexMCPClient:
    def __init__(self, project_root: Path) -> None:
        self.bridge_dir = project_root / BRIDGE_DIRNAME
        self.request_path = self.bridge_dir / REQUEST_FILENAME
        self.response_path = self.bridge_dir / RESPONSE_FILENAME

    def call(self, method: str, params: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        request_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        if self.response_path.exists():
            self.response_path.unlink()
        self.request_path.write_text(json.dumps(request, indent=2))

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.response_path.exists():
                try:
                    response = json.loads(self.response_path.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.2)
                    continue
                if response.get("id") == request_id:
                    if self.request_path.exists():
                        self.request_path.unlink()
                    return response
            time.sleep(0.2)

        raise TimeoutError(f"Timed out waiting for Codex response to {request_id}")


class CodexMCPHost:
    def __init__(self, project_root: Path) -> None:
        self.bridge_dir = project_root / BRIDGE_DIRNAME
        self.request_path = self.bridge_dir / REQUEST_FILENAME
        self.response_path = self.bridge_dir / RESPONSE_FILENAME
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register_handler(self, method: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.handlers[method] = handler

    def process_once(self) -> bool:
        if not self.request_path.exists():
            return False

        try:
            request = json.loads(self.request_path.read_text())
        except json.JSONDecodeError:
            return False

        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method not in self.handlers:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        else:
            try:
                result = self.handlers[method](params)
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result,
                }
            except Exception as exc:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(exc)},
                }

        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self.response_path.write_text(json.dumps(response, indent=2))
        return True


def _demo_handler(params: dict[str, Any]) -> dict[str, Any]:
    topic = params.get("topic", "general")
    prompt = params.get("prompt", "")
    return {
        "topic": topic,
        "response": f"Codex host received topic={topic!r} prompt={prompt!r}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repo-local Codex collaboration transport.")
    parser.add_argument("--project-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    call_parser = subparsers.add_parser("call", help="Send one mailbox request.")
    call_parser.add_argument("method")
    call_parser.add_argument("payload", help="JSON object for params")
    call_parser.add_argument("--timeout", type=float, default=30.0)

    ask_parser = subparsers.add_parser("ask", help="Send a collaborative consultation request.")
    ask_parser.add_argument("topic", choices=["architecture", "design", "implementation", "review", "general"])
    ask_parser.add_argument("prompt")
    ask_parser.add_argument("--timeout", type=float, default=60.0)

    subparsers.add_parser("serve-demo", help="Run a demo host with ask_codex.")

    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()

    if args.command == "call":
        client = CodexMCPClient(project_root)
        params = json.loads(args.payload)
        response = client.call(args.method, params, timeout=args.timeout)
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0

    if args.command == "ask":
        client = CodexMCPClient(project_root)
        params = {"topic": args.topic, "prompt": args.prompt}
        response = client.call("ask_codex", params, timeout=args.timeout)
        
        if "error" in response:
            print(f"[ERROR] {response['error']['message']}", file=sys.stderr)
            return 1
            
        result = response.get("result", {})
        print(result.get("response", "No response content."))
        return 0

    if args.command == "serve-demo":
        host = CodexMCPHost(project_root)
        host.register_handler("ask_codex", _demo_handler)
        print(f"[codex_mcp] serving demo mailbox at {project_root / BRIDGE_DIRNAME}", file=sys.stderr)
        try:
            while True:
                host.process_once()
                time.sleep(0.25)
        except KeyboardInterrupt:
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

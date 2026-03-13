#!/usr/bin/env python3
"""Reciprocal bridge for Gemini-to-Codex communication using file-backed transport."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# Add infra to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

BRIDGE_DIRNAME = ".codex_bridge"
REQUEST_FILENAME = "request.json"
RESPONSE_FILENAME = "response.json"


class CodexBridgeClient:
    """Client for Gemini to send requests to Codex."""

    def __init__(self, project_root: Path) -> None:
        self.bridge_dir = project_root / BRIDGE_DIRNAME
        self.request_path = self.bridge_dir / REQUEST_FILENAME
        self.response_path = self.bridge_dir / RESPONSE_FILENAME

    def call(self, method: str, params: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        """
        Send a request and wait for a response.
        This uses a simple file-based polling mechanism for local collaboration.
        """
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

        request_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        # Clear any stale response
        if self.response_path.exists():
            self.response_path.unlink()

        # Write the request
        self.request_path.write_text(json.dumps(request, indent=2))

        # Poll for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.response_path.exists():
                try:
                    response = json.loads(self.response_path.read_text())
                    if response.get("id") == request_id:
                        self.request_path.unlink()  # Clean up request
                        return response
                except json.JSONDecodeError:
                    pass  # Wait for file to be fully written
            time.sleep(0.5)

        raise TimeoutError(f"Codex bridge timeout after {timeout}s waiting for {request_id}")


class CodexBridgeHost:
    """Host for Codex to receive and respond to Gemini requests."""

    def __init__(self, project_root: Path) -> None:
        self.bridge_dir = project_root / BRIDGE_DIRNAME
        self.request_path = self.bridge_dir / REQUEST_FILENAME
        self.response_path = self.bridge_dir / RESPONSE_FILENAME
        self.handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register_handler(self, method: str, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.handlers[method] = handler

    def process_once(self) -> bool:
        """Process a single request if it exists."""
        if not self.request_path.exists():
            return False

        try:
            request = json.loads(self.request_path.read_text())
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            if method in self.handlers:
                try:
                    result = self.handlers[method](params)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": result
                    }
                except Exception as e:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": str(e)}
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }

            self.response_path.write_text(json.dumps(response, indent=2))
            return True
        except (json.JSONDecodeError, PermissionError):
            return False  # Wait for valid file access

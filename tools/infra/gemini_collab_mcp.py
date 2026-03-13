#!/usr/bin/env python3
"""Compatibility shim for the canonical Gemini MCP bridge."""

from __future__ import annotations

from gemini_mcp import GeminiMCPServer
from gemini_mcp import main

GeminiCollabServer = GeminiMCPServer


if __name__ == "__main__":
    raise SystemExit(main())

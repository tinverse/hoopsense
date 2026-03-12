#!/usr/bin/env python3
"""Compatibility wrapper for the renamed Gemini collaboration MCP bridge."""

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gemini_collab_mcp import GeminiCollabConfig as GeminiBridgeConfig
from gemini_collab_mcp import GeminiCollabServer as GeminiBridgeServer
from gemini_collab_mcp import main


if __name__ == "__main__":
    raise SystemExit(main())

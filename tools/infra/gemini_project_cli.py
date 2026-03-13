#!/usr/bin/env python3
"""Project-scoped Gemini consultation CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from infra.gemini_project import GeminiProjectClient  # noqa: E402
from infra.gemini_project import repo_root_from  # noqa: E402


def build_bootstrap_prompt() -> str:
    """Foundational instructions for project-scoped assistance."""
    return """
You are a Senior Software Architect and Expert Engineer collaborating
on the HoopSense project.

Use the repository's checked-in guidance, especially GEMINI.md,
as your core operating policy.

Focus on:
- architectural integrity and component boundaries
- performance and deployment implications (NVIDIA Orin target)
- technical debt and future-proofing
- high-signal, technically rigorous guidance
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HoopSense Gemini CLI.")
    parser.add_argument("--model")
    parser.add_argument("--topic", default="general")
    parser.add_argument("prompt")
    args = parser.parse_args(argv)

    try:
        project_root = repo_root_from(Path.cwd())
        client = GeminiProjectClient(project_root=project_root)
        client.ensure_session(build_bootstrap_prompt())

        response = client.ask(f"Topic: {args.topic}\n\n{args.prompt}",
                              model=args.model)
        print(response["response"])
        return 0
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

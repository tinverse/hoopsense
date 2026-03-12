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

from infra.gemini_project import GeminiProjectClient
from infra.gemini_project import repo_root_from
from review.review_change import build_bootstrap_prompt

VALID_TOPICS = (
    "requirements",
    "usecases",
    "architecture",
    "design",
    "implementation",
    "review",
    "general",
)


def build_consult_prompt(topic: str, request: str) -> str:
    scope_line = {
        "requirements": "Focus on requirements quality, scope, missing constraints, and acceptance criteria.",
        "usecases": "Focus on user flows, actors, edge cases, and output artifacts.",
        "architecture": "Focus on boundaries, contracts, components, and likely refactor pressure.",
        "design": "Focus on detailed design, interfaces, data flow, and testability.",
        "implementation": "Focus on implementation tradeoffs, sequencing, and regression risk.",
        "review": "Focus on review concerns, likely findings, and what should be checked before merge.",
        "general": "Focus on the highest-value project guidance for the request.",
    }[topic]
    return f"""
Act as my project collaborator for HoopSense.

Topic: {topic}
Guidance:
- {scope_line}
- Prefer checked-in project artifacts over generic advice.
- Be concise and technical.

Request:
{request}
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consult Gemini using a project-scoped session.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    consult = subparsers.add_parser("consult", help="Ask Gemini about a project topic.")
    consult.add_argument("topic", choices=VALID_TOPICS)
    consult.add_argument("request", nargs="+")
    consult.add_argument("--model")

    status = subparsers.add_parser("status", help="Show the current project session id.")
    bootstrap = subparsers.add_parser("bootstrap", help="Create the project Gemini session if it does not exist.")
    bootstrap.add_argument("--model")

    args = parser.parse_args(argv)

    try:
        project_root = repo_root_from(Path.cwd())
        client = GeminiProjectClient(project_root=project_root)

        if args.command == "status":
            print(client.load_session_id() or "No project Gemini session.")
            return 0

        if args.command == "bootstrap":
            session_id = client.ensure_session(build_bootstrap_prompt())
            print(session_id)
            return 0

        client.ensure_session(build_bootstrap_prompt())
        prompt = build_consult_prompt(args.topic, " ".join(args.request))
        response = client.ask(prompt, model=args.model)
        print(response["response"])
        return 0
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

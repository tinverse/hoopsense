#!/usr/bin/env python3
"""Run a project-scoped Gemini review for the current git diff."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from infra.gemini_project import GeminiProjectClient  # noqa: E402
from infra.gemini_project import repo_root_from  # noqa: E402


def build_bootstrap_prompt() -> str:
    return """
You are the persistent reviewer for the HoopSense project.

Retain project context across this session. Your standing role is code review
for changes made in this repository. Use the repository's checked-in
guidance, especially GEMINI.md, as the review policy.

Review style:
- Findings first.
- Focus on bugs, regressions, contract drift, and documentation drift.
- Be concise and technical.
- When there are no findings, say so explicitly.
""".strip()


def build_review_prompt(diff: str) -> str:
    return f"""
You are acting as reviewer for changes made in the HoopSense project.

Review the following git diff for:
1. Compliance with the Teach-First and Design-First mandates in GEMINI.md.
2. Logic bugs, regressions, broken assumptions, or contract drift.
3. Rust best practices where relevant.
4. Python best practices where relevant.
5. Missing or weak tests.
6. Missing documentation or status updates.

Return:
- Findings first, ordered by severity.
- Then a short summary with either Approved or Changes Requested.

DIFF:
{diff}
""".strip()


def get_diff(*, staged: bool, revision_range: str | None) -> str:
    if revision_range:
        return subprocess.check_output(
            ["git", "diff", revision_range],
            text=True,
        )
    cmd = ["git", "diff", "--cached"] if staged else ["git", "diff"]
    return subprocess.check_output(cmd, text=True)


def run_agent_review(diff: str, client: GeminiProjectClient,
                     model: str | None = None) -> str:
    client.ensure_session(build_bootstrap_prompt())
    response = client.ask(build_review_prompt(diff), model=model)
    return response["response"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gemini review.")
    parser.add_argument("--model")
    parser.add_argument("--unstaged", action="store_true",
                        help="Review unstaged changes.")
    parser.add_argument("--diff-range",
                        help="Explicit git diff range.")
    parser.add_argument("--print-prompt", action="store_true",
                        help="Print the prompt instead of calling.")
    args = parser.parse_args(argv)

    try:
        project_root = repo_root_from(Path.cwd())
        diff = get_diff(staged=not args.unstaged,
                        revision_range=args.diff_range)
        if not diff.strip():
            print("No changes to review.")
            return 0

        if args.print_prompt:
            print(build_review_prompt(diff))
            return 0

        client = GeminiProjectClient(project_root=project_root)
        print(run_agent_review(diff, client=client, model=args.model))
        return 0
    except subprocess.CalledProcessError:
        print("Error: This script must be run within a git repository.")
        return 1
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

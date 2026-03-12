# Agent Workflow

This repository uses a project-scoped Gemini collaboration workflow in addition to the primary coding agent.

## Persistent Gemini Session

- Use the project-scoped Gemini session, not ad hoc folder-specific chats.
- Session bootstrap and consultation entrypoint:
  - `python tools/infra/gemini_project_cli.py bootstrap`
  - `python tools/infra/gemini_project_cli.py consult <topic> "<request>"`
- Valid consultation topics:
  - `requirements`
  - `usecases`
  - `architecture`
  - `design`
  - `implementation`
  - `review`
  - `general`

## When To Consult Gemini

- Consult Gemini for substantial tasks that involve tradeoffs, architecture, design, or ambiguous implementation paths.
- Consultation may span multiple rounds.
- It is acceptable to explore alternatives, backtrack, and choose a different path after new evidence appears.

## Decision Standard

- Treat Gemini as a second engineering voice, not as authority.
- Critically review Gemini's comments against:
  - checked-in repo artifacts
  - current code
  - tests
  - task constraints
- Move forward when there is clear consensus or when one option is materially better and the reasoning is defensible.
- Reject or narrow Gemini feedback that is generic, weak, or contradicted by local evidence.

## Required Review Flow

- For substantial code changes:
  1. Consult Gemini when needed on requirements, use cases, architecture, design, or implementation.
  2. Implement or revise the change locally.
  3. Re-consult Gemini if implementation changes the tradeoffs.
  4. Run Gemini review on the diff with `python tools/review/review_change.py`.
  5. Critically adjudicate review findings before making follow-up edits.

## Review Output Standard

- Findings first.
- Focus on bugs, regressions, contract drift, missing tests, and documentation drift.
- Do not accept review comments blindly.
- The primary coding agent remains responsible for final technical judgment and integration quality.

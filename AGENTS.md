# Agent Workflow

This repository uses a project-scoped Gemini collaboration workflow in addition to the primary coding agent.

## Planning Invariant

- Plan-first execution is required for substantial work.
- Before architecture changes, feature work, or multi-file implementation, update `docs/plan/PLAN_TREE.yaml`.
- Decompose requirements into executable tasks in the plan tree before coding.
- Prioritize work from the current plan frontier instead of ad hoc task selection.
- When coordinating with Gemini or other agents, use the plan tree as the source of truth for:
  - task boundaries
  - ownership
  - priority
  - completion status
- If implementation changes the task graph, update the plan tree before or alongside the code change.
- `TASK_STATUS.md` is a human-readable execution snapshot and should be kept aligned with the plan tree, not used as the sole planning artifact.

## Planning Workflow

For substantial tasks:
1. Update `docs/plan/PLAN_TREE.yaml`.
2. Expand the relevant requirement into concrete child tasks until the next frontier is implementable and testable.
3. Prioritize the frontier.
4. Consult Gemini when useful on requirements, use cases, architecture, design, or implementation.
5. Execute the highest-priority task slice.
6. Reconcile code, tests, docs, and `TASK_STATUS.md` back to the plan tree.

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

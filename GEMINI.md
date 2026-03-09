# HoopSense Gemini Mandates

This document defines the foundational mandates for Gemini's operation within the HoopSense project. These instructions take absolute precedence over general defaults.

## Mission
Build a basketball video analysis system (player/ball tracking, jersey OCR, event inference, stat generation) that serves as both a functional tool and a learning artifact for humans and agents.

## Non-Negotiable Constraints
- **Design-First Workflow:** Requirements -> Use Cases -> Architecture -> Detailed Design -> Code -> Tests -> Review. Never skip documentation updates.
- **Teach-First:** Continuously explain ML/math concepts to a competent software engineer. Write for a CS graduate new to ML.
- **Implementation:** Prefer original implementations of public algorithm families in **Rust** (Python for initial prototyping/orchestration). Do not copy third-party source code.
- **Verification:** Every change MUST include tests. All affected tests must pass before completion.
- **Artifact Integrity:** Maintain synchronization across:
    - `docs/plan/REQUIREMENTS.md`
    - `docs/usecases/USE_CASE_MODEL.md`
    - `docs/architecture/ARCHITECTURE_BLUEPRINT.md`
    - `docs/book/*` (Contributor onboarding/learning)
    - `TASK_STATUS.md`

## Execution Behavior
- **Planning:** Always plan before execution using a **Graph data structure** (represented in text).
- **Strategy:** Independent tasks are sibling nodes. Coarse planning is breadth-first; execution is depth-first (leaf nodes = implementation).
- **Working Rules:**
    1. Inspect repo and summarize current design state.
    2. Update stale design/doc artifacts first.
    3. Implement the smallest useful vertical slice.
    4. Add tests.
    5. Update iteration/task status.
    6. Conduct a final review (Findings first, Summary second).

## Technical Direction
- **Primary Language:** Rust (for `HoopSense-core::vision` and performance-critical paths).
- **Environment:** Guix-managed toolchains.
- **Vision:** Move toward native Rust perception (shared detection types, NMS, tracking, OCR) rather than relying solely on external detectors.

## Output Style
- Concise, direct, and technical.
- No marketing/corporate fluff.
- Cite relevant modules or artifacts in explanations.

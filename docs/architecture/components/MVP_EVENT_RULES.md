# MVP Event Rules

This document defines the first machine-readable basketball event-evidence
contract for the HoopSense MVP.

The checked-in source of truth is:

- `specs/mvp_event_rules.yaml`

## Purpose

The MVP should not count stats directly from neural labels. It should count
stats from deterministic event rules that consume:

- live-play state
- player identity
- ball evidence
- possession/evidence constraints

## Scope

The current contract intentionally covers only the first sellable slice:

- `pass`
- `catch`
- `shot_attempt`
- `made_shot`
- `missed_shot`
- `rebound_off`
- `rebound_def`
- `turnover`
- `steal`
- `assist_candidate`

This is enough to support the first scorebook-style box score without
pretending the repo already supports a full basketball semantics engine.

## Rule Shape

Each event rule separates four concerns:

1. `prerequisites`
   - what must already be true before an event can be considered
2. `actors`
   - which identities are required and what role they play
3. `temporal`
   - what must happen before or after the event, and within what time bounds
4. `counting`
   - how the event contributes to scorebook columns

## Why This Exists

This contract is the bridge between:

- low-level evidence constraints
- possession/event attribution
- deterministic stat counting

It keeps the MVP aligned to the actual sellable output instead of allowing
event semantics to drift into ad hoc code.

# HoopSense MVP Stats Contract

## Purpose

The first sellable HoopSense MVP is not a generic "basketball understanding" demo. It is a
scorebook-style stat product that can reproduce the core per-player box score expected by a real
team-facing basketball stats sheet.

This document defines the MVP stat contract that higher layers must satisfy before the product
can be treated as commercially credible.

## Product Contract

This document is the normalized source of truth for the first sellable output. HoopSense should be
judged against this stat contract, not against raw model confidence or informal basketball
intuition.

## MVP Outcome

For a single game, HoopSense should produce auditable per-player box-score rows that match the
core stat fields and deterministic formulas of a conventional youth-basketball team scorebook.
Season totals and averages can be derived by aggregation after the game-level rows are
trustworthy.

The MVP output should be explainable through:

- attributed event records
- possession context
- deterministic stat formulas
- reviewable provenance for the stat schema/formulas used

## In-Scope Stats

### Core counting stats

These are the minimum sellable stats for the first MVP:

- `PTS`
- `O. Reb.`
- `D. Reb.`
- `T. Reb.`
- `Assists`
- `Steals`
- `Blocks`
- `TOs`
- `2P FGM`
- `2P FGA`
- `3P FGM`
- `3P FGA`
- `FTM`
- `FTA`

### Deterministic derived stats

These should be computed from the counted events rather than inferred independently:

- `FG%`
- `2PPS`
- `3P FG%`
- `3PPS`
- `TFGM`
- `TFGA`
- `TFG %`
- `TPPS`
- `FT %`
- `EFF`
- `OIS`
- `OIS%`

### Presentation target

The rendered output should look very close to a sellable team stats sheet, whether it is exported
as CSV, JSON, HTML, or XLSX later. The product contract is the row/column shape and the formulas,
not the spreadsheet file itself.

### Useful but non-MVP fields

These may remain pending until the core box score is reliable:

- `Hi 5s`
- `D/J/L`
- `D. EFF`
- `A.PTSG`
- `E.PTS +/-`
- shot-profile splits such as layups, catch-and-shoot jumpers, and off-the-dribble jumpers

## Required Event Primitives

The scorebook contract implies a narrower and more concrete event set than the current generic
ledger.

The MVP should be able to attribute at least:

- `pass`
- `catch`
- `dribble`
- `shot_attempt`
- `shot_make`
- `shot_miss`
- `free_throw_make`
- `free_throw_miss`
- `rebound_off`
- `rebound_def`
- `steal`
- `block`
- `turnover`
- `assist`

The Action Brain should remain a local motion classifier. These events must still be derived from
motion signals plus possession context, ball state, and court geometry.

## Required Possession Context

Reliable stats require possession fields that are not yet fully proven in the repo. At minimum,
the ledger must support:

- `possession_id`
- `team_id`
- `ballhandler_id`
- `start_t_ms`
- `possession_origin`
- `dribble_count`
- `pass_count`
- `offense_zone`
- `transition_flag`

The possession contract should make event attribution explainable, especially for assists,
turnovers, and offensive vs defensive rebounds.

## Identity Requirement

The MVP requires stable internal player identity within a game. It does not require full roster
or jersey-to-real-name identification in the first slice, but it does require enough continuity
to assign box-score events to the correct internal player ID.

## What Is Explicitly Deferred

The following should not block the first sellable stats MVP:

- full shot chart visual polish
- defensive IQ metrics beyond direct counted stats such as steals and blocks
- avatar/export features
- broad cloud deployment polish
- full season reporting UX
- advanced shot-profile categories unless the core box score is already reliable

## Auditability Standard

The MVP is only credible if every output stat can be traced back to:

- a generated event record
- the possession context around that event
- the exact stat header/formula template version
- the code revision and fixture used for validation

This means stat generation must remain separate from neural logits. The system should be able to
show why a rebound, assist, or turnover was counted.

## Suggested Artifact Pattern

The sibling `../shush` project has a useful pattern worth reusing conceptually:

- a checked-in stats header template
- a checked-in stats formula template
- a declarative rules layer between attributed events and final stat rows
- formula-template provenance hashing
- explicit validation that formulas only reference known output columns
- separation between event inference and stats aggregation/export

HoopSense should adopt the idea, but not the implementation. Nothing should be copied from
`../shush` into this MVP path. The deliverable here is a HoopSense-native rules and template
contract that is informed by the pattern, not derived from the code.

## Implementation Order

1. Publish and freeze the normalized MVP stat contract in the repo.
2. Check in canonical stats header and formula templates for the MVP columns.
3. Define a HoopSense-native declarative rules layer that maps attributed events and possession
   state into counted stats.
4. Map every MVP stat to the event primitives and possession fields required to compute it.
5. Extend the ledger and attribution layer until those inputs are present.
6. Generate auditable per-player box-score rows from attributed events.
7. Validate generated rows against trusted scorebook fixtures before widening scope.

## Exit Criteria

The MVP stats contract is satisfied when HoopSense can:

- emit per-player game rows for the in-scope fields
- derive percentages and efficiency metrics deterministically
- explain each counted stat through events plus possession state
- compare generated rows against trusted scorebook fixtures with explicit pass/fail evidence

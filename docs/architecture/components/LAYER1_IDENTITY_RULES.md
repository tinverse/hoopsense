# Layer 1 Identity Rules

This document defines the current Layer 1 identity-policy contract for basketball video in HoopSense.

The machine-readable source of truth is:

- `specs/layer1_identity_policy.yaml`

## Purpose

Layer 1 identity logic must stay aligned with basketball-specific priors instead of drifting into generic tracking heuristics. In particular:

- active on-court players are conserved with high probability
- disappearance inside a continuous segment usually means occlusion or detector miss
- true identity resets should mainly happen at detected video discontinuities
- bounded identity hypotheses are allowed only in short ambiguous windows

## Policy Structure

The policy now formalizes four distinct things:

1. assumptions
2. continuity segmentation behavior
3. evidence model
4. hard constraints

That separation matters because the tracker should not mix up:

- what was observed
- what makes an explanation more likely
- what explanations are impossible

## Assumptions

The checked-in assumptions remain intentionally simple:

- player-set persistence is the default over short windows
- disappearance usually means occlusion or detector miss
- continuity is conserved inside one continuity segment
- detected video discontinuity is the default reset condition

These are priors, not proofs. They bias the runtime toward conservative identity persistence unless stronger contrary evidence appears.

## Evidence Model

The evidence model defines the soft signals that can support or weaken an identity explanation. Soft signals may change a score, but they do not by themselves make a hypothesis impossible.

### 1. Observation Contract

The runtime should treat these fields as the minimum identity observation shape:

- required: `frame_idx`, `t_ms`, `bbox_xyxy`, `confidence`, `continuity_segment_id`
- optional: `court_xy`, `footpoint_xy`, `smoothed_velocity_xy`, `motion_speed_px`, pose fields, appearance fields, grounding/bootstrap state, SAM refinement state, and live-play context

This keeps the identity layer tied to actual emitted detections instead of implicit local variables.

### 2. On-Court Plausibility Evidence

This is the soft evidence used to decide whether a raw person detection is a credible on-court player candidate.

Positive signals:

- detector confidence
- bbox height / scale
- court grounding
- foreground support near the feet and center
- pose coherence
- track persistence
- local motion
- SAM support

Penalty signals:

- frame-edge bias
- merge risk from oversized or malformed detections
- appearance mismatch against current team-color prototypes
- shared low-relative-motion patterns that usually indicate crowd drift during a pan

This score answers: "does this look like a real player on the court right now?"

### 3. Short-Gap Identity-Link Evidence

This is the soft evidence used when deciding whether two short track fragments should be treated as the same player.

The current MVP link score uses:

- predicted image-plane position continuity
- velocity alignment
- bbox size consistency
- coarse uniform-bucket compatibility
- optional court-space continuity
- temporal gap length
- confidence floor

This score answers: "if one track ended and another began shortly after, how plausible is it that they are the same player?"

### 4. Scene-State Evidence

Identity decisions are contextual. The runtime should use scene state to decide how much to trust prior continuity.

The current scene-state contract includes:

- `continuity_segment_id` as the partition field
- `discontinuity_label` as the reset hint
- a rule that identity confidence resets or regrounds at discontinuities
- a rule that missing detections inside one segment default to occlusion or detector miss
- live-play state as contextual evidence, not a hard gate

## Hard Constraints

Hard constraints define explanations the runtime must reject regardless of score.

### 1. Discontinuity Partition

If identity reset on discontinuity is enabled, the runtime must not bridge tracks across different continuity segments as if nothing happened.

### 2. Short-Gap Motion / Geometry Bounds

For MVP short-gap repair, reject a candidate link when any of these hold:

- the temporal gap exceeds the configured short-gap window
- court-space distance exceeds the configured maximum when both observations are grounded
- predicted center displacement exceeds the configured image-plane ratio bound
- size consistency falls below the minimum threshold

These are the first explicit physics-like guards for player continuity.

### 3. Assignment Exclusivity

The current linker enforces one-to-one local repair:

- one predecessor cannot claim multiple successors
- one successor cannot claim multiple predecessors

This prevents local duplicate merges inside one ambiguity window.

### 4. Temporal-Overlap Exclusion

Two track spans that overlap in time cannot be merged into one canonical identity. The runtime may keep both as ambiguous alternatives, but it should not collapse them into the same player.

### 5. Occlusion-First Within Segment

Inside one continuity segment, a short disappearance should prefer retaining recent identity through missing state before spawning a new canonical player identity.

This is not a scoring preference only; it is a behavioral constraint on repair logic.

## Consumption in Code

The Layer 1 artifact generator should consume the machine-readable policy and apply it in four places:

1. continuity segmentation
2. on-court player plausibility scoring
3. short-gap identity repair and hypothesis ranking
4. artifact emission for review and regression analysis
5. downstream consumers such as jersey OCR, which may use bounded identity options as evidence without forcing immediate identity collapse
6. later-pass global-hypothesis reranking, where jersey-supported option consensus can break close ties between bounded identity worlds without rewriting committed runtime identities

Artifacts should record the policy version and source path that were used, so reviewed failures can always be interpreted against the exact rule set that produced them.

## Evolution Path

This contract is intentionally MVP-sized.

It is designed so that later bounded MHT or another probabilistic association layer can reuse the same pieces:

- observation contract
- soft evidence families
- hard rejection rules
- scene-state partitioning
- local conflict-group hypotheses
- bounded global identity hypotheses assembled from unresolved groups

The current next-step extension is a bounded global hypothesis layer: keep one committed identity world for high-margin groups, retain multiple ranked worlds for ambiguous groups, and emit per-track canonical-identity options without forcing the stitcher to collapse an ambiguous decision too early.

That way a future probabilistic layer reasons over explicit basketball evidence instead of over raw detector outputs alone.

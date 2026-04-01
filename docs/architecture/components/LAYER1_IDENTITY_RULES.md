# Layer 1 Identity Rules

This document defines the current Layer 1 identity-policy contract for
basketball video in HoopSense.

The machine-readable source of truth is:

- `specs/layer1_identity_policy.yaml`

## Purpose

Layer 1 identity logic must stay aligned with basketball-specific priors instead
of drifting into generic tracking heuristics. In particular:

- active on-court players are conserved with high probability
- disappearance inside a continuous segment usually means occlusion or detector
  miss
- true identity resets should mainly happen at detected video discontinuities
- bounded identity hypotheses are allowed only in short ambiguous windows

## Policy

### 1. Continuity Comes First

The video is partitioned into continuity segments.

Inside one continuity segment:

- identity continuity is the default
- short disappearance should be interpreted as occlusion or missed detection
- new identity births should be treated conservatively

Across a detected discontinuity:

- identity continuity is no longer assumed
- repair logic may stop bridging tracks across the boundary

### 2. Player-Set Persistence

Basketball does not behave like an open-world crowd scene.

The active player set is expected to be stable over short spans:

- the same players are likely still present
- locations can change continuously
- visibility can change because of occlusion
- substitutions are relatively rare and should not be the default explanation

### 3. Occlusion-First Repair

Short-gap repair should prefer:

- continuing an existing identity through a brief miss

over:

- declaring a genuinely new player identity

The default repair window is intentionally short and conservative.

### 4. Bounded Identity Hypotheses

When local evidence is ambiguous, Layer 1 may keep several plausible identity
bridges alive for review and later resolution.

This is intentionally bounded:

- only for short gaps
- only for local predecessor/successor conflicts
- only a small number of candidates per ambiguity group

This is not a full-scene, full-game global MHT system.

## Consumption in Code

The Layer 1 artifact generator should consume the machine-readable policy and
apply it in three places:

1. continuity segmentation
2. short-gap identity repair
3. bounded identity-hypothesis summarization

Artifacts should record the policy version and source path that were used, so
reviewed failures can always be interpreted against the exact rule set that
produced them.

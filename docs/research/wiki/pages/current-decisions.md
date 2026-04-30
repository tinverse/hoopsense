# Current Decisions

This page records the design decisions that are already active or strongly
supported by current evidence, so they do not need to be re-derived in chat.

## Perception Architecture

- Use a staged perception stack, not a monolithic detector.
- Current intended order is:
  - Grounding DINO for sparse open-vocabulary proposals and play-region priors
  - SAM or SAM3 for recovery, refinement, and manual-labeling support
  - YOLO for cheap dense detection once the search space is narrowed or the
    detector path is already known to work
- Detector-first remains the right runtime backbone for player recall.
- Pose is an attribute extractor over retained player boxes, not the only person
  detector.

## Tracking and Runtime

- Tracker backend choice must be explicit, not an invisible library default.
- Current short-sample timing evidence says the default persistent tracker path
  is much slower than ByteTrack.
- That does not automatically make ByteTrack the final default because the speed
  gain may come with more identity fragmentation.
- The correct next step is measured comparison through Layer 1 artifacts and
  scorecards, not an assumption that the faster tracker is always better.

## Identity and Re-ID

- Re-ID is not the next fix for missed players.
- Missing-player recall must be addressed before relying on Re-ID.
- Identity continuity should use fused evidence:
  - appearance
  - geometry
  - temporal continuity
  - team cues
  - jersey evidence
- Full identity resolution should sit above local detections and tracklets, not
  be delegated entirely to the online tracker.

## Ball

- The current ball path should evolve toward a dedicated subsystem.
- Generic full-frame YOLO plus heuristics is not enough by itself.
- The right direction is:
  - initial detection or bootstrap
  - predictive ROI search
  - context-aware recovery when missing
  - smoothing and backward or forward fill
- Player context is legitimate signal for missing-ball inference.

## Action Brain

- Commentary or higher-level event generation should not consume raw noisy
  detector output directly.
- It should consume stabilized game-state features from Layer 1 and possession
  logic.

## MLOps

- Research sources should be cached locally under `.local_wiki/raw/`.
- Synthesized design conclusions should be maintained in
  `docs/research/wiki/pages/`.
- Review presets and evaluation harnesses should be the basis for comparing
  thresholds, model choices, and pipeline order.

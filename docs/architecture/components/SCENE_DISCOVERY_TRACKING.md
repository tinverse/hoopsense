# Scene Discovery And Tracking

This document defines the recommended Layer 1 system design for HoopSense when
processing basketball video from:

- iPhone and Android devices
- DSLR and handheld camera footage
- clips with partial-court visibility
- clips with unstable framing, zoom, or pan
- scenes with spectators, benches, and sideline clutter

It exists because the problem class is not "fixed clean sports broadcast." The
system must work when geometry is weak at the start, objects appear only later,
and better evidence arrives after the initial frame-level interpretation.

## Design Goal

HoopSense should build an evolving game scene over time rather than pretending
each frame can be interpreted independently and finalized immediately.

The architecture must support:

- weak-start perception without perfect calibration
- promptable high-quality discovery for players, ball, hoop, and referee
- cheap temporal continuity between expensive discovery calls
- bounded retrospective repair when later evidence resolves earlier ambiguity
- geometry that improves opportunistically as more court structure becomes visible

## Core Principle

Layer 1 is not a single detector.

It is a staged evidence system:

1. scene prior
2. object discovery
3. temporal tracking
4. bounded retrofit
5. geometry refinement
6. review-time probing

Each stage has a distinct job and should not be collapsed into one opaque
"inference" step.

## Current Repo Status

The staged design in this document is the intended Layer 1 architecture. The
current repository only partially materializes it.

What is implemented today in the Layer 1 artifact path:

- `generate_layer1_annotations.py` emits detection-centric frame records plus
  `bootstrap_context`, `grounding_context`, `scene_prior`, and
  `discovery_proposals`
- `scene_prior` is emitted as a normalized per-frame artifact contract
- `discovery_proposals` currently cover accepted SAM-driven player refinement
  and player recovery outputs only
- the normalized `scene_prior` and `discovery_proposals` contracts are emitted
  for auditability and downstream integration planning

What is not true yet:

- earlier runtime stages do not yet consume `scene_prior` and
  `discovery_proposals` as the authoritative handoff contracts
- the runtime still reads `bootstrap_context` and `grounding_context` directly
- discovery proposals do not yet cover ball, hoop, or referee
- `mask_or_polygon` is currently persisted as a bbox proxy rather than a full
  mask or polygon payload

## Why This Architecture

For imperfect basketball footage:

- calibration may be missing or weak in the first seconds
- the court may be only partially visible
- a player may become recognizable only after a pan or zoom settles
- the ball may be visible only intermittently
- spectators and sideline figures create persistent false positives
- later evidence often explains earlier ambiguity better than the original frame

A calibration-first or detector-only architecture is therefore too brittle.

The system should instead:

- start with soft priors
- discover objects in likely regions
- preserve continuity by default
- repair recent ambiguity when better evidence appears
- strengthen geometry only when the scene supports it

## Component Model

### 1. Video Contract

Responsibility:
- decode video into deterministic frame and timestamp units
- preserve clip identity, frame ordering, and continuity boundaries

Inputs:
- raw clip path

Outputs:
- `FrameRecord`

Required fields:
- `clip_id`
- `frame_idx`
- `t_ms`
- `width`
- `height`
- `discontinuity_label`
- `discontinuity_score`

This stage must be deterministic before any perception logic is trusted.

### 2. Scene Prior

Responsibility:
- answer "where is meaningful basketball activity likely to be?"
- suppress obvious background, crowd, and non-play regions before expensive
  object discovery

Primary implementation direction:
- DINO foreground/play-region priors
- connected-component or irregular-mask region proposals
- optional geometry-informed restriction when calibration is already trustworthy

Non-goals:
- final player detection
- final ball detection
- exact court mapping

Current emitted contract:
- `ScenePrior`

Required fields:
- `frame_idx`
- `prior_status`
- `region_mask_shape`
- `region_mask_grid`
- `proposal_regions`
- `source_model`
- `trigger_reason`

Useful optional fields:
- `crowd_band_regions`
- `top_band_rejected`
- `pan_invalidated`
- `geometry_overlap_score`

Key rule:
- the scene prior should remain irregular and component-based; do not collapse
  it to one rectangle unless the underlying evidence actually is rectangular

Current implementation note:
- the emitted `scene_prior` contract is derived from `bootstrap_context` and
  `grounding_context`, but runtime handoff still occurs through those lower-level
  fields

### 3. Object Discovery

Responsibility:
- turn scene-prior regions or explicit prompts into high-quality object evidence

Primary implementation direction:
- SAM3 for promptable segmentation
- DINO/SAM3 handoff for region-local discovery
- optional detector proposals for cheap seeding

Discovery targets:
- players
- ball
- hoop / rim / backboard
- referee

Target normalized contract:
- `DiscoveryProposal`

Required fields:
- `frame_idx`
- `entity_type`
- `bbox_xyxy`
- `mask_or_polygon`
- `score`
- `source_model`
- `source_prompt`
- `source_region`

Useful optional fields:
- `mask_area_px`
- `mask_area_ratio`
- `anchor_region_kind`
- `source_track_id`
- `source_trigger_reason`

Key rule:
- discovery is expensive and high quality
- it should be invoked selectively, not assumed to be the cheapest primitive in
  the system

Current implementation note:
- current `discovery_proposals` are emitted from accepted SAM-driven player
  refinement and recovery outputs only
- the current artifact stores `mask_or_polygon` as a bbox proxy until persisted
  masks or polygons are part of the supported contract

### 4. Temporal Tracking

Responsibility:
- preserve identity and motion continuity between discovery calls
- keep the scene state alive cheaply over time

Primary implementation direction:
- track state for players and ball
- short-horizon prediction for missing detections
- continuity-aware scoring rather than frame-local re-detection only

Output contract:
- `TrackState`

Required fields:
- `entity_id`
- `entity_type`
- `frame_idx`
- `bbox_xyxy`
- `center_xy`
- `velocity_xy`
- `state`
- `confidence`
- `source`

Useful optional fields:
- `identity_hypothesis_group_id`
- `nearest_player_track_id`
- `missing_gap_frames`
- `motion_mode`

State examples:
- `observed`
- `predicted_short_gap`
- `occluded_likely`
- `missing`

Key rule:
- once an object is found, the default action is to track it, not rediscover it
  globally every frame

### 5. Bounded Retrofit

Responsibility:
- revise a recent span of frames when later evidence clarifies earlier ambiguity

Examples:
- a player enters frame during a pan and becomes clearly visible only later
- the ball reappears after short occlusion and resolves the earlier path
- a merged box later splits into two real players
- a hoop/post anchor becomes visible and disambiguates earlier geometry

Output contract:
- `RetrofitWindow`

Required fields:
- `start_frame_idx`
- `end_frame_idx`
- `entity_ids_affected`
- `repair_reason`
- `repair_source`
- `confidence_delta`

Useful optional fields:
- `repaired_frames`
- `previous_state_summary`
- `updated_state_summary`

Key rule:
- retrofit is local and bounded
- it is not permission to rewrite the entire game repeatedly

Recommended behavior:
- maintain a rolling window for repair
- allow backward adjustment only within that window
- freeze older windows once confidence is sufficient

### 6. Geometry Refinement

Responsibility:
- opportunistically improve court-relative understanding as more anchors become
  visible

Primary sources:
- sideline and baseline evidence
- lane edges and arcs
- hoop / rim / backboard
- stable play-region support

Output contract:
- `GeometryState`

Required fields:
- `frame_idx`
- `geometry_status`
- `h_matrix` or equivalent transform state
- `anchor_count`
- `anchor_families`
- `confidence`

Useful optional fields:
- `court_visibility_ratio`
- `rim_xy`
- `backboard_xy`
- `supporting_prior_id`

Key rule:
- geometry should be consumable as confidence-weighted evidence
- the system must not require full trusted geometry before useful perception can
  begin

### 7. Review-Time Probing

Responsibility:
- let the operator ask targeted questions of one paused frame without rerunning
  the entire clip

Primary implementation direction:
- labeller-side promptable SAM3 frame exploration

Use cases:
- "players on court"
- "basketball player in white"
- "referee"
- "orange basketball"
- "basketball hoop"

Key rule:
- review-time probing is a first-class debugging and data-building surface
- it is not automatically the production runtime path

## Recommended End-To-End Flow

### Player Flow

1. compute `ScenePrior`
2. seed candidate person regions from:
   - detector boxes
   - DINO proposal regions
   - explicit review-time prompts when needed
3. run SAM3 segmentation inside selected regions
4. score player plausibility using:
   - scene prior overlap
   - temporal continuity
   - geometry evidence when available
   - appearance and motion consistency
   - crowd/top-band rejection
5. track players forward
6. retrofit short spans when later evidence is better

### Ball Flow

1. run a ball discovery path using:
   - SAM3 ball prompt
   - optional detector proposals
   - player-conditioned local ROIs when recently possessed
2. choose at most one ball state per frame
3. carry it with a cheap predictive tracker
4. when confidence collapses, re-run discovery
5. retrofit recent ball state when reacquisition clarifies the path

### Hoop / Post Flow

1. periodically probe for:
   - `basketball hoop`
   - `basketball rim`
   - `backboard`
2. choose the most stable prompt behavior
3. track the anchor locally
4. feed it into geometry refinement and later event logic

## Trigger Policy

Expensive discovery should be event-driven.

### Discovery triggers

- clip start
- discontinuity
- strong camera pan
- live-play collapse
- ball lost
- new component entering the scene prior
- review-time explicit user request

### Geometry refresh triggers

- new hoop/rim/post evidence
- stronger court-line evidence
- improved scene prior support
- sustained layout drift

### Retrofit triggers

- object reappearance after short occlusion
- stable identity evidence after ambiguity
- split/merge resolution
- later geometry that changes earlier plausibility

## MVP Runtime Modes

The repo should support explicit operating modes rather than one hidden mode.

### Mode A: review / truth-building

- favors SAM3 and exploratory prompts
- latency is secondary
- best for failure analysis and dataset building

### Mode B: offline batch inference

- uses scene prior + selective SAM3 + tracking
- favors quality over strict real-time performance

### Mode C: edge / live inference

- uses cached priors and tracked state aggressively
- invokes expensive discovery only on triggers
- may omit some review-only probing surfaces

## Failure Modes The Architecture Must Handle

- partial court only
- handheld pan and zoom
- late-entering players
- spectators mistaken for players
- players merged into one box
- ball visible but tiny
- ball temporarily absent
- imperfect geometry early in the clip
- later evidence contradicting earlier frame-local interpretation

The system design is correct only if these failure modes are expected and given
an explicit place to be resolved.

## Relationship To Existing Docs

- [LAYERED_FEATURE_SCHEMA.md](./LAYERED_FEATURE_SCHEMA.md)
  - defines the broader feature ladder and downstream ownership split
- [FEATURE_SCHEMA_V2.md](./FEATURE_SCHEMA_V2.md)
  - remains the frozen Action Brain tensor contract
- [BALL_STATE_STAGE1.md](./BALL_STATE_STAGE1.md)
  - defines the narrow ball-state substrate inside this larger architecture
- [LAYER1_IDENTITY_RULES.md](./LAYER1_IDENTITY_RULES.md)
  - defines the continuity-first identity policy consumed by tracking and retrofit

## Near-Term Implementation Order

1. keep video/frame contracts deterministic
2. strengthen `ScenePrior` with DINO region contracts
3. formalize SAM3 discovery contracts for players and ball
4. keep tracking conservative and continuity-first
5. add bounded retrofit windows and provenance
6. opportunistically improve geometry using stable anchors
7. expose review-time probing in the labeller

## Non-Goals

This document does not claim that HoopSense already has:

- robust full-game global optimization
- final real-time Orin performance
- final player, ball, or hoop models
- perfect calibration-free court mapping

It defines the architecture that should guide the next implementation slices.

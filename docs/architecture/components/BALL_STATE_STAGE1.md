# Ball State Stage 1

This document defines the narrowest next implementation for basketball ball
evidence in HoopSense.

It exists because a real long-game partial inference run on
`mindright_dynamite_7_8.mp4` produced player rows but zero ball rows, zero
attributed events, and therefore zero MVP stat output. The current blocker is
not the reporting ladder. The blocker is weak ball evidence.

## Goal

Stage 1 does **not** try to solve full ball tracking, possession, or shot
classification. It does one smaller thing:

- materialize a conservative, auditable `ball_state`
- keep it stable across short misses
- expose it in Layer 1 review artifacts and runtime JSONL
- make it usable as evidence for later `shot_attempt`, `made/missed_shot`, and
  `rebound` attribution

## Why This Stage

The current repo already has:

- YOLO class `32` requested in both review and runtime paths
- a minimal per-frame `ball_detection` in Layer 1 artifacts
- a minimal per-frame `ball` row in runtime inference
- deterministic MVP event rules that already expect ball-aware evidence

What it does **not** yet have is a stable ball substrate.

Stage 1 is therefore contract-and-smoothing first, not model-replacement first.

## Primary-Source Takeaways

The following primary sources were used to shape this plan:

- TrackNet: temporal sports-ball tracking from short frame windows
  - <https://arxiv.org/abs/1907.03698>
- TrackNetV4: motion-aware feature fusion for fast small-object tracking
  - <https://arxiv.org/abs/2409.14543>
- DeepSportLab: unified team-sports scene model with explicit ball prediction
  - <https://arxiv.org/abs/2112.00627>

Useful implications for HoopSense:

- Raw frame-by-frame ball detection is too brittle for small, fast, occluded
  sports objects.
- Motion history is a first-class signal, not a nice-to-have.
- Team-sports scenes benefit from a dedicated ball contract rather than treating
  the ball as just another COCO class.
- For HoopSense, the smallest viable next step is not a new end-to-end model.
  It is a conservative temporal ball-state layer sitting on top of existing
  detections.

## Ball State Contract

Stage 1 should add a dedicated `ball_state` object per frame in both:

- Layer 1 review artifacts
- runtime JSONL output

### Frame-level fields

Required:

- `state`
  - one of:
    - `observed`
    - `predicted_short_gap`
    - `missing`
- `confidence`
  - score for the selected observation after candidate scoring
- `center_xy`
  - pixel-space center
- `bbox_xywh`
  - best selected bounding box when observed
- `court_xy`
  - projected court position when homography exists
- `velocity_xy`
  - smoothed image-space velocity
- `speed_px`
  - scalar speed in pixels/frame
- `missing_gap_frames`
  - number of consecutive frames since last direct observation
- `source`
  - one of:
    - `detector`
    - `smoothed_prediction`
- `candidate_count`
  - number of raw ball candidates seen this frame

Useful optional fields:

- `nearest_player_track_id`
- `nearest_player_distance_px`
- `candidate_scores`
  - compact audit trail for top candidates

### Artifact/runtime provenance

Each artifact or JSONL session should also record the policy or thresholds used
for ball-state selection and smoothing under a top-level metadata block such as:

- `postprocess.ball_state_policy`

## Candidate Selection Logic

Stage 1 should remain conservative and simple.

Per frame:

1. collect raw class-`32` detections
2. score each candidate
3. select at most one best candidate if it clears threshold
4. otherwise fall back to short-gap prediction if allowed
5. otherwise mark the state as `missing`

### Candidate score inputs

Use only signals already available or cheap to compute:

- raw detector confidence
- distance to previous selected/predicted ball center
- size sanity
  - reject implausibly large or tiny boxes
- velocity continuity
  - penalize impossible jumps
- optional player-proximity prior
  - modest bonus if near the current handler or a plausible interacting player

### What not to do in Stage 1

- do not add a new learned ball detector yet
- do not create multi-ball hypotheses
- do not overfit by hardcoding clip-specific positions
- do not let a low-confidence stray detection override continuity

## Smoothing / Persistence

Stage 1 smoothing should be deliberately narrow:

- constant-velocity or nearly constant-velocity state
- short prediction window only
- bounded by explicit physics constraints

Recommended behavior:

- if a direct observation exists and passes score threshold:
  - update the smoothed state
- if no observation exists:
  - predict for up to `1-5` frames
  - keep `state = predicted_short_gap`
- after the gap limit:
  - set `state = missing`

This is enough to stabilize handoff into later shot/rebound evidence without
pretending we already have robust ball control tracking.

## Physics / Evidence Constraints

Stage 1 should enforce only the constraints that are already defensible:

- no teleportation without discontinuity
- bounded displacement per frame
- bounded acceleration
- continuity preferred over abrupt jumps
- short occlusion more likely than true disappearance

These constraints should be implemented as score penalties or hard rejection
gates for candidate selection.

## Integration Points

### Layer 1 review artifacts

Files:

- `tools/review/labeller/generate_layer1_annotations.py`
- `tools/review/labeller/static/app.js`

Changes:

- replace or supersede raw `ball_detection` with `ball_state`
- render predicted and observed states differently in the labeller
- surface `state`, `confidence`, `missing_gap_frames`, and nearest-player info
- use `ball_state` instead of raw detector confidence in live-play gating

### Runtime inference

Files:

- `pipelines/inference.py`
- later possibly a dedicated helper module such as `pipelines/ball_state.py`

Changes:

- replace `last_ball_2d` plus naive per-frame update with a real `ball_state`
- emit JSONL `ball` rows from the selected/smoothed state, not every raw class
  `32` detection
- make `construct_features_v2(...)` consume the selected/smoothed ball center,
  not the last raw detection

### MVP event attribution

Files:

- `pipelines/inference.py`
- `pipelines/mvp_event_engine.py`
- `specs/mvp_event_rules.yaml`

Stage 1 unlocks better evidence for:

- `shot_attempt`
  - recent observed or predicted ball continuity plus release-like separation
- `made_shot` / `missed_shot`
  - not fully solved in Stage 1, but now possible to build from stable ball
    presence near the rim and subsequent trajectory
- `rebound_off` / `rebound_def`
  - detect loose-ball descent followed by recovery by a player

## Staged Implementation Plan

### Slice A: contract and review visibility

Purpose:

- define `ball_state`
- emit it in Layer 1 artifacts
- show it in the labeller

Acceptance:

- representative artifacts contain `ball_state`
- review UI shows observed vs predicted vs missing state

### Slice B: conservative runtime ball-state adapter

Purpose:

- replace naive `extract_ball_state(...)` with scored candidate selection and
  short-gap persistence

Acceptance:

- real-game runtime emits ball rows on frames where evidence is plausible
- raw stray detections do not dominate the stream

### Slice C: use ball-state in live-play and shot-attempt evidence

Purpose:

- promote live-play from raw `ball_detection` to stable `ball_state`
- require stable ball continuity for `shot_attempt`

Acceptance:

- reviewed live-play clips show more meaningful ball-aware reasoning
- runtime produces more credible `shot_attempt` candidates

### Slice D: unlock rebound-result evidence

Purpose:

- use post-shot ball continuity and nearest-player recovery to emit unresolved
  `missed_shot` and rebound candidates

Acceptance:

- first auditable rebound candidates appear in JSONL

## Explicit Non-Goals

Stage 1 does not promise:

- perfect ball detection on every frame
- full 3D ball trajectory
- possession resolution
- made-vs-missed adjudication by itself
- a new trained ball-specific model

It is a substrate step.

## Recommended Next Task

The next implementation slice should be:

- materialize `ball_state` as a checked-in contract in Layer 1 artifacts and
  runtime JSONL
- implement conservative candidate scoring and short-gap persistence
- then rerun the real-game 10-minute inference sample and compare:
  - ball row count
  - shot-attempt candidate count
  - first attributed ball-aware events

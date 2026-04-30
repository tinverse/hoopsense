# HoopSense Subsystem Map

This page maps the main HoopSense subsystems to the papers that are most useful
for implementation decisions, not just general background reading.

## 1. Ball Tracking

### Primary cached papers

- TrackNet: `.local_wiki/raw/tracknet-ball-tracking/`
- WASB-SBDT: `.local_wiki/raw/wasb-sbdt/`
- TOTNet: `.local_wiki/raw/totnet-ball-tracking/`
- Ball trajectory from multi-agent contexts: `.local_wiki/raw/ball-trajectory-from-context/`

### How to use them

- `TrackNet` is the classic specialized small-fast-ball baseline. It matters
  because it shows the value of temporal context for a tiny, blurred object.
- `WASB-SBDT` is the stronger multi-sport baseline and is more relevant than
  TrackNet alone for a modern general sports-ball stack.
- `TOTNet` is useful specifically for occlusion-heavy offline analysis and gives
  a better read on what a dedicated ball path can do when visibility degrades.
- `Ball trajectory from multi-agent contexts` is important because it treats the
  ball as inferable from player context, which is directly relevant when the
  ball is intermittently invisible in basketball.

### HoopSense implication

- MVP path: keep the current staged detector plus predictive ROI plus smoothing
  path, but move toward a dedicated ball subsystem instead of only relying on
  generic YOLO plus heuristics.
- Near-term best fit:
  - specialized ball detector or tracker at sparse/key moments
  - player-context-conditioned recovery when the ball is missing
  - short-gap trajectory smoothing and backward or forward fill
- The multi-agent trajectory inference paper is especially relevant for future
  missing-ball imputation, not only visible-ball detection.

## 2. Player Tracking and Local Identity Continuity

### Primary cached papers

- DeepSORT: `.local_wiki/raw/deepsort/`
- ByteTrack: `.local_wiki/raw/bytetrack/`
- BoT-SORT: `.local_wiki/raw/botsort/`
- MOT literature review: `.local_wiki/raw/mot-literature-review/`

### How to use them

- `DeepSORT` is the clean reference for motion plus appearance association.
- `ByteTrack` matters because it recovers tracking quality by associating low
  confidence detections instead of dropping them too early.
- `BoT-SORT` matters because it adds stronger motion and appearance handling,
  but it is also more expensive in runtime and may be the source of timing pain
  in the current Layer 1 loop.
- The `MOT review` is the taxonomy and evaluation background.

### HoopSense implication

- We should treat tracker backend choice as an explicit system decision, not a
  hidden Ultralytics default.
- ByteTrack currently looks attractive for Layer 1 runtime because it is
  materially cheaper than the default persistent tracker path.
- Generic MOT is not enough for basketball. HoopSense still needs:
  - court constraints
  - segment-discontinuity handling
  - engagement filtering
  - bounded global identity stitching or MHT

## 3. Re-Identification

### Primary cached papers

- Person Re-ID survey: `.local_wiki/raw/person-reid-survey/`
- Occluded person Re-ID survey: `.local_wiki/raw/occluded-person-reid-survey/`
- DeepSORT: `.local_wiki/raw/deepsort/`
- BoT-SORT: `.local_wiki/raw/botsort/`

### How to use them

- The general and occluded Re-ID surveys define the evidence families and common
  failure modes.
- DeepSORT and BoT-SORT show how Re-ID enters an online tracker, but not how to
  solve a court-aware single-camera sports identity problem end to end.

### HoopSense implication

- Re-ID should be one evidence source in a fused identity model.
- Appearance alone is too weak because of:
  - similar uniforms
  - motion blur
  - occlusion
  - partial bodies
  - camera zoom and pan
- The correct architecture remains:
  - high-recall detections first
  - temporal smoothing and tracklets
  - fused identity evidence with geometry, team, jersey, and continuity

## 4. Action Brain and Commentary

### Primary cached papers

- Sports action recognition survey: `.local_wiki/raw/sports-action-recognition-survey/`
- NSVA large-scale sports video: `.local_wiki/raw/nsva-large-scale-sports-video/`
- Sports video analysis survey: `.local_wiki/raw/sports-video-analysis-survey/`

### How to use them

- The action-recognition survey gives the taxonomy for moving from framewise
  perception to event understanding.
- `NSVA` is important because it treats sports video as a large-scale modeling
  problem and includes captioning and salient-player tasks.
- The older sports video survey is the higher-level content-analysis backdrop.

### HoopSense implication

- The commentary system should not be built directly from raw detector output.
- It should consume stabilized game-state features:
  - player tracklets
  - ball state and inferred possession
  - court context
  - event proposals
- `NSVA` is relevant as a reference point for future captioning-style or
  sequence-model heads, but the MVP still depends on building better structured
  features from Layer 1 and possession logic first.

## 5. What This Means For The Current Build Order

1. Strengthen the Layer 1 runtime and evaluation harness.
2. Make tracker backend choice explicit and measured.
3. Continue toward a dedicated ball subsystem with context-aware recovery.
4. Add fused Re-ID and tracklet-level identity reasoning only after detection
   and smoothing are stable enough.
5. Feed the stabilized state into possession and event layers before attempting
   natural-language commentary generation.

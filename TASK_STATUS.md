# HoopSense Task Status

This file is a human-readable execution snapshot derived from `docs/plan/PLAN_TREE.yaml`.

The old chapter-style status model is retired. This file now tracks the current plan frontier and active L-level work only.

## Current Frontier

The current highest-priority frontier is:
- `L3.96` define an explicit DINO to SAM to YOLO grounding handoff contract for Layer 1 bootstrap and segment re-grounding
  - `L3.97` rerun person YOLO inside DINO-grounded play-region policy after segment bootstrap and discontinuity refresh
  - `L3.98` surface grounding-policy provenance and regression coverage in Layer 1 artifacts and review tooling
  - `L3.99` trigger segment regrounding on sustained live-play detection collapse without requiring a hard discontinuity
  - `L3.100` route SAM recovery proposals through explicit grounding-anchor regions instead of only implicit bootstrap blobs
  - `L3.117` materialize a per-frame `scene_prior` contract from bootstrap and grounding contexts in Layer 1 artifacts
  - `L3.118` materialize a per-frame `discovery_proposals` contract from accepted SAM-driven recovery and refinement outputs
  - `L3.119` emit an artifact-level staged perception summary so the DINO to SAM to YOLO handoff is auditable
    - `L3.122` summarize clip-level staged handoff evidence across bootstrap, grounded reruns, SAM adoption, identity repair, and ball recovery
    - `L3.123` re-run the current review clip with the staged summary enabled and verify the emitted artifact fields
  - `L3.121` consume `scene_prior` and `discovery_proposals` directly in Layer 1 scoring and short-gap identity repair
- `L3.101` emit raw pre-filter ball detections in Layer 1 artifacts so detector recall can be audited separately from ball-state heuristics
- `L3.102` add a player-conditioned fallback ROI ball detector pass when full-frame ball detection is absent or weak
- `L3.103` add a predictive ball-search layer that uses recent ball kinematics and nearby-player context before broad fallback reacquisition
  - `L3.104` define a coarse ball motion-mode contract (`carry_or_hold`, `dribble_like`, `pass_or_loose`, `unknown_recent`) for ROI shaping
  - `L3.105` run recent-state predictive ROI search between full-frame scan and broad player-local reacquisition
  - `L3.106` emit predictive-search provenance in Layer 1 artifacts so per-frame ball recovery decisions are auditable
  - `L3.107` compare predictive ROI recovery against prior broad-fallback audits on reviewed visible-ball clips
  - `L3.128` backfill pre-first-observation frames by walking backward from the first credible ball detection with reverse ROI search and player-local fallback
    - `L3.129` re-run the representative clip and verify the artifact backfills early ball frames while preserving player and jersey outputs
- `L3.108` add an experimental SAM3 text-prompted basketball detector path for visual ball-recall evaluation on reviewed clips
  - `L3.109` render a web-safe MP4 overlay of SAM3 basketball detections for direct visual review
  - `L3.110` use Layer 1 active-player candidates as SAM3 ROI seeds for segmented active-player review
  - `L3.111` render a web-safe MP4 overlay of SAM3 active-player masks for direct visual review
  - `L3.112` add a frame-local SAM3 detect endpoint to the labeller for paused-frame promptable segmentation
  - `L3.113` add labeller UI controls for prompt entry, detect, and overlay display of same-frame SAM3 results
  - `L3.124` use SAM3 basketball detections to bootstrap the first credible runtime ball state before predictive ROI handoff
  - `L3.125` re-enter SAM3 ball reacquisition only after sustained missing-ball evidence or segment reset and emit handoff provenance
  - `L3.126` align image-model SAM checkpoint defaults with the upstream SAM3 image-builder contract and remove partial-load warnings
    - `L3.127` move experimental SAM and OCR model caches into a repo-local git-ignored mount so Docker runs do not re-download weights
- `L3.93` add a rollback-safe experimental Orin Docker image variant on JetPack 6 / CUDA 12.x for official SAM 3 validation
  - `L3.94` containerize the full Layer 1 artifact generation path so the existing pipeline can be exercised end to end inside the SAM 3-capable image
  - `L3.95` add a bounded GPU validation and rollback workflow for the experimental Orin SAM 3 image without mutating the stable Orin runtime
- `L3.89` add optional SAM 3-assisted player recovery on unexplained DINO play-region blobs and ambiguous YOLO ROIs in Layer 1 artifacts
  - `L3.90` route official SAM 3 text-prompted ROI refinement through the Layer 1 artifact pipeline with explicit fail-closed runtime statuses
  - `L3.91` keep Orin setup honest about SAM 3 runtime prerequisites and skip installation when the pinned Python runtime is too old
  - `L3.92` add regression coverage for SAM ROI proposal generation and recovered-detection wiring without requiring live SAM weights
- `L3.88` split the labeller overlay into promoted versus demoted detection states so review can preserve raw evidence without confusing it for trusted on-court players
- `L3.87` discount camera-pan-induced apparent motion in Layer 1 player plausibility scoring so seated spectators are not promoted by shared frame drift
- `L3.86` add multi-signal on-court player plausibility scoring so spectators and merged detections are down-ranked before active-player promotion
- `L3.84` make DINO play-region priors ephemeral and invalidate them on strong camera pan or layout drift
- `L3.85` feed the current play-region prior into geometry fitting so court evidence is restricted to likely in-play regions
- `L3.72` extend attributed MVP event emission to shot and rebound evidence once ball-result signals are available
- `L3.66` add bounded multi-hypothesis identity infrastructure for ambiguous short-gap track continuity
  - `L3.132` run a bounded longer-clip BoT-SORT evaluation outside the checked-in runtime path and measure track fragmentation and suspicious identity jumps before deciding on any tracker work
  - `L3.133` formalize the Layer 1 player identity evidence model and hard constraints into a machine-readable policy and emitted artifact contract
  - `L3.134` implement explicit Layer 1 identity decision staging with hard-filtered candidate records, soft-evidence scoring, and selected-link provenance
  - `L3.135` extract Layer 1 identity-resolution helpers into a dedicated module and route runtime short-gap repair through the explicit contract
  - `L3.136` remove obsolete labeller identity-resolution wrappers and make tests and runtime call the extracted module directly
  - `L3.137` emit explicit temporal-overlap and max-gap hard-rejection records for identity-link candidates instead of silently dropping them
  - `L3.138` extract short-gap identity repair mechanics into a dedicated Layer 1 tracklet stitcher module
  - `L3.139` validate identity-link provenance on a bounded real clip after the stitcher extraction and hard-rejection contract cleanup
  - `L3.140` bound temporal-overlap candidate generation so identity provenance keeps near-boundary overlap evidence without flooding the ledger with irrelevant overlapping tracklet pairs
  - `L3.141` bound max-gap candidate generation so identity provenance keeps near-boundary over-gap evidence without flooding the ledger with irrelevant long-distance tracklet pairs
  - `L3.142` enumerate bounded connected-component assignment hypotheses for identity links instead of relying on greedy pairwise link selection
  - `L3.143` emit alternative assignment hypotheses and ambiguity margins per identity conflict group in Layer 1 artifacts
  - `L3.144` validate bounded assignment-hypothesis identity selection on a reviewed real artifact and compare it against prior greedy pairwise selection
  - `L3.145` sweep reviewed Layer 1 artifacts for nontrivial identity conflict components that actually exercise bounded assignment-hypothesis selection
  - `L3.146` compare bounded assignment-hypothesis selection against a greedy edge baseline on the first reviewed artifact with a nontrivial conflict component
- `L3.64` add a minimal ball artifact to Layer 1 review outputs and use it to refine live-play gating
- `L3.63` add explicit playback transport controls and conservative jersey-OCR display gating in the labeller
- `L3.62` add a rule-based live-play/dead-ball gate with per-frame scores, stitched segments, and labeller review support
- `L3.53` add bidirectional short-gap repair for clustered or briefly missed player tracks in Layer 1 artifacts
- `L3.61` add a coarse torso-color appearance cue to further suppress seated spectators and sideline bystanders
- `L3.46` check in canonical MVP stats header and formula templates with provenance rules
- `L3.51` define a HoopSense-native declarative rules layer for stat attribution without reusing Shush implementation
- `L3.47` map each MVP stat to required event primitives and possession fields
- `L3.9` define `PossessionContext` fields and ledger serialization contract
- `L3.10` track `ballhandler_id`, `dribble_count`, and `pass_count` for one possession slice
- `L3.12` define the first stat-ready event set: `pass`, `catch`, `dribble`, `shot_attempt`, `rebound`, `turnover`
- `L3.48` materialize auditable box-score rows that match the core scorebook columns and formulas
- `L3.49` add scorebook-backed validation fixtures that compare generated stats to trusted totals
- `L3.50` defer shot-profile and advanced report outputs until the core box score is reliable
- `L3.37` decide the first migration target, if any, based on performance, determinism, and implementation risk
- `L3.24` make Docker docs explicitly cloud-oriented and not the native Orin story
- `L3.26` publish separate runbooks for cloud/x86 training and Jetson/ARM64 runtime validation
- `L3.29` define functional-core CI ownership and path triggers
- `L3.30` add a lightweight integration pipeline that depends on selected core workflows
- `L3.31` gate training smoke/evaluation workflows behind ML-specific changes or manual triggers
- `L3.32` define which GCP resources stay on direct `gcloud` versus move into Terraform
- `L3.33` add initial Terraform layout for shared buckets, registry, and service accounts
- `L3.20` add first slice-based evaluation outputs
- `L3.27` define dataset version lineage and promotion rules across Oracle/synthetic training sets
- `L3.28` define multi-arch or explicitly split cloud-vs-Orin container strategy

## L0 Product Goal

- [>] Trustworthy basketball understanding from video

## L1 Workstreams

- [>] Perception and geometry foundation
- [>] Action Brain and synthetic Oracle training loop
- [>] Possession context and game-state reasoning
- [>] Event attribution and stats generation
- [>] Deployment, runtime, and operator workflow
- [>] Collaboration, review, and project control
- [>] MLOps governance and model lifecycle control
- [>] DevOps reproducibility and delivery control

## L2 Status By Workstream

### Perception and Geometry Foundation
- [x] Multi-source ingestion and deterministic video contracts
- [x] Court geometry, homography, and camera state
- [>] Identity fusion, tracking, and temporal continuity
- [>] Pose estimation and kinematic lifting
- [>] Perception-and-geometry readiness gate
- [>] Python-versus-Rust boundary decision for perception and geometry

### Action Brain and Synthetic Oracle
- [x] Stable Action Brain feature contract (`features_v2`)
- [>] Oracle MoCap ingestion, FK, and dataset generation
- [>] Action Brain training, evaluation, and checkpoint lifecycle
- [ ] Sim-to-real fine-tuning and validation loop

### Possession Context and Game-State Reasoning
- [>] Possession context contract in the ledger
- [>] Ball control, dribble count, and pass-chain tracking
- [>] Offense zone, transition, and drive semantics
- [ ] Referee-assisted rewind and correction logic

### Event Attribution and Stats Generation
- [>] Event attribution rules combining motion, ball, and possession state
- [>] Stat ledger and box-score generation
- [ ] Shot chart, defensive metrics, and report outputs

### Deployment, Runtime, and Operator Workflow
- [>] Native Orin environment and ARM64 runtime validation
- [>] Cloud/x86 training environment and artifact parity
- [ ] Real-time inference pipeline and model optimization

### Collaboration, Review, and Project Control
- [x] Gemini/Codex collaboration bridge and review workflow
- [x] Plan-driven execution and task synchronization
- [>] Documentation and task-status alignment

### MLOps Governance and Model Lifecycle Control
- [>] Dataset manifests, lineage, and validation policy
- [>] Training run lineage and checkpoint lifecycle
- [ ] Slice-based evaluation and drift monitoring
- [ ] Deployment compatibility reporting across cloud and edge targets

### DevOps Reproducibility and Delivery Control
- [>] Guix-first development and environment reproducibility
- [>] Docker fallback packaging and cloud image discipline
- [>] CI quality gates for code, contracts, and docs
- [ ] Cloud and edge delivery guidance with explicit target boundaries
- [ ] Multi-pipeline CI architecture by functional core
- [ ] Terraform adoption for stable shared GCP infrastructure

## Key L3 Execution Slices

### Completed
- [x] Preserve the Oracle MVP parser/FK path for one fixture motion
- [x] Maintain a reproducible GPU training smoke test and checkpoint write path
- [x] Require plan-tree updates before substantial architecture or implementation changes
- [x] Define measurable readiness checks for ingestion, tracking, pose, geometry, and lifting
- [x] Add a perception-quality report artifact for representative clips
- [x] Materialize one real Layer 1 annotation artifact from GPU-backed Ultralytics inference on a representative 5-second clip
- [x] Render real detection, track, and pose overlays in the labeller from a persisted annotation artifact
- [x] Verify the Orin GPU container command used for representative Layer 1 artifact generation
- [x] Add coarse light-versus-dark uniform bucket estimation to representative Layer 1 perception artifacts
- [x] Add active-player scoring to suppress spectators and bench-side false positives in Layer 1 artifacts
- [x] Add a rule-based live-play/dead-ball gate with per-frame scores, stitched segments, and labeller review support
- [x] Refactor `extract_game_dna` into focused inference pipeline classes without changing output contract
- [x] Add video discontinuity detection and continuity segments to Layer 1 artifacts before stronger identity repair
- [x] Check in an explicit Layer 1 identity-policy contract and machine-readable rules file
- [x] Check in machine-readable MVP event-evidence rules and a loader for deterministic stat attribution
- [x] Make the first deterministic event-attribution path consume the checked-in MVP event rules
- [x] Emit first attributed MVP event payloads in the shape required by the deterministic event-rule engine
- [x] Remove stale legacy labeling from the active inference path while preserving explicit compatibility fallbacks
- [x] Emit running MVP stat-update rows from deterministic attributed events in the inference pipeline
- [x] Emit per-player MVP stat snapshots from accumulated running totals in the inference pipeline
- [x] Emit a terminal per-game MVP stat sheet snapshot from accumulated deterministic totals
- [x] Define a staged Stage-1 ball-state contract and implementation plan from primary-source review
- [x] Materialize Stage-1 `ball_state` selection and short-gap persistence in runtime and Layer 1 artifacts
- [x] Introduce a separate runtime and review ball-detector model instead of relying on the pose model for class-32 detections
- [x] Add DINOv3 availability to the cloud/Docker environment for future bootstrap segmentation work
- [x] Add a rollback-safe experimental Orin Dockerfile variant for DINOv3 without mutating the stable Orin image
- [x] Add an optional DINOv3 bootstrap foreground/background pre-pass to the Layer 1 artifact workflow
- [x] Port segment-aware DINOv3 bootstrap context into the main inference loop and rebootstrap after discontinuities
- [ ] Make DINO play-region priors ephemeral and invalidate them on strong camera pan or layout drift
- [ ] Feed the current play-region prior into geometry fitting so court evidence is restricted to likely in-play regions
- [ ] Add a minimal ball artifact to Layer 1 review outputs and use it to refine live-play gating
- [ ] Add bounded multi-hypothesis identity infrastructure for ambiguous short-gap track continuity
  - [x] Formalize the Layer 1 player identity evidence model and hard constraints into a machine-readable policy and emitted artifact contract
  - [x] Implement explicit Layer 1 identity decision staging with hard-filtered candidate records, soft-evidence scoring, and selected-link provenance
  - [x] Sweep reviewed Layer 1 artifacts for nontrivial identity conflict components that actually exercise bounded assignment-hypothesis selection
  - [x] Compare bounded assignment-hypothesis selection against a greedy edge baseline on the first reviewed artifact with a nontrivial conflict component
  - [x] Build bounded global identity hypotheses across ambiguous local conflict groups instead of exposing only isolated group-level alternatives
  - [x] Emit per-track canonical-identity options induced by the top bounded global identity hypotheses for review and downstream consumers
  - [x] Keep runtime stitching conservative while exposing unresolved multi-hypothesis identity state in Layer 1 artifacts and summaries
  - [x] Make the jersey-OCR consensus path consume bounded identity track options instead of only committed identity_track_id assignments
  - [x] Emit per-track jersey option consensus for ambiguous identity tracks without forcing runtime identity collapse
  - [x] Surface hypothesis-aware jersey consensus counts and option summaries in Layer 1 artifacts and tests
  - [x] Re-rank bounded global identity hypotheses using jersey option consensus as later-pass evidence without mutating committed runtime identities
  - [x] Emit conservative jersey-preferred canonical track annotations per detection from the selected later-pass global identity world
  - [x] Surface jersey-aware global identity resolution summaries and tie-break regressions in Layer 1 artifacts and tests
  - [x] Surface base versus jersey-reranked global identity world status in the labeller UI for reviewed Layer 1 artifacts
  - [x] Show per-track preferred canonical identity changes caused by jersey-aware reranking in the labeller track inspector and selectors
  - [x] Add labeller regressions or frontend wiring checks for the new identity-world review surfaces and align plan/docs
- [x] Materialize per-detection identity option summaries from bounded global hypotheses in stitched Layer 1 frames
- [x] Preserve ambiguous track-option provenance on repaired and unrepaired detections without mutating committed identities
- [x] Add regressions and artifact summary fields for runtime identity-option annotations
- [x] Surface ambiguity-aware identity counts in clip-level staged perception summaries instead of only committed repaired identity totals
- [x] Make regression audit reports consume bounded identity-option fields and preferred-canonical deltas instead of only committed canonical IDs
- [x] Add regressions for ambiguity-aware staged identity summaries and audit signals
- [x] Raise review-clip generation defaults from 5 seconds to minute-scale windows so labeller review can inspect detection progression over time
- [x] Expose explicit clip-duration controls in review-clip sampling and slicing utilities instead of hard-coding short windows
- [x] Add regressions for minute-scale review-clip generation defaults and duration overrides
- [ ] Add explicit playback transport controls and conservative jersey-OCR display gating in the labeller
- [ ] Add bidirectional short-gap repair for clustered or briefly missed player tracks in Layer 1 artifacts
- [x] Attach jersey-number evidence and consensus fields to persistent player identity without rewriting raw tracker IDs
- [ ] Add a coarse torso-color appearance cue to further suppress seated spectators and sideline bystanders
- [x] Add Kalman-smoothed track state and motion-aware scoring to Layer 1 artifacts
- [x] Support typed partial-court calibration landmarks and solver flow in the labeller without depending on visible corners
- [x] Add an explicit general-note save path in the labeller so freeform perception review notes are not silently lost
- [x] Replace named-point calibration clicks with primitive-based line and arc sampling in the labeller
- [x] Normalize labeller feedback into a categorized regression fixture for Layer 1 tracking and postprocess work
- [x] Add a regression-audit report over reviewed Layer 1 failure cases to guide tracking changes without clip-specific overfitting
- [x] Add a structured perception-feedback workflow for false positives, misses, merges, and track errors in the labeller
- [x] Verify actual CUDA acceleration in the Orin validation path and publish a repeatable probe artifact
- [x] Publish a mobile-friendly external HoopSense demo page with one representative clip, overlay, and feedback prompt
- [x] Publish a normalized MVP stats contract for a sellable scorebook-style output
- [x] Document the staged perception system design for imperfect basketball video with partial-court visibility
- [x] Define explicit Layer 1 contracts for scene priors, discovery, tracking, retrofit, and geometry refinement
- [x] Align the architecture blueprint and layered feature schema with the staged discovery-and-tracking design
- [x] Reconcile architecture, DevOps, and inline Layer 1 documentation with the current emitted staged-perception contract
- [ ] Scale Oracle ingestion to Subject 124 while preserving `features_v2`
- [ ] Define `PossessionContext` fields and ledger serialization contract
- [ ] Track `ballhandler_id`, `dribble_count`, and `pass_count` for one possession slice
- [ ] Define the first stat-ready event set: `pass`, `catch`, `dribble`, `shot_attempt`, `rebound`, `turnover`
- [ ] Generate MVP box-score rows from attributed events
- [ ] Check in canonical MVP stats header and formula templates with provenance rules
- [ ] Define a HoopSense-native declarative rules layer for stat attribution without reusing Shush implementation
- [ ] Map each MVP stat to required event primitives and possession fields
- [ ] Materialize auditable box-score rows that match the core scorebook columns and formulas
- [ ] Add scorebook-backed validation fixtures that compare generated stats to trusted totals
- [ ] Defer shot-profile and advanced report outputs until the core box score is reliable
- [ ] Implement offense zone and transition flag derivation in the possession context
- [ ] Document candidate Python-versus-Rust ownership split for the perception layer
- [ ] Implement first-pass dataset manifests with SHA-256 hashing
- [ ] Integrate training lineage recording (git commit + data hash) into training loop
- [ ] Establish initial CI plumbing (.github/workflows/ci.yml)
- [ ] Implement dynamic perception audit script (scripts/run_perception_audit.sh)

### Active or Next
- [ ] Define an explicit DINO to SAM to YOLO grounding handoff contract for Layer 1 bootstrap and segment re-grounding
  - [ ] Rerun person YOLO inside DINO-grounded play-region policy after segment bootstrap and discontinuity refresh
  - [ ] Surface grounding-policy provenance and regression coverage in Layer 1 artifacts and review tooling
  - [ ] Trigger segment regrounding on sustained live-play detection collapse without requiring a hard discontinuity
  - [ ] Route SAM recovery proposals through explicit grounding-anchor regions instead of only implicit bootstrap blobs
  - [ ] Materialize a per-frame `scene_prior` contract from bootstrap and grounding contexts in Layer 1 artifacts
  - [ ] Materialize a per-frame `discovery_proposals` contract from accepted SAM-driven recovery and refinement outputs
  - [x] Emit an artifact-level staged perception summary so the DINO to SAM to YOLO handoff is auditable
    - [x] Summarize clip-level staged handoff evidence across bootstrap, grounded reruns, SAM adoption, identity repair, and ball recovery
    - [x] Re-run the current review clip with the staged summary enabled and verify the emitted artifact fields
  - [x] Consume `scene_prior` and `discovery_proposals` directly in Layer 1 scoring and short-gap identity repair
- [ ] Emit raw pre-filter ball detections in Layer 1 artifacts so detector recall can be audited separately from ball-state heuristics
- [ ] Add a player-conditioned fallback ROI ball detector pass when full-frame ball detection is absent or weak
- [ ] Add a predictive ball-search layer that uses recent ball kinematics and nearby-player context before broad fallback reacquisition
  - [ ] Define a coarse ball motion-mode contract (`carry_or_hold`, `dribble_like`, `pass_or_loose`, `unknown_recent`) for ROI shaping
  - [ ] Run recent-state predictive ROI search between full-frame scan and broad player-local reacquisition
  - [ ] Emit predictive-search provenance in Layer 1 artifacts so per-frame ball recovery decisions are auditable
  - [ ] Compare predictive ROI recovery against prior broad-fallback audits on reviewed visible-ball clips
  - [x] Backfill pre-first-observation frames by walking backward from the first credible ball detection with reverse ROI search and player-local fallback
    - [x] Re-run the representative clip and verify the artifact backfills early ball frames while preserving player and jersey outputs
- [ ] Add an experimental SAM3 text-prompted basketball detector path for visual ball-recall evaluation on reviewed clips
  - [ ] Render a web-safe MP4 overlay of SAM3 basketball detections for direct visual review
  - [ ] Use Layer 1 active-player candidates as SAM3 ROI seeds for segmented active-player review
  - [ ] Render a web-safe MP4 overlay of SAM3 active-player masks for direct visual review
  - [ ] Add a frame-local SAM3 detect endpoint to the labeller for paused-frame promptable segmentation
  - [ ] Add labeller UI controls for prompt entry, detect, and overlay display of same-frame SAM3 results
  - [ ] Use SAM3 basketball detections to bootstrap the first credible runtime ball state before predictive ROI handoff
  - [ ] Re-enter SAM3 ball reacquisition only after sustained missing-ball evidence or segment reset and emit handoff provenance
  - [x] Align image-model SAM checkpoint defaults with the upstream SAM3 image-builder contract and remove partial-load warnings
    - [x] Move experimental SAM and OCR model caches into a repo-local git-ignored mount so Docker runs do not re-download weights
- [ ] Add a rollback-safe experimental Orin Docker image variant on JetPack 6 / CUDA 12.x for official SAM 3 validation
  - [ ] Containerize the full Layer 1 artifact generation path so the existing pipeline can be exercised end to end inside the SAM 3-capable image
  - [ ] Add a bounded GPU validation and rollback workflow for the experimental Orin SAM 3 image without mutating the stable Orin runtime
- [ ] Add optional SAM 3-assisted player recovery on unexplained DINO play-region blobs and ambiguous YOLO ROIs in Layer 1 artifacts
  - [ ] Route official SAM 3 text-prompted ROI refinement through the Layer 1 artifact pipeline with explicit fail-closed runtime statuses
  - [x] Keep Orin setup honest about SAM 3 runtime prerequisites and skip installation when the pinned Python runtime is too old
  - [x] Add regression coverage for SAM ROI proposal generation and recovered-detection wiring without requiring live SAM weights
- [ ] Split the labeller overlay into promoted versus demoted detection states so review can preserve raw evidence without confusing it for trusted on-court players
- [ ] Discount camera-pan-induced apparent motion in Layer 1 player plausibility scoring so seated spectators are not promoted by shared frame drift
- [ ] Add multi-signal on-court player plausibility scoring so spectators and merged detections are down-ranked before active-player promotion
- [ ] Make DINO play-region priors ephemeral and invalidate them on strong camera pan or layout drift
- [ ] Feed the current play-region prior into geometry fitting so court evidence is restricted to likely in-play regions
- [ ] Add bounded multi-hypothesis identity infrastructure for ambiguous short-gap track continuity
  - [x] Formalize the Layer 1 player identity evidence model and hard constraints into a machine-readable policy and emitted artifact contract
  - [x] Implement explicit Layer 1 identity decision staging with hard-filtered candidate records, soft-evidence scoring, and selected-link provenance
  - [x] Sweep reviewed Layer 1 artifacts for nontrivial identity conflict components that actually exercise bounded assignment-hypothesis selection
  - [x] Compare bounded assignment-hypothesis selection against a greedy edge baseline on the first reviewed artifact with a nontrivial conflict component
  - [x] Build bounded global identity hypotheses across ambiguous local conflict groups instead of exposing only isolated group-level alternatives
  - [x] Emit per-track canonical-identity options induced by the top bounded global identity hypotheses for review and downstream consumers
  - [x] Keep runtime stitching conservative while exposing unresolved multi-hypothesis identity state in Layer 1 artifacts and summaries
  - [x] Make the jersey-OCR consensus path consume bounded identity track options instead of only committed identity_track_id assignments
  - [x] Emit per-track jersey option consensus for ambiguous identity tracks without forcing runtime identity collapse
  - [x] Surface hypothesis-aware jersey consensus counts and option summaries in Layer 1 artifacts and tests
  - [x] Re-rank bounded global identity hypotheses using jersey option consensus as later-pass evidence without mutating committed runtime identities
  - [x] Emit conservative jersey-preferred canonical track annotations per detection from the selected later-pass global identity world
  - [x] Surface jersey-aware global identity resolution summaries and tie-break regressions in Layer 1 artifacts and tests
  - [x] Surface base versus jersey-reranked global identity world status in the labeller UI for reviewed Layer 1 artifacts
  - [x] Show per-track preferred canonical identity changes caused by jersey-aware reranking in the labeller track inspector and selectors
  - [x] Add labeller regressions or frontend wiring checks for the new identity-world review surfaces and align plan/docs
- [ ] Extend attributed MVP event emission to shot and rebound evidence once ball-result signals are available
- [ ] Add a minimal ball artifact to Layer 1 review outputs and use it to refine live-play gating
- [ ] Add explicit playback transport controls and conservative jersey-OCR display gating in the labeller
- [ ] Add a rule-based live-play/dead-ball gate with per-frame scores, stitched segments, and labeller review support
- [ ] Add bidirectional short-gap repair for clustered or briefly missed player tracks in Layer 1 artifacts
- [ ] Add a coarse torso-color appearance cue to further suppress seated spectators and sideline bystanders
- [ ] Check in canonical MVP stats header and formula templates with provenance rules
- [ ] Define a HoopSense-native declarative rules layer for stat attribution without reusing Shush implementation
- [ ] Map each MVP stat to required event primitives and possession fields
- [ ] Define `PossessionContext` fields and ledger serialization contract
- [ ] Track `ballhandler_id`, `dribble_count`, and `pass_count` for one possession slice
- [ ] Define the first stat-ready event set: `pass`, `catch`, `dribble`, `shot_attempt`, `rebound`, `turnover`
- [ ] Materialize auditable box-score rows that match the core scorebook columns and formulas
- [ ] Add scorebook-backed validation fixtures that compare generated stats to trusted totals
- [ ] Decide the first migration target based on performance and risk (TrackManager identified)
- [ ] Separate ARM64/Jetson runtime guidance from cloud/x86 Docker guidance
- [ ] Add first slice reports for action class, camera/view, and pose quality

## Product Reality Check

What is currently true:
- the Action Brain is a narrow local-motion classifier
- `features_v2` remains the current frozen neural contract
- the layered feature architecture is documented
- the staged scene-prior, discovery, tracking, retrofit, and geometry-refinement architecture is now documented
- the geometry layer has a shared module and readiness report artifact
- an Orin container logic probe exists
- basic manifest hashing, training lineage scaffolding, and initial CI plumbing now exist locally
- possession and stat primitives exist locally in Rust and Python, but need tighter end-to-end verification
- the MVP product contract is now explicitly captured as a normalized scorebook-style output spec

What is not yet true:
- Oracle scale-out has not yet been re-materialized and verified as an on-disk dataset artifact
- the possession context contract is not yet proven end to end in inference output
- the scorebook-style box-score contract is not yet materialized as headers, formulas, and fixture-backed validation
- dataset promotion rules are not yet defined
- the CI architecture is not yet split by functional core
- no Terraform layer exists yet for shared GCP infrastructure

## Quality Gates

- [ ] Rust core logic verified for PossessionContext and box-score changes
- [ ] Python behavior-engine/perception gate tests verified
- [x] Synthetic Oracle tests passing
- [x] Tooling review tests passing
- [x] Documentation updated for layered feature architecture
- [x] Planning invariant checked into `AGENTS.md`
- [>] Orin hardware validation (CUDA/PyTorch)
- [ ] Scorebook-style stat contract materialized as checked-in templates and validation fixtures
- [ ] Oracle dataset manifest validation implemented
- [ ] Checkpoint lineage recording implemented
- [ ] Slice-based evaluation report implemented
- [x] Geometry readiness report artifact implemented
- [ ] Initial CI workflow implemented
- [x] Check in an explicit Layer 1 identity-policy contract and machine-readable rules file

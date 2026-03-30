# HoopSense Task Status

This file is a human-readable execution snapshot derived from `docs/plan/PLAN_TREE.yaml`.

The old chapter-style status model is retired. This file now tracks the current plan frontier and active L-level work only.

## Current Frontier

The current highest-priority frontier is:
- `L3.43` publish a mobile-friendly external HoopSense demo page with one representative clip, overlay, and feedback prompt
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
- [x] Add a structured perception-feedback workflow for false positives, misses, merges, and track errors in the labeller
- [x] Verify actual CUDA acceleration in the Orin validation path and publish a repeatable probe artifact
- [ ] Scale Oracle ingestion to Subject 124 while preserving `features_v2`
- [ ] Define `PossessionContext` fields and ledger serialization contract
- [ ] Track `ballhandler_id`, `dribble_count`, and `pass_count` for one possession slice
- [ ] Define the first stat-ready event set: `pass`, `catch`, `dribble`, `shot_attempt`, `rebound`, `turnover`
- [ ] Generate MVP box-score rows from attributed events
- [ ] Implement offense zone and transition flag derivation in the possession context
- [ ] Document candidate Python-versus-Rust ownership split for the perception layer
- [ ] Implement first-pass dataset manifests with SHA-256 hashing
- [ ] Integrate training lineage recording (git commit + data hash) into training loop
- [ ] Establish initial CI plumbing (.github/workflows/ci.yml)
- [ ] Implement dynamic perception audit script (scripts/run_perception_audit.sh)

### Active or Next
- [ ] Publish a mobile-friendly external HoopSense demo page with one representative clip, overlay, and feedback prompt
- [ ] Decide the first migration target based on performance and risk (TrackManager identified)
- [ ] Separate ARM64/Jetson runtime guidance from cloud/x86 Docker guidance
- [ ] Add first slice reports for action class, camera/view, and pose quality

## Product Reality Check

What is currently true:
- the Action Brain is a narrow local-motion classifier
- `features_v2` remains the current frozen neural contract
- the layered feature architecture is documented
- the geometry layer has a shared module and readiness report artifact
- an Orin container logic probe exists
- basic manifest hashing, training lineage scaffolding, and initial CI plumbing now exist locally
- possession and stat primitives exist locally in Rust and Python, but need tighter end-to-end verification

What is not yet true:
- Oracle scale-out has not yet been re-materialized and verified as an on-disk dataset artifact
- the possession context contract is not yet proven end to end in inference output
- CUDA acceleration is not yet proven in every runtime path
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
- [ ] Oracle dataset manifest validation implemented
- [ ] Checkpoint lineage recording implemented
- [ ] Slice-based evaluation report implemented
- [x] Geometry readiness report artifact implemented
- [ ] Initial CI workflow implemented

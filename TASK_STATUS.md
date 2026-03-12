# HoopSense Task Status

This file is a human-readable execution snapshot derived from `docs/plan/PLAN_TREE.yaml`.

The old chapter-style status model is retired. This file now tracks the current plan frontier and active L-level work only.

## Current Frontier

The current highest-priority frontier is:
- `L3.34` define measurable readiness checks for ingestion, tracking, pose, geometry, and lifting
- `L3.35` add a perception-quality report artifact for representative clips
- `L3.36` document candidate Python-versus-Rust ownership split for the perception layer
- `L3.37` decide the first migration target, if any, based on performance, determinism, and implementation risk
- `L3.17` align this status file with the plan tree and layered architecture
- `L3.18` define Oracle dataset manifests and validation checks
- `L3.19` record Action Brain training lineage beside checkpoints
- `L3.24` make Docker docs explicitly cloud-oriented and not the native Orin story
- `L3.25` add a checked-in CI workflow for Python, Rust, docs, and contract checks
- `L3.26` publish separate runbooks for cloud/x86 training and Jetson/ARM64 runtime validation
- `L3.29` define functional-core CI ownership and path triggers
- `L3.30` add a lightweight integration pipeline that depends on selected core workflows
- `L3.31` gate training smoke/evaluation workflows behind ML-specific changes or manual triggers
- `L3.32` define which GCP resources stay on direct `gcloud` versus move into Terraform
- `L3.33` add initial Terraform layout for shared buckets, registry, and service accounts
- `L3.4` scale the Oracle from the MVP fixture to Subject 124 while preserving `features_v2`
- `L3.9` define the `PossessionContext` contract
- `L3.10` implement the first ballhandler/dribble/pass tracking slice
- `L3.12` define the first stat-ready event set
- `L3.20` add first slice-based evaluation outputs
- `L3.27` define dataset version lineage and promotion rules across Oracle/synthetic training sets
- `L3.28` define multi-arch or explicitly split cloud-vs-Orin container strategy

## L0 Product Goal

- [>] Trustworthy basketball understanding from video

## L1 Workstreams

- [>] Perception and geometry foundation
- [>] Action Brain and synthetic Oracle training loop
- [ ] Possession context and game-state reasoning
- [ ] Event attribution and stats generation
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
- [>] Perception-and-geometry readiness gate before Action Brain expansion
- [ ] Python-versus-Rust boundary decision for perception and geometry

### Action Brain and Synthetic Oracle
- [x] Stable Action Brain feature contract (`features_v2`)
- [>] Oracle MoCap ingestion, FK, and dataset generation
- [>] Action Brain training, evaluation, and checkpoint lifecycle
- [ ] Sim-to-real fine-tuning and validation loop

### Possession Context and Game-State Reasoning
- [ ] Possession context contract in the ledger
- [ ] Ball control, dribble count, and pass-chain tracking
- [ ] Offense zone, transition, and drive semantics
- [>] Referee-assisted rewind and correction logic

### Event Attribution and Stats Generation
- [ ] Event attribution rules combining motion, ball, and possession state
- [ ] Stat ledger and box-score generation
- [ ] Shot chart, defensive metrics, and report outputs

### Deployment, Runtime, and Operator Workflow
- [>] Native Orin environment and ARM64 runtime validation
- [>] Cloud/x86 training environment and artifact parity
- [ ] Real-time inference pipeline and model optimization

### Collaboration, Review, and Project Control
- [x] Gemini/Codex collaboration bridge and review workflow
- [>] Plan-driven execution and task synchronization
- [>] Documentation and task-status alignment

### MLOps Governance and Model Lifecycle Control
- [ ] Dataset manifests, lineage, and validation policy
- [ ] Training run lineage and checkpoint lifecycle
- [ ] Slice-based evaluation and drift monitoring
- [ ] Deployment compatibility reporting across cloud and edge targets

### DevOps Reproducibility and Delivery Control
- [>] Guix-first development and environment reproducibility
- [>] Docker fallback packaging and cloud image discipline
- [ ] CI quality gates for code, contracts, and docs
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

### Active or Next
- [ ] Document candidate Python-versus-Rust ownership split for ingestion, tracking, pose, geometry, and lifting
- [ ] Decide the first migration target, if any, based on performance, determinism, and implementation risk
- [ ] Bring `TASK_STATUS.md` into alignment with the layered feature plan
- [ ] Define Oracle dataset manifest fields and validation checks
- [ ] Record Action Brain training lineage beside each checkpoint
- [ ] Make Docker docs explicitly cloud-oriented and not the native Orin story
- [ ] Add a checked-in CI workflow for Python, Rust, docs, and contract checks
- [ ] Publish separate runbooks for cloud/x86 training and Jetson/ARM64 runtime validation
- [ ] Define functional-core CI ownership and path triggers
- [ ] Add a lightweight integration pipeline that depends on selected core workflows
- [ ] Gate training smoke/evaluation workflows behind ML-specific changes or manual triggers
- [ ] Define which GCP resources stay on direct `gcloud` versus move into Terraform
- [ ] Add initial Terraform layout for shared buckets, registry, and service accounts
- [ ] Scale Oracle ingestion to Subject 124 while preserving `features_v2`
- [ ] Define `PossessionContext` fields and ledger serialization contract
- [ ] Track `ballhandler_id`, `dribble_count`, and `pass_count` for one possession slice
- [ ] Define the first stat-ready event set: `pass`, `catch`, `dribble`, `shot_attempt`, `rebound`, `turnover`
- [ ] Separate ARM64/Jetson runtime guidance from cloud/x86 Docker guidance
- [ ] Add first slice reports for action class, camera/view, and pose quality

## Product Reality Check

What is currently true:
- the Action Brain is a narrow local-motion classifier, not the full game-reasoning system
- `features_v2` remains the current frozen neural contract
- the Oracle MVP is restored and usable for one fixture motion
- a GPU training path has produced an initial checkpoint
- the layered feature architecture is now documented
- MLOps is now a first-class product concern in the requirements, architecture, and plan
- DevOps is now a first-class product concern in the requirements, architecture, and plan
- the geometry layer now has a shared pure-Python module and a readiness-report artifact

What is not yet true:
- perception and geometry are not yet explicitly gated as training-readiness inputs
- the Python-versus-Rust boundary for this layer is not yet decided
- possession context is not yet implemented as a durable ledger contract
- event attribution and stats generation are not yet complete
- dataset manifests and checkpoint lineage are not yet implemented
- a checked-in CI workflow does not yet exist
- the CI architecture is not yet split by functional core
- no Terraform layer exists yet for shared GCP infrastructure
- deployment guidance is not yet fully separated between Jetson/ARM64 and cloud/x86
- `TASK_STATUS.md` was previously stale and is now being reset around the plan tree

## Quality Gates

- [x] Rust core logic tests passing
- [x] Python behavior-engine tests passing
- [x] Synthetic Oracle tests passing
- [x] Tooling review tests passing
- [x] Documentation updated for layered feature architecture
- [x] Planning invariant checked into `AGENTS.md`
- [ ] Oracle dataset manifest validation implemented
- [ ] Checkpoint lineage recording implemented
- [ ] Slice-based evaluation report implemented
- [x] Geometry readiness report artifact implemented
- [ ] CI workflow implemented
- [ ] Functional-core CI split implemented
- [ ] Conditional training workflow implemented
- [ ] Terraform baseline implemented for stable shared GCP resources
- [ ] Cloud/x86 and Jetson/ARM64 runbooks published

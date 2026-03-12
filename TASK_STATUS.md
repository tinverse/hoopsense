# HoopSense Task Status

This file is a human-readable execution snapshot derived from `docs/plan/PLAN_TREE.yaml`.

## Current Frontier

The current highest-priority frontier is:
- align this status file with the plan tree and layered architecture
- define Oracle dataset manifests and validation checks
- record Action Brain training lineage beside checkpoints
- make Docker docs explicitly cloud-oriented and not the native Orin story
- add a checked-in CI workflow for Python, Rust, docs, and contract checks
- publish separate runbooks for cloud/x86 training and Jetson/ARM64 runtime validation
- scale the Oracle from the MVP fixture to Subject 124 while preserving `features_v2`
- define the `PossessionContext` contract and the first ballhandler/dribble/pass tracking slice
- define the first stat-ready event set
- separate cloud/x86 and Jetson/ARM64 deployment guidance
- add first slice-based evaluation outputs

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

## Key L3 Execution Slices

### Completed
- [x] Preserve the Oracle MVP parser/FK path for one fixture motion
- [x] Maintain a reproducible GPU training smoke test and checkpoint write path
- [x] Require plan-tree updates before substantial architecture or implementation changes

### Active or Next
- [ ] Bring `TASK_STATUS.md` into alignment with the layered feature plan
- [ ] Define Oracle dataset manifest fields and validation checks
- [ ] Record Action Brain training lineage beside each checkpoint
- [ ] Make Docker docs explicitly cloud-oriented and not the native Orin story
- [ ] Add a checked-in CI workflow for Python, Rust, docs, and contract checks
- [ ] Publish separate runbooks for cloud/x86 training and Jetson/ARM64 runtime validation
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

What is not yet true:
- possession context is not yet implemented as a durable ledger contract
- event attribution and stats generation are not yet complete
- dataset manifests and checkpoint lineage are not yet implemented
- a checked-in CI workflow does not yet exist
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
- [ ] CI workflow implemented
- [ ] Cloud/x86 and Jetson/ARM64 runbooks published

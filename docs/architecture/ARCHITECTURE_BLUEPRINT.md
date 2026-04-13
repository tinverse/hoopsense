# HoopSense Architecture Blueprint

HoopSense uses a **Core-and-Satellite** architecture to separate performance-critical vision logic from high-level application orchestration.

The product architecture is also governed by layered MLOps and DevOps strategies. Data artifacts, feature contracts, evaluation reports, model promotion records, environment definitions, and runtime compatibility records are first-class components of the system, not implementation byproducts.

## 1. The Prototyping Intelligence Stack (Python-First)

Currently, HoopSense utilizes a Python-based perception pipeline for rapid iteration and model validation. The long-term goal is to migrate performance-critical perception to Rust.

Near-term priority:
- solidify the Perception and Geometry layer as a trusted input substrate before expanding Action Brain work
- explicitly evaluate the Python-vs-Rust boundary for ingestion, tracking continuity, geometry, and lifting

### Layer 1: The Perceiver (Python Prototype)
- **Tech:** staged scene priors, promptable discovery, and temporal tracking
- **Responsibility:** builds a revisable scene state from imperfect basketball video rather than treating each frame as final truth
- **Primary stages:**
  - deterministic video contract
  - scene prior (`DINO` / play-region evidence)
  - object discovery (`SAM3`, detector proposals, promptable review probes)
  - temporal tracking for players and ball
  - bounded retrospective repair when later evidence clarifies earlier ambiguity
  - opportunistic geometry refinement as court anchors become visible
- **Output:** HoopSense Perception artifacts for raw observations plus tracked and repair-aware scene state

Reference:
- `docs/architecture/components/SCENE_DISCOVERY_TRACKING.md`

### Layer 2: Identity & Logic Heuristics (Python Prototype)
- **Tech:** EasyOCR + HSV Clustering + Heuristic Rules.
- **Responsibility:** Maps track IDs to jersey numbers; identifies high-level actions (Jump Shot) and referee signals via kinematic heuristics.

### Layer 3: Action Brain (Python Training / Inference Runtime)
- **Tech:** Temporal Transformer over `features_v2`.
- **Responsibility:** Performs narrow local-motion classification from pose, velocity, ball-distance, and court-position features.
- **Boundary:** Does not directly own possession state, event attribution, or box-score stats.

### Layer 4: The Geometric Muscle (Rust Core)
- **Implementation:** **Rust (hoopsense-core)**
- **Responsibility:** High-performance spatial math. Handles Lens Undistortion, Homography (DLT), Camera Pose (PnP), and Dynamic SLAM-lite state tracking.
- **Value:** This is the deterministic "measurement" layer that ensures 3D stat accuracy.

Important boundary:
- geometry should improve Layer 1 as confidence grows
- HoopSense should not assume full trusted calibration before useful perception begins

### Perception-and-Geometry Readiness Gate

Before substantial Action Brain expansion, HoopSense should verify:
- video ingestion stability: deterministic frame/timestamp handling
- detection/tracking quality: stable player and ball tracks with measured ID-switch/dropout behavior
- pose quality: acceptable keypoint completeness and confidence on representative basketball motion
- court geometry quality: sane homography/projection behavior on known court anchors
- lifting quality: consistent `X,Y,Z` output and biomechanical sanity

This gate exists because the Action Brain only sees the representation produced by these layers.

### Python vs Rust Boundary

The repo should explicitly evaluate which parts of this layer belong in Python versus Rust.

Python is currently favored for:
- rapid model iteration
- detector/tracker integration
- experimentation-heavy pose workflows

Rust is currently favored for:
- deterministic geometry
- low-level numeric kernels
- contract-sensitive ledger/math components
- performance-critical re-entry and validation paths

The decision should be made subsystem by subsystem, not as an all-or-nothing rewrite.

### Layer 5: The Game State Ledger (Rust Core)
- **Responsibility:** Consolidates possession context and events into an official, retroactive ledger with temporal reconciliation.
- **Output:** HoopSense Ledger JSONL for validated events + Player Statistics.

### Layer 6: Physics & Trajectory (Rust Core)
- **Tech:** Parabolic Least Squares Fitting.
- **Responsibility:** Predicts ball flight paths; identifies shot apex (arc); and detects 3D rim intersections for deterministic scoring.
- **Value:** Smoothes noisy visual detections into physically valid movements.

### Layer 7: Acoustic Auditing (Audio Head)
- **Tech:** Keyword Spotting (CNN/Whisper).
- **Responsibility:** Identifies localized vocal cues ("Sub!", "Time-out!") to provide temporal anchors for state changes.
- **Value:** Ignores non-localized environmental noise like distant whistles.

## 2. Orchestration (The Bridge)

### Spatial Processor (core/src/bin/spatial_processor.rs)
- **Tech:** Rust Binary.
- **Responsibility:** Ingests the Python-generated JSONL stream; applies the Spatial Resolver (Chapter 3) to every row; validates actions against the Geometric Referee (Rules Engine); and writes the final validated Game DNA.
- **Value:** This is the high-performance bridge that turns "Visual Guesses" into "Geometric Truths."

## 3. The Development Lifecycle (Quality Gates)

### Agent-Based Review (tools/review)
- **Tech:** Gemini Generalist Sub-Agent.
- **Responsibility:** Audits every feature implementation for compliance with project mandates (Teach-First, Design-First), logic bugs, and NCAA consistency.
- **Mechanism:** A pre-commit/manual trigger that generates a comprehensive review prompt based on staged git diffs.

### MLOps Control Plane
- **Responsibility:** Governs dataset manifests, feature-contract validation, training run lineage, evaluation reports, promotion state, and deployment compatibility.
- **Key Rule:** No dataset or checkpoint should be treated as production-ready without explicit validation evidence.
- **Scope:** Operates inside the broader DevOps layer and covers both cloud/x86 training and Jetson/ARM64 deployment targets.

### DevOps Control Plane
- **Responsibility:** Governs developer environments, CI quality gates, build packaging, cloud job execution, and target-specific runtime guidance.
- **Key Rule:** Prefer explicit target-specific environments: stable native Orin scripts for Jetson validation, dedicated Docker images for cloud and experimental JetPack 6 or SAM3 work, and repo-managed shell definitions only as optional developer conveniences.
- **Scope:** Covers local development, stable Orin smoke tests and early training, experimental containerized Orin validation, then cloud training images and deployment paths.

## 4. The Capture Ecosystem (Satellites)

### Mobile Lens (Thin Client)
- **Environment:** Android/iOS.
- **Feature:** IMU sensor fusion (gyro/accel) to assist the Spatial Resolver. Uses "Virtual Tripod" UI for guided capture.

### HoopBox (Edge AI)
- **Hardware:** NVIDIA Jetson Orin Nano.
- **Feature:** Standardized high-angle fixed mount for "Always-On" gym monitoring.

### DSLR Firmware (The Glass Play)
- **Tech:** Magic Lantern fork.
- **Feature:** High-bitrate capture and focal-length telemetry for registered photographers.

## 5. Cloud Deployment Strategy (Scaling)

### Containerization And Runtime Boundaries
- **Method:** Use dedicated Docker images for cloud training and for experimental JetPack 6 or SAM3 validation paths; keep the stable native Orin scripts as the baseline Jetson runtime path.
- **Goal:** Make runtime boundaries explicit instead of pretending one environment model covers native Jetson, cloud, and experimental GPU validation equally well.
- **Value:** Reduces environment drift while acknowledging that Jetson-native validation and experimental containerized validation are separate operational modes.

In current repo reality, the stable native Orin path is the baseline for Jetson smoke tests and initial runs, while Docker is used for cloud workloads and experimental JetPack 6 or SAM3 validation.

### Infrastructure (Google Cloud Platform)
- **Vertex AI (Colab Enterprise):** Used for interactive model development and fine-tuning with custom Docker images.
- **Compute Engine (GCE) + GPU:** Headless batch processing of large video datasets using the HoopSense Docker image.
- **Cloud Storage:** Standardized landing zone for raw video ingestion and HoopSense JSONL artifacts.

## 6. The Data Flow (HoopSense JSONL Contracts)
HoopSense uses staged JSONL contracts for machine-readable interchange:
1. **HoopSense Perception JSONL**
   Raw and staged perception output from scene priors, discovery, tracking, retrofit, pose, and geometry evidence.
2. **HoopSense Ledger JSONL**
   Ledger-validated events and possession-aware outputs used for downstream stats and reporting.

This split avoids conflating raw observations with validated basketball meaning.

## 7. The ML Artifact Flow

HoopSense treats the ML lifecycle as an explicit artifact graph:
1. source data and Oracle assets
2. validated datasets and manifests
3. feature-contract validation
4. training configs and runs
5. candidate checkpoints
6. evaluation and slice reports
7. promotion records
8. deployment-target compatibility reports

This ML artifact flow is part of the product architecture because bad lineage or silent contract drift can invalidate downstream basketball reasoning.

## 8. The Build and Delivery Flow

HoopSense treats build and delivery as an explicit control path:
1. Stable native scripts define the supported Jetson runtime path
2. CI validates code, contracts, and documentation
3. Docker images package cloud training and experimental runtime boundaries
4. Vertex/GCP job specs execute cloud workloads
5. Runtime guidance documents vendor, Python, and CUDA boundary assumptions explicitly

This build-and-delivery flow is part of the product architecture because deployment ambiguity can invalidate reproducibility claims.

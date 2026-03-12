# HoopSense Architecture Blueprint

HoopSense uses a **Core-and-Satellite** architecture to separate performance-critical vision logic from high-level application orchestration.

The product architecture is also governed by layered MLOps and DevOps strategies. Data artifacts, feature contracts, evaluation reports, model promotion records, environment definitions, and runtime compatibility records are first-class components of the system, not implementation byproducts.

## 1. The Prototyping Intelligence Stack (Python-First)

Currently, HoopSense utilizes a Python-based perception pipeline for rapid iteration and model validation. The long-term goal is to migrate performance-critical perception to Rust.

### Layer 1: The Perceiver (Python Prototype)
- **Tech:** YOLOv8-pose + BoT-SORT.
- **Responsibility:** Ingests raw video frames; identifies skeletons for players, the ball, and referees.
- **Output:** `Shush-P` JSONL stream for raw perception observations.

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

### Layer 5: The Game State Ledger (Rust Core)
- **Responsibility:** Consolidates possession context and events into an official, retroactive ledger with temporal reconciliation.
- **Output:** `Shush-L` JSONL stream for validated ledger events + Player Statistics.

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
- **Scope:** Covers both cloud/x86 training and Jetson/ARM64 deployment targets.

### DevOps Control Plane
- **Responsibility:** Governs reproducible developer environments, CI quality gates, build packaging, cloud job execution, and target-specific runtime guidance.
- **Key Rule:** Prefer Guix-managed reproducibility where possible; use Docker as the fallback runtime and packaging boundary.
- **Scope:** Covers local development, cloud training images, and Jetson/Orin deployment paths.

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

### Containerization (The Guix-Docker Bridge)
- **Method:** `guix pack -f docker`
- **Goal:** Create a deterministic, bit-identical image containing the Rust core, Python inference engine, and all CUDA/system dependencies.
- **Value:** Eliminates "it works on my machine" bugs when moving from local development to the cloud.

In current repo reality, Docker is the explicit cloud-training path, while Guix remains the preferred local reproducibility mechanism. Jetson/Orin requires an additional documented vendor-library bridge.

### Infrastructure (Google Cloud Platform)
- **Vertex AI (Colab Enterprise):** Used for interactive model development and fine-tuning with custom Docker images.
- **Compute Engine (GCE) + GPU:** Headless batch processing of large video datasets using the HoopSense Docker image.
- **Cloud Storage:** Standardized landing zone for raw video ingestion and JSONL "Shush" outputs.

## 6. The Data Flow (The "Shush" Contract)
HoopSense uses the name "Shush" for JSONL-based machine-readable interchange, but there are distinct contract stages:
1. `Shush-P`
   Raw perception output from detectors, trackers, pose, and referee sensing.
2. `Shush-L`
   Ledger-validated events and possession-aware outputs used for downstream stats and reporting.

This naming split avoids conflating raw observations with validated basketball meaning.

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
1. Guix manifests define reproducible development inputs where possible
2. CI validates code, contracts, and documentation
3. Docker images package cloud training or constrained runtime boundaries
4. Vertex/GCP job specs execute cloud workloads
5. Jetson/Orin runtime guidance documents the vendor boundary and validation steps

This build-and-delivery flow is part of the product architecture because deployment ambiguity can invalidate reproducibility claims.

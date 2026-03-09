# HoopSense Architecture Blueprint

HoopSense uses a **Core-and-Satellite** architecture to separate performance-critical vision logic from high-level application orchestration.

## 1. The Prototyping Intelligence Stack (Python-First)

Currently, HoopSense utilizes a Python-based perception pipeline for rapid iteration and model validation. The long-term goal is to migrate performance-critical perception to Rust.

### Layer 1: The Perceiver (Python Prototype)
- **Tech:** YOLOv8-pose + BoT-SORT.
- **Responsibility:** Ingests raw video frames; identifies skeletons for players, the ball, and referees.
- **Output:** Shush Contract JSONL stream.

### Layer 2: Identity & Logic Heuristics (Python Prototype)
- **Tech:** EasyOCR + HSV Clustering + Heuristic Rules.
- **Responsibility:** Maps track IDs to jersey numbers; identifies high-level actions (Jump Shot) and referee signals via kinematic heuristics.

### Layer 3: The Geometric Muscle (Rust Core)
- **Implementation:** **Rust (hoopsense-core)**
- **Responsibility:** High-performance spatial math. Handles Lens Undistortion, Homography (DLT), Camera Pose (PnP), and Dynamic SLAM-lite state tracking.
- **Value:** This is the deterministic "measurement" layer that ensures 3D stat accuracy.

### Layer 4: The Game State Ledger (Rust Core)
- **Responsibility:** Consolidates events into an official, retroactive ledger with temporal reconciliation.
- **Output:** Shush Contract JSONL stream + Player Statistics.

### Layer 5: Physics & Trajectory (Rust Core)
- **Tech:** Parabolic Least Squares Fitting.
- **Responsibility:** Predicts ball flight paths; identifies shot apex (arc); and detects 3D rim intersections for deterministic scoring.
- **Value:** Smoothes noisy visual detections into physically valid movements.

### Layer 6: Acoustic Auditing (Audio Head)
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

## 2. The Capture Ecosystem (Satellites)

### Mobile Lens (Thin Client)
- **Environment:** Android/iOS.
- **Feature:** IMU sensor fusion (gyro/accel) to assist the Spatial Resolver. Uses "Virtual Tripod" UI for guided capture.

### HoopBox (Edge AI)
- **Hardware:** NVIDIA Jetson Orin Nano.
- **Feature:** Standardized high-angle fixed mount for "Always-On" gym monitoring.

### DSLR Firmware (The Glass Play)
- **Tech:** Magic Lantern fork.
- **Feature:** High-bitrate capture and focal-length telemetry for registered photographers.

## 3. Cloud Deployment Strategy (Scaling)

### Containerization (The Guix-Docker Bridge)
- **Method:** `guix pack -f docker`
- **Goal:** Create a deterministic, bit-identical image containing the Rust core, Python inference engine, and all CUDA/system dependencies.
- **Value:** Eliminates "it works on my machine" bugs when moving from local development to the cloud.

### Infrastructure (Google Cloud Platform)
- **Vertex AI (Colab Enterprise):** Used for interactive model development and fine-tuning with custom Docker images.
- **Compute Engine (GCE) + GPU:** Headless batch processing of large video datasets using the HoopSense Docker image.
- **Cloud Storage:** Standardized landing zone for raw video ingestion and JSONL "Shush" outputs.

## 4. The Data Flow (The "Shush" Contract)
All layers communicate via a standardized JSONL format, ensuring the "Brain" (Rust) can process data regardless of the "Eye" (Mobile/HoopBox/DSLR).

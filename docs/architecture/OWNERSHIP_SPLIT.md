# Perception Ownership Split (Python vs. Rust)

This document defines the architectural boundary between Python-based prototyping and Rust-based performance/determinism for the HoopSense perception layer.

## Current State (Stage 1: Python-Heavy)
- **Ingestion:** Python (OpenCV)
- **Detection & Tracking:** Python (Ultralytics BoT-SORT)
- **Pose Estimation:** Python (Ultralytics YOLOv8-pose)
- **Spatial Resolution:** Rust (SpatialResolver via Bridge)
- **Behavior Engine:** Python (PossessionEngine + DSL Evaluator)
- **Ledger:** Rust (GameStateLedger)

## Proposed Target State (Stage 2: Deterministic Core)

### 1. The Python Frontier (The "Eye")
**Responsibility:** Rapid experimentation with SOTA ML models.
- **ML Inference:** Detection, Pose, and OCR models should remain in Python to leverage optimized PyTorch/TensorRT ecosystems.
- **Audio Analysis:** Python (AudioHead) for referee whistle detection.
- **Initial JSONL Production:** Emitting raw "Shush-P" streams.

### 2. The Rust Interior (The "Brain")
**Responsibility:** Performance-critical processing, temporal consistency, and geometric grounding.
- **Track Management (Move from Python):** Implementation of the `TrackManager` and Kalman Filters should migrate to Rust. This ensures that track-state is consistent between real-time inference and offline re-processing.
- **Kinematic Lifting (Move from Python):** The `lift_keypoints_to_3d` logic should be entirely in Rust to ensure bone-length constraints are enforced at the type-system level.
- **Possession Logic (Move from Python):** The `PossessionEngine` should migrate to Rust. Global ball-player proximity is a geometric query that belongs in the `GeometricReferee`.

## Migration Rationale

| Component | Target | Rationale |
| :--- | :--- | :--- |
| **TrackManager** | Rust | High-frequency update loop; memory safety for large track sets. |
| **Kalman Filter** | Rust | Deterministic floating-point math; reusable by multiple agents. |
| **PossessionEngine**| Rust | Transitions system from perception to rules; must be bug-free. |
| **Model Inference** | Python | Fragility of ML library bindings in Rust; Python ecosystem is superior. |

## First Migration Target: TrackManager
The `TrackManager` is the ideal first candidate for migration (L3.37). By moving tracking state to Rust, we can:
1.  Remove the `deque` history management in Python.
2.  Perform multi-track spatial reasoning without crossing the bridge for every player.
3.  Ensure that track persistence logic is identical across ARM64 and x86 targets.

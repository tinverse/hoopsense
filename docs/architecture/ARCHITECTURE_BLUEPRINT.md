# HoopSense Architecture Blueprint

HoopSense uses a **Core-and-Satellite** architecture to separate performance-critical vision logic from high-level application orchestration.

## 1. The Hierarchical Intelligence Stack

### Layer 1: The Perceiver (Detection & Tracking)
- **Tech:** YOLOv11 + ByteTrack/BoT-SORT.
- **Responsibility:** Ingests raw video frames and identifies bounding boxes for 10 players, the ball, and the hoops.
- **Output:** Temporary `track_id` with `(x, y)` pixel coordinates.

### Layer 2: The Identity Fusion Layer (Recognition)
- **Tech:** PaddleOCR/EasyOCR + Re-ID (ResNet50).
- **Responsibility:** Maps fragile `track_id`s to persistent `player_id`s via jersey OCR and team color clustering.
- **Method:** Track-level majority voting over a 30-frame buffer.

### Layer 3: The Spatial Resolver (Geometric Math)
- **Implementation:** **Rust (HoopSense-core::vision)**
- **Responsibility:** Transforms 2D pixel coordinates into 3D court coordinates (X, Y, Z in cm) using Projective Geometry.
- **Features:** PnP (Perspective-n-Point) solver for camera pose, dynamic Homography for panning cameras, and lens undistortion.

### Layer 4: The Referee API (Auditor)
- **Tech:** Temporal-CNN.
- **Responsibility:** Decodes official hand signals (3-pt, fouls, score) to validate AI-inferred statistics.
- **Logic:** 2-second "Short-Term Memory" buffer for event reconciliation.

### Layer 5: Action Recognition (Behavior)
- **Tech:** Temporal-Transformer (32-frame window).
- **Responsibility:** Classifies player movement into high-level basketball events (Shot, Crossover, Step-back) and decodes referee hand signals.
- **Logic:** Uses Self-Attention on skeletal keypoint velocity vectors to identify semantic "events" from raw movement.

### Layer 6: The Game State Ledger (Logic Engine)
- **Responsibility:** Manages the official "Game DNA." Implements retroactive event-rewind for fouls and possession logic.
- **Output:** Shush Contract JSONL stream + Player Statistics.

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

## 3. The Data Flow (The "Shush" Contract)
All layers communicate via a standardized JSONL format, ensuring the "Brain" (Rust) can process data regardless of the "Eye" (Mobile/HoopBox/DSLR).

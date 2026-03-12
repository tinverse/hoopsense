# Action Brain: Feature Schema V2 (Frozen)

This document defines the exact ordering and normalization of the 72-dimensional Multimodal Feature Tensor.

For the broader multi-layer feature system used by possession reasoning and stats generation, see `LAYERED_FEATURE_SCHEMA.md`.

## Schema Version: 2.0.0

| Index Range | Feature Group | Sub-Features | Normalization |
| :--- | :--- | :--- | :--- |
| **0 - 33** | **Local Pose** | 17 joints (x, y) | Box-relative (0.0 to 1.0) |
| **34 - 67** | **Temporal** | 17 joints (Δx, Δy) | Pixels per frame (scaled by 0.1) |
| **68 - 69** | **Interaction** | Ball-to-Wrist distance (L, R) | Centimeters (scaled by 0.01) |
| **70 - 71** | **Global** | Court Position (x, y) | Court-relative (cm scaled by 0.001) |

### Feature Ordering (Joints 0-16):
0: nose, 1: L_eye, 2: R_eye, 3: L_ear, 4: R_ear, 5: L_sho, 6: R_sho, 7: L_elb, 8: R_elb, 9: L_wri, 10: R_wri, 11: L_hip, 12: R_hip, 13: L_kne, 14: R_kne, 15: L_ank, 16: R_ank.

## Rejection Policy
Training and inference pipelines MUST reject any input where `D != 72`.

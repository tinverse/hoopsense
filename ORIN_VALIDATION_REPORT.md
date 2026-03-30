# Orin Validation Report

- Generated: `2026-03-30T06:09:01.720916+00:00`
- Overall Status: `PASS`

## Environment
- Python: `/data/projects/hoopsense/.venv_orin310/bin/python`
- Platform: `Linux-5.15.185-tegra-aarch64-with-glibc2.35`
- Machine: `aarch64`
- PyTorch: `2.10.0`
- PyTorch Module: `/data/projects/hoopsense/.venv_orin310/lib/python3.10/site-packages/torch/__init__.py`
- CUDA Available: `True`
- CUDA Device: `Orin`
- Probe Device Type: `cuda`
- ActionBrain Device: `cuda`
- ActionBrain Latency (ms): `5.331652006134391`

## Checks
- `torch`: `PASS` (required)  PyTorch imports successfully in the intended runtime environment.
- `numpy`: `PASS` (supplemental)  NumPy C-extensions import cleanly inside the validation environment.
- `cuda_available`: `PASS` (required)  PyTorch sees a CUDA-capable device from the current runtime path.
- `cuda_tensor`: `PASS` (required)  A test tensor can be allocated onto the CUDA device, not just detected.
- `action_brain`: `PASS` (required)  The repo's ActionBrain model can load, move to the target device, and run a forward pass.
- `pose_association`: `SKIP` (supplemental)  The lightweight player-pose matching helper behaves correctly on a simple multi-player fixture.
- `dsl_rule`: `SKIP` (supplemental)  The behavior-engine rule path still recognizes a basic jump-shot style sequence.
- `rust_bridge`: `PASS` (supplemental)  The Rust spatial processor binary can still be invoked from the validation environment.

## Notes
- Rust bridge stderr: `warning: unused import: `NCAA_RIM_HEIGHT`
 --> src/physics.rs:1:20
  |
1 | use crate::rules::{NCAA_RIM_HEIGHT, NCAA_RIM_RADIUS};
  |                    ^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: unused import: `Vector3`
 --> src/physics.rs:2:24
  |
2 | use nalgebra::{Point3, Vector3};
  |                        ^^^^^^^

warning: unused variable: `u`
   --> src/spatial.rs:198:18
    |
198 |             for (u, v) in kpts_2d {
    |                  ^ help: if this is intentional, prefix it with an underscore: `_u`
    |
    = note: `#[warn(unused_variables)]` (part of `#[warn(unused)]`) on by default

warning: unused import: `Point2`
 --> src/bin/spatial_processor.rs:4:16
  |
4 | use nalgebra::{Point2, Point3};
  |                ^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default`

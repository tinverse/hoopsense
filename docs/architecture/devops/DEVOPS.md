# HoopSense DevOps

This document defines the practical operations strategy for HoopSense.

Current execution order:
1. Make the native Jetson Orin path reliable for smoke tests and initial training runs.
2. Use that path to validate training behavior, artifacts, and runtime assumptions.
3. Move heavier or scaled training to cloud/x86 only after the Orin path is stable.

## Scope

DevOps in HoopSense covers:
- reproducible development environments
- build and packaging flows
- CI quality gates
- cloud and edge runtime guidance
- infrastructure ownership

MLOps is part of the operations layer and is documented separately in [MLOPS.md](/data/projects/hoopsense/docs/architecture/devops/MLOPS.md).

## Current Repo Reality

What is implemented:
- `scripts/setup_orin.sh` for the stable native Orin Python 3.10 path
- `scripts/run_layer1_labeller.sh` and `scripts/generate_layer1_annotations.sh` for the stable GPU-backed Layer 1 review path
- `Dockerfile` for cloud-oriented packaging
- `Dockerfile.orin.sam3` and companion scripts for the experimental JetPack 6 / Python 3.12 SAM3 container path
- `tools/infra/cloud_train_wrapper.sh` and `vertex_job.yaml` for cloud training entrypoints
- cloud `Dockerfile` now makes Grounding DINO and related Hugging Face bootstrap dependencies available for later play-region grounding experiments
- `guix.scm` and `guix_orin.scm` remain checked in as optional legacy developer shells and are not the primary operating path

What is not implemented yet:
- checked-in CI workflows
- checked-in Terraform configuration
- a unified multi-arch container story
- fully separated cloud and Jetson runbooks

The Docker path is cloud-oriented. It is not the native Orin story.

## Operating Modes

### Mode A: Local Development

Preferred path:

```bash
python3 -m unittest
python3 -m py_compile tools/review/labeller/generate_layer1_annotations.py
```

Use this for:
- normal development
- docs and plan updates
- unit tests
- non-device-specific validation

Notes:
- do not assume Guix is the default developer entrypoint
- use repo-managed shells only if they materially help a local workflow

### Mode B: Jetson Orin Smoke Tests and Early Training

This is the current first-class target.

Preferred path:

```bash
./scripts/setup_orin.sh
python3 tests/validate_orin_env.py
./scripts/run_orin_cuda_probe.sh
python3 tools/training/train_action_brain.py --smoke-test
```

Use this for:
- GPU smoke validation
- repeatable CUDA/runtime probe artifacts
- first training runs
- confirming model checkpoints progress correctly on target hardware
- catching Jetson-specific ABI or driver issues before cloud spend

Constraint:
- Jetson depends on host-installed NVIDIA libraries and JetPack components
- this path is reproducible only up to that vendor boundary
- `scripts/setup_orin.sh` creates the validated Python 3.10 `.venv_orin310`
- that venv uses the Jetson `jp6/cu126` torch wheel and `nvidia-cudss-cu12`
- the same venv carries the stable Layer 1 review stack (`ultralytics`, `flask`, `lap`, `torchvision`) without replacing the Jetson torch build
- Jetson system OpenCV is linked into the venv instead of installing a conflicting pip wheel
- `scripts/run_orin_cuda_probe.sh` intentionally exposes only `libcudss` from the venv and uses JetPack system CUDA/cuDNN libs for the rest

Stable versus experimental Orin split:
- stable native path: JetPack-backed Python 3.10 via `.venv_orin310`
- experimental SAM3 path: JetPack 6 / L4T R36.x containerized Python 3.12 via `Dockerfile.orin.sam3`
- do not treat these as interchangeable environments

Layer 1 review path:

```bash
./scripts/run_layer1_labeller.sh
./scripts/generate_layer1_annotations.sh data/raw_clips/youth/DSC_4951_sample_0.mp4
```

Reproducible preset-driven review generation:

```bash
./scripts/generate_layer1_review_preset.sh \
  data/raw_clips/youth/DSC_4951_sample_0.mp4 \
  --preset grounded_sam3_review
```

Use `--dry-run` first when checking the exact command and manifest without running inference:

```bash
./scripts/generate_layer1_review_preset.sh \
  data/raw_clips/youth/DSC_4951_sample_0.mp4 \
  --preset grounded_sam3_review \
  --dry-run
```

Notes:
- presets live in `specs/layer1_review_presets.yaml`
- each preset run writes a run manifest under `tmp_runs/layer1_review_runs/`
- `grounded_sam3_review` defaults to the Docker runner because it depends on the experimental Grounding DINO plus SAM3 stack
- keep `scripts/generate_layer1_annotations.sh` as the stable low-friction native path, but use the preset runner for artifact comparisons and review sets

Optional Grounding DINO bootstrap pre-pass for Layer 1 artifacts:

```bash
./scripts/generate_layer1_annotations.sh \
  data/raw_clips/youth/DSC_4951_sample_0.mp4 \
  --bootstrap-foreground-backend grounding_dino \
  --bootstrap-foreground-model IDEA-Research/grounding-dino-tiny \
  --bootstrap-foreground-prompt "basketball court. basketball player. basketball hoop. basketball referee."
```

Notes:
- this bootstrap path is optional and artifact-oriented
- it runs ahead of geometry/perception and emits a coarse foreground/background summary
- it requires a CUDA-capable environment with Grounding DINO support available
- the artifact now carries a segment-oriented `bootstrap_context` handoff so continuity segments can re-bootstrap after detected discontinuities

### Mode C: Cloud Training

Cloud is the scale-out path after Orin is green.

Preferred path:

```bash
docker build -t hoopsense-cloud:latest .
docker run --rm -it hoopsense-cloud:latest python3 tools/training/train_action_brain.py --smoke-test
```

Current orchestration artifacts:
- `tools/infra/cloud_train_wrapper.sh`
- `vertex_job.yaml`

Use cloud for:
- larger training runs
- longer evaluations
- repeatable artifact publication
- research dependencies that are not yet part of the native Orin runtime path, such as `Grounding DINO`

Current Grounding DINO boundary:
- `Dockerfile` carries the Hugging Face and transformers stack needed for Grounding DINO bootstrap experiments
- the cloud image installs it in editable mode with `--no-deps`
- this is intentionally a cloud/x86 packaging feature only
- `Dockerfile.orin` is unchanged; Grounding DINO is not yet part of the Jetson runtime contract

Rollback-safe experimental Orin path:
- `Dockerfile.orin` remains the stable Jetson baseline
- `Dockerfile.orin.dinov3` is legacy naming for the old bootstrap experiment and should not be treated as the active Grounding DINO runtime contract
- `Dockerfile.orin.sam3` is a separate experimental JetPack 6 path for official `facebookresearch/sam3`
- rollback is operationally trivial:
  - keep using `Dockerfile.orin`
  - or stop tagging/publishing the experimental image
- recommended image naming:
  - stable: `hoopsense-orin:stable`
  - experimental: `hoopsense-orin:grounding-dino-exp1`
  - experimental SAM3: `hoopsense-orin:sam3-exp1`

Experimental SAM3 container notes:
- this path is for JetPack 6 / L4T R36.x hosts only
- it is intended to validate the full Layer 1 artifact generation path end to end, not just a SAM sidecar
- it currently builds from the official NVIDIA image `nvcr.io/nvidia/l4t-jetpack:r36.4.0` and layers the Jetson PyTorch wheel plus repo-local dependencies on top
- use:
  - `scripts/build_orin_sam3_docker.sh`
  - `scripts/run_orin_layer1_docker.sh`
  - `scripts/generate_layer1_annotations_docker.sh`
  - `scripts/run_orin_cuda_probe_docker.sh`
- the stable native Orin path remains:
  - `scripts/setup_orin.sh`
  - `scripts/run_orin_layer1_python.sh`
- do not mount `.venv_orin310` into the container; the Python 3.12 runtime must stay self-contained

## Reproducibility Strategy

HoopSense uses this priority order:
1. explicit target-specific scripts and pinned runtime assumptions
2. `Docker` for cloud packaging and experimental runtime isolation
3. optional repo-managed shells where they still provide local value

This means:
- the stable native Orin scripts are the source of truth for Jetson validation
- Docker is the primary boundary for cloud work and experimental JetPack 6 / SAM3 paths
- checked-in Guix files are legacy conveniences, not the primary operating model

## CI Strategy

CI should be split by functional core, not monolithic.

Planned core pipelines:
- `perception-inference`
- `ml-training`
- `tooling-review`
- `docs-plan`
- `packaging-runtime`
- `core-logic`

Rules:
- run fast checks by default
- run training only when ML paths change or a manual trigger requests it
- run an integration pipeline only when cross-core validation is needed

Minimum CI checks to add first:
- Python compile and unit tests
- Rust build and tests
- docs and plan coherence
- contract validation

## Infrastructure Strategy

Use direct `gcloud` while the cloud surface is still moving.

Move stable shared infrastructure into Terraform when the shape settles:
- buckets
- registry
- service accounts
- IAM bindings
- stable Vertex-related resources

Do not introduce Terraform for still-fluid experiments.

## Near-Term Priorities

1. Keep the Orin validation path documented and repeatable.
2. Make Docker docs explicitly cloud-only.
3. Add the first checked-in CI workflow.
4. Publish separate Jetson and cloud runbooks.
5. Add Terraform only for stable shared GCP resources.

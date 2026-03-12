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
- `guix.scm` for the general development shell
- `guix_orin.scm` for the Orin-oriented shell
- `scripts/setup_orin.sh` for the Guix plus JetPack bridge
- `Dockerfile` for cloud-oriented packaging
- `tools/infra/cloud_train_wrapper.sh` and `vertex_job.yaml` for cloud training entrypoints

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
guix shell -m guix.scm
guix shell -m guix.scm -- python3 -m unittest
```

Use this for:
- normal development
- docs and plan updates
- unit tests
- non-device-specific validation

### Mode B: Jetson Orin Smoke Tests and Early Training

This is the current first-class target.

Preferred path:

```bash
guix shell -m guix_orin.scm
./scripts/setup_orin.sh
python3 tests/validate_orin_env.py
python3 tools/training/train_action_brain.py --smoke-test
```

Use this for:
- GPU smoke validation
- first training runs
- confirming model checkpoints progress correctly on target hardware
- catching Jetson-specific ABI or driver issues before cloud spend

Constraint:
- Jetson depends on host-installed NVIDIA libraries and JetPack components
- this path is reproducible only up to that vendor boundary

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

## Reproducibility Strategy

HoopSense uses this priority order:
1. `Guix` where the environment can be controlled directly
2. `Guix` plus explicit vendor bridge on Jetson
3. `Docker` where packaging or runtime isolation is needed

This means:
- Guix is the preferred developer entrypoint
- Docker is the fallback packaging boundary
- Docker should not be presented as the native Orin environment

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

# HoopSense DevOps Strategy

This document defines the practical DevOps strategy for HoopSense across local development, CI, cloud training, and Jetson deployment.

## 1. Scope

The strategy covers:
- environment definitions
- build and test automation
- CI quality gates
- cloud image packaging
- cloud training execution
- Jetson/Orin runtime execution
- artifact compatibility and publication

## 2. Strategy Goals

1. Keep local development reproducible with Guix where possible.
2. Provide a clear Docker fallback for cloud packaging and constrained runtimes.
3. Separate cloud/x86 and Jetson/ARM64 operating modes explicitly.
4. Make CI enforce code, contract, and documentation quality.
5. Make build and deployment artifacts traceable and repeatable.

## 3. Current Artifact Baseline

Current repo artifacts:
- `guix.scm`
  General development shell.
- `guix_orin.scm`
  Orin-oriented shell manifest.
- `Dockerfile`
  Cloud training container image.
- `tools/infra/cloud_train_wrapper.sh`
  Cloud entrypoint for training, GCS sync, and model export.
- `vertex_job.yaml`
  Vertex AI custom-job spec.
- `tools/infra/colab_manager.py`
  GCP/Colab runtime helper.
- `scripts/setup_orin.sh`
  Orin bridge script for Guix plus JetPack runtime access.

Important current gap:
- no checked-in CI workflow yet

## 4. DevOps Requirements

### Functional Requirements

- **DOR-01: Versioned Environment Definitions**
  Local and target environments must be defined in checked-in manifests or build specs.

- **DOR-02: Guix-First Development**
  Local development and test workflows should prefer Guix-managed shells where supported.

- **DOR-03: Docker Fallback Packaging**
  Cloud and other constrained runtimes must have containerized fallback packaging.

- **DOR-04: CI Quality Gates**
  CI must run the relevant test, lint, contract, and documentation checks for changed areas.

- **DOR-05: Cloud Training Packaging**
  The project must support building a cloud training image and running it via a documented job spec.

- **DOR-06: Edge Runtime Guidance**
  The project must provide a documented runtime path for Jetson/Orin devices, including vendor-boundary assumptions.

- **DOR-07: Artifact Compatibility Reporting**
  Build and runtime artifacts must state which targets they support.

- **DOR-08: Publication Discipline**
  Image, model, and dataset publication paths must be explicit and reviewable.

### Non-Functional Requirements

- **DON-01: Reproducibility**
  Developers should be able to recreate intended environments from checked-in definitions.

- **DON-02: Clarity**
  The repo must not imply that one runtime path covers cloud/x86 and Jetson/ARM64 when it does not.

- **DON-03: Verification**
  Environment and deployment claims should be backed by runnable validation steps.

- **DON-04: Operational Simplicity**
  Common developer and operator tasks should have short, documented execution paths.

## 5. Architecture

### 5.1 Environment Layers

HoopSense DevOps uses four environment layers:

1. **Developer Shell**
   Prefer Guix via `guix.scm`.

2. **Jetson-Oriented Shell**
   Prefer Guix plus documented host-library bridging via `guix_orin.scm` and `scripts/setup_orin.sh`.

3. **Cloud Training Image**
   Use `Dockerfile` plus `tools/infra/cloud_train_wrapper.sh`.

4. **Cloud Job Definition**
   Use `vertex_job.yaml` or equivalent job runner configuration.

### 5.2 CI Layers

The intended CI pipeline should have staged gates:

1. **Fast checks**
   - Python syntax/compilation
   - unit tests
   - Rust build/test

2. **Contract checks**
   - schema validation
   - feature contract validation
   - doc/plan coherence checks

3. **Packaging checks**
   - Docker image build
   - optional Guix shell smoke command

4. **Target checks**
   - cloud job spec validation
   - Jetson runtime docs/commands consistency check

There is no checked-in GitHub Actions workflow yet, so this section is strategic design rather than current implementation.

### 5.3 Deployment Modes

#### Mode A: Local Development
- primary path: `guix shell -m guix.scm`
- purpose: day-to-day development, tests, and non-device-specific runs

#### Mode B: Cloud Training
- primary path: Docker image build from `Dockerfile`
- runtime entrypoint: `tools/infra/cloud_train_wrapper.sh`
- orchestration target: Vertex AI or equivalent GCP GPU runner

#### Mode C: Jetson / Orin Runtime
- primary path: host JetPack plus Guix-managed userland where possible
- bridge path: `scripts/setup_orin.sh`
- caveat: vendor libraries remain outside pure Guix control

## 6. Design

### 6.1 Guix Design

Guix should define:
- language runtimes
- compilers
- core libraries
- repeatable developer tooling

Guix should be the default documented entrypoint for repo development.

Canonical commands:
```bash
guix shell -m guix.scm
guix shell -m guix.scm -- python3 -m unittest
```

### 6.2 Docker Design

Docker should be used for:
- cloud training images
- packaging a known runtime boundary
- environments where platform tooling expects a container

The Docker image should be described honestly:
- cloud/x86 image for GPU training
- not the authoritative Jetson runtime story

Canonical cloud packaging commands:
```bash
docker build -t hoopsense-cloud:latest .
docker run --rm -it hoopsense-cloud:latest python3 tools/training/train_action_brain.py
```

### 6.3 CI Design

The target CI design should include:
- push and PR validation
- matrix-aware checks where helpful
- separation between fast gates and slower packaging gates
- explicit failure on plan/doc drift for architecture-heavy changes

### 6.4 Cloud Design

Cloud training should be treated as a controlled path:
- image build
- dataset/material sync
- training run
- model export
- lineage and compatibility recording

The existing `cloud_train_wrapper.sh` and `vertex_job.yaml` are the seeds of that path.

### 6.5 Edge Design

Edge runtime should be treated as a documented compatibility target, not a copy of cloud runtime assumptions.

For Jetson/Orin:
- document host requirements
- document validation commands
- document where Guix ends and JetPack begins

Canonical Jetson-oriented commands:
```bash
bash scripts/setup_orin.sh
./hoops-orin-shell python3 tools/training/train_action_brain.py
```

Current caveat:
- `scripts/setup_orin.sh` depends on host JetPack/NVIDIA libraries and is not a pure Guix runtime.

## 7. Recommended Near-Term Implementation Order

1. Write and maintain the DevOps strategy and target operating modes.
2. Separate cloud/x86 and Jetson/ARM64 guidance in docs and scripts.
3. Add a checked-in CI workflow for Python, Rust, docs, and contract checks.
4. Add Docker build validation in CI.
5. Add Guix smoke validation where practical.
6. Add compatibility records for cloud and Jetson targets.

## 8. Explicit Current Gaps

- no checked-in CI workflow
- Dockerfile is cloud-oriented and should not be described as the native Orin image
- Orin setup remains dependent on host-installed NVIDIA libraries
- no multi-arch container strategy is defined yet; cloud/x86 and Orin are still separate stories
- cloud and edge artifact compatibility is not yet formally recorded

These gaps should remain visible in planning until implemented.

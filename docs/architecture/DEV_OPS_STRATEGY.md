# HoopSense DevOps Strategy

This document defines the practical DevOps strategy for HoopSense across local development, CI, cloud training, and Jetson deployment.

In HoopSense, DevOps combines:
- `Guix` for reproducible development environments where possible
- `Docker` for fallback packaging and runtime boundaries
- `Terraform` for stable shared cloud infrastructure once the platform surface stops changing rapidly

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
- no checked-in Terraform layer yet

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

- **DOR-09: Infrastructure-as-Code**
  Stable shared cloud infrastructure should move into version-controlled IaC rather than remain ad hoc.

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

5. **Infrastructure Definition**
   Use Terraform for stable shared GCP resources once the resource model justifies it.

### 5.2 CI Layers

The intended CI system should be multi-pipeline, not monolithic.

#### Functional-Core Pipelines

HoopSense should isolate CI by functional core:
- `core-logic`
  - Rust math, ledger logic, native contracts
- `perception-inference`
  - Python inference, feature construction, runtime checks
- `ml-training`
  - dataset validation, training code, evaluation/report generation
- `tooling-review`
  - review tooling, infra scripts, agent bridges
- `docs-plan`
  - requirements, architecture, plan tree, task snapshot
- `packaging-runtime`
  - Guix manifests, Docker build, cloud specs, Orin scripts

These pipelines should be triggered primarily by path ownership.

#### CI Tiers

The intended CI pipeline stack should have staged gates:

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

5. **Conditional heavy checks**
   - training smoke runs
   - heavier evaluation
   - packaging publication checks

Training should not be a default CI step. It should run only when:
- training code changes
- feature/schema contracts change
- Oracle/data-generation code changes
- training config changes
- or a manual/label-based trigger requests it

There is no checked-in GitHub Actions workflow yet, so this section is strategic design rather than current implementation.

#### Integration Pipeline

Above the functional cores, HoopSense should have an integration pipeline that:
- depends on or triggers selected core pipelines
- verifies cross-core contract compatibility
- runs end-to-end smoke validation when warranted

This keeps routine PR checks fast while still allowing a broader integration run.

### 5.3 Deployment Modes

#### Mode A: Local Development
- primary path: `guix shell -m guix.scm`
- purpose: day-to-day development, tests, and non-device-specific runs

#### Mode B: Cloud Training
- primary path: Docker image build from `Dockerfile`
- runtime entrypoint: `tools/infra/cloud_train_wrapper.sh`
- orchestration target: Vertex AI or equivalent GCP GPU runner
- infrastructure control: direct `gcloud` first, Terraform when shared resources stabilize

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
- path-based triggering by functional core
- optional/manual integration runs
- conditional ML/training runs instead of always-on training

Recommended first workflow split:
1. `core-logic`
2. `perception-inference`
3. `docs-plan`
4. `packaging-runtime`
5. `integration`
6. `ml-training` only on relevant changes or manual trigger

## 6.6 CI Tooling Guidance

Good CI tooling for HoopSense should be judged by:
- cost
- path-based triggering
- artifact handling
- secrets/GCP integration
- support for manual and scheduled runs

Practical options:

### GitHub Actions
- best default if the repo stays on GitHub
- cheap and simple for CPU-heavy checks
- easy path filters, matrices, artifacts, and manual triggers
- not ideal for frequent GPU training in hosted runners

### Buildkite / self-hosted runners
- strong option if you want your own machines or Jetson/cloud runners
- better when you want exact control over hardware and cost
- more operational overhead than GitHub Actions

### Google Cloud Build / Vertex-triggered workflows
- good for cloud-native packaging and training orchestration
- useful for image builds and heavy GCP-integrated jobs
- weaker as the only general-purpose CI surface for docs/tests/plans

Recommended stance:
- use a normal CI system for code/docs/contracts
- use GCP-native jobs for heavy build/train/integration steps when needed
- do not run expensive GPU training on every PR

### 6.4 Cloud Design

Cloud training should be treated as a controlled path:
- image build
- dataset/material sync
- training run
- model export
- lineage and compatibility recording

The existing `cloud_train_wrapper.sh` and `vertex_job.yaml` are the seeds of that path.

### 6.5 Infrastructure-as-Code Design

Terraform should own stable shared cloud resources such as:
- GCS buckets
- Artifact Registry repositories
- service accounts and IAM bindings
- long-lived build or training support resources

Direct `gcloud` should remain acceptable for:
- early prototyping
- one-off jobs
- rapidly changing experimental workflows

Recommended operating split:
- `Guix`: reproducible local and developer environments
- `Docker`: portable cloud runtime/image boundary
- `Terraform`: durable shared GCP resource management
- `gcloud`: iterative cloud orchestration during early stages

### 6.6 Edge Design

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

## 6.7 Infrastructure-as-Code Guidance

Terraform is optional, not mandatory.

Use Terraform when:
- you are managing repeatable cloud infrastructure at non-trivial scope
- you want stable GCS buckets, service accounts, artifact registries, and job definitions

Use direct `gcloud` when:
- the system is still evolving quickly
- the infrastructure footprint is small
- you are prototyping cloud workflows

Recommended current stance for HoopSense:
- direct `gcloud` is acceptable now
- adopt Terraform once the cloud footprint stabilizes
- keep job specs and environment definitions in repo even before Terraform

Practical split:
- `Guix` for environments
- `Docker` for images
- `Terraform` for shared infrastructure
- `gcloud` for early orchestration and experiments

## 7. Recommended Near-Term Implementation Order

1. Write and maintain the DevOps strategy and target operating modes.
2. Separate cloud/x86 and Jetson/ARM64 guidance in docs and scripts.
3. Add a checked-in CI workflow for Python, Rust, docs, and contract checks.
4. Split CI into functional-core workflows with a separate integration pipeline.
5. Add Docker build validation in CI.
6. Add Guix smoke validation where practical.
7. Add compatibility records for cloud and Jetson targets.
8. Add Terraform once the shared GCP surface is stable enough to justify IaC maintenance.

## 8. Explicit Current Gaps

- no checked-in CI workflow
- no functional-core CI split yet
- Dockerfile is cloud-oriented and should not be described as the native Orin image
- Orin setup remains dependent on host-installed NVIDIA libraries
- no multi-arch container strategy is defined yet; cloud/x86 and Orin are still separate stories
- cloud and edge artifact compatibility is not yet formally recorded
- no Terraform layer exists yet for shared GCP infrastructure

These gaps should remain visible in planning until implemented.

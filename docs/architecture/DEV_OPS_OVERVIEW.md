# HoopSense DevOps Overview

HoopSense uses DevOps to make development, build, test, training, and deployment flows reproducible across local machines, cloud GPUs, and Jetson-class edge devices.

## Core Rule

The DevOps rule for HoopSense is:

`use Guix for reproducibility where possible; use Docker as the fallback packaging and runtime boundary where Guix alone is not sufficient`

This reflects the actual repo shape:
- `guix.scm` for general development
- `guix_orin.scm` for Orin-oriented development/runtime support
- `Dockerfile` and `vertex_job.yaml` for cloud training packaging
- `scripts/setup_orin.sh` for Jetson/Orin runtime bridging

## What DevOps Covers

In HoopSense, DevOps covers:
- reproducible developer environments
- build and test automation
- packaging for cloud and edge runtime targets
- artifact movement between local, cloud, and device environments
- CI quality gates
- deployment and runtime compatibility

## Why HoopSense Needs This

HoopSense spans multiple environments with different constraints:
- local development laptops
- cloud/x86 GPU training environments
- Jetson/Orin ARM64 inference or training targets

The build and runtime story cannot assume one universal environment. DevOps must manage that complexity without hiding it.

## Platform Position

### Preferred: Guix

Guix is the preferred environment mechanism when:
- the dependency is available and stable in Guix
- reproducibility matters more than vendor-runtime coupling
- the task is local development, testing, or non-vendor-specific execution

### Fallback: Docker

Docker is the fallback when:
- cloud training platforms expect container images
- vendor CUDA/runtime requirements make pure Guix impractical
- deployment needs a standardized runtime boundary

### Edge Reality: Jetson / Orin

Jetson-class devices require explicit interaction with host-installed vendor libraries and drivers. In this case:
- Guix should still define as much of the userland as possible
- the host JetPack environment remains part of the runtime contract
- Docker may be used where it simplifies runtime parity or vendor access

## Current Repo Reality

What exists now:
- Guix manifests for local and Orin-oriented environments
- a Docker image for cloud training
- a Vertex AI job spec
- a cloud training wrapper for GCS sync and model export
- a Colab/GCP helper path
- an Orin setup script that bridges Guix and JetPack libraries

What does not exist yet:
- a checked-in CI workflow under `.github/workflows/`
- a complete release pipeline
- a fully separated cloud/x86 vs Jetson/ARM64 deployment guide
- a formal artifact publication matrix

The DevOps strategy should document both the working parts and the missing parts.

# HoopSense MLOps

This document defines the ML lifecycle inside the broader HoopSense operations strategy.

See [DEVOPS.md](/data/projects/hoopsense/docs/architecture/devops/DEVOPS.md) for environment, CI, packaging, and infrastructure guidance.

## Scope

MLOps in HoopSense covers:
- dataset materialization and manifests
- feature and schema validation
- training run lineage
- evaluation and promotion
- deployment compatibility reporting

It applies to:
- Oracle-generated Action Brain datasets
- training and checkpoint artifacts
- later learned layers that may be added above Action Brain

## Core Rules

1. Treat datasets as first-class artifacts.
2. Keep feature contracts explicit and versioned.
3. Record lineage for every training run and checkpoint.
4. Promote models only with evidence.
5. Keep Jetson and cloud compatibility explicit.

## Artifact Classes

The current artifact graph is:
- `dataset`
- `dataset_manifest`
- `feature_contract`
- `training_config`
- `checkpoint`
- `evaluation_report`
- `promotion_record`
- `deployment_compatibility_report`

## Training Lifecycle

1. Ingest or generate source data.
2. Validate schema, geometry, and feature compatibility.
3. Materialize dataset plus manifest.
4. Run training and record lineage.
5. Generate evaluation output.
6. Mark the checkpoint as candidate, validated, default, or retired.

## Orin-First Training Discipline

Near-term training order:
1. Run smoke tests and short initial training on Orin.
2. Confirm checkpoints write correctly and metrics move in the expected direction.
3. Move to cloud only after the Orin path is stable.

This keeps early failures cheap and catches target-specific issues before cloud spend.

## Required Controls

### Dataset Controls

Each training dataset should have:
- a dataset id
- schema version
- lineage to the generating code and source data
- label counts
- validation summary

Oracle datasets should additionally record:
- subject and trial coverage
- parser/FK version
- projection settings
- feature generation version

### Training Run Controls

Each training run should record:
- run id
- code revision
- dataset id
- dataset manifest reference
- config snapshot
- environment target
- output checkpoint paths
- evaluation paths

### Evaluation Controls

Each evaluation should include:
- aggregate metrics
- per-class metrics
- confusion matrix
- slice output where available

### Promotion Controls

Checkpoint lifecycle states:
- `candidate`
- `validated`
- `default`
- `retired`

Promotion should not be implicit.

## Near-Term Priorities

1. Add Oracle dataset manifests and validation checks.
2. Record Action Brain training lineage beside checkpoints.
3. Add first slice-based evaluation reports.
4. Record cloud/x86 versus Jetson/ARM64 compatibility explicitly.

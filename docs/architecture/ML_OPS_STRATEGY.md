# HoopSense MLOps Strategy

This document defines the practical MLOps strategy for HoopSense across requirements, architecture, and design.

## 1. Scope

The strategy covers:
- data creation and governance
- feature and schema governance
- model training and evaluation
- model promotion and deployment
- monitoring, drift, and feedback loops

It applies to:
- perception-related training artifacts
- Action Brain training and evaluation
- later learned components that may be added on top of possession or event reasoning

## 2. Strategy Goals

1. Make training and evaluation reproducible.
2. Keep feature contracts stable and auditable.
3. Make model promotion evidence-based.
4. Detect drift between Oracle assumptions and real basketball footage.
5. Preserve parity across cloud training and edge deployment.
6. Support layered debugging across perception, action, possession, events, and stats.

## 3. MLOps Requirements

### Functional Requirements

- **MLR-01: Dataset Manifests**
  Every training dataset must ship with a manifest containing source lineage, schema version, label distribution, generation recipe, and validation summary.

- **MLR-01a: Dataset Version Lineage**
  Training datasets such as `synthetic_dataset_v2.jsonl` and `oracle_dataset_v3.jsonl` must have explicit lineage and promotion relationships rather than being treated as ad hoc replacements.

- **MLR-02: Contract Validation**
  Training and inference inputs must be validated against explicit schema and feature contracts before use.

- **MLR-03: Reproducible Training Runs**
  Every training run must record code revision, dataset version, config, environment target, and resulting artifacts.

- **MLR-04: Slice-Based Evaluation**
  Evaluation must report both aggregate metrics and slice metrics for known failure axes.

- **MLR-05: Promotion Workflow**
  Model artifacts must move through explicit lifecycle states with validation gates.

- **MLR-06: Drift Detection**
  The system must detect shifts in data quality, action distribution, pose quality, and ball-observation quality over time.

- **MLR-07: Feedback Intake**
  Human validation signals, referee-aligned events, and curated corrections must be ingestible as feedback artifacts.

- **MLR-08: Environment Compatibility Tracking**
  Training and inference artifacts must declare compatible runtime targets such as cloud/x86 or Jetson/ARM64.

### Non-Functional Requirements

- **MLN-01: Auditability**
  A promoted model must be traceable to the exact data and code that produced it.

- **MLN-02: Explainability**
  Evaluation and production failures must be localizable to a layer or artifact class.

- **MLN-03: Repeatability**
  The same dataset, config, and code revision should reproduce materially similar training outputs.

- **MLN-04: Operational Clarity**
  Operators must be able to determine which model is active, why it was promoted, and what evidence supports it.

## 4. Architecture

### 4.1 Artifact Classes

The core artifact classes are:
- `dataset`
- `dataset_manifest`
- `feature_contract`
- `training_config`
- `checkpoint`
- `evaluation_report`
- `promotion_record`
- `deployment_compatibility_report`

### 4.2 Pipeline Stages

1. **Ingest**
   Raw source assets, Oracle source files, or real-world labeled examples enter the system.

2. **Validate**
   Data shape, schema, label quality, geometry sanity, and feature compatibility are checked.

3. **Materialize**
   The dataset and manifest are written as stable artifacts.

4. **Train**
   A candidate model is trained with explicit config and lineage recording.

5. **Evaluate**
   Aggregate and slice-based metrics are generated.

6. **Promote or Reject**
   The artifact is marked `validated` or rejected based on evidence.

7. **Deploy**
   Runtime-compatible artifacts are published to the intended target.

8. **Monitor**
   Real-world drift and failure slices are tracked.

### 4.3 Layer-Aware Ownership

- Perception data quality is owned by ingestion and tracking validation.
- `features_v2` quality is owned by Action Brain feature validation.
- possession/event/stat correctness is owned by higher-layer contract and ledger validation.

This prevents the model from absorbing all blame for failures that originate elsewhere.

## 5. Design

### 5.1 Dataset Design

Each dataset should have:
- a stable dataset identifier
- a manifest file
- schema version
- source lineage
- label counts
- validation outcomes
- recommended training uses

For Oracle datasets, the manifest should also include:
- source subjects and trials
- parser/FK version
- projection settings
- feature generation version
- predecessor dataset id when superseding an older dataset

### 5.2 Feature Governance

Feature governance follows the layered architecture:
- `features_v2` remains the frozen local-action contract
- possession context gets its own schema
- event attribution artifacts get their own schema
- stat outputs get their own schema

No layer should silently absorb schema changes from another layer.

### 5.3 Training Run Design

Each training run should emit:
- run id
- dataset id
- dataset manifest reference
- code revision
- config snapshot
- environment target
- checkpoint paths
- evaluation paths
- promotion recommendation

### 5.4 Evaluation Design

Evaluation should include:
- aggregate accuracy/F1
- per-class metrics
- confusion matrix
- slice metrics by camera/view/pose quality/ball visibility
- notes on unsupported or low-confidence regions

### 5.5 Promotion Design

Promotion should be explicit:
- `candidate`: training succeeded
- `validated`: evidence meets the current bar
- `default`: chosen for standard inference
- `retired`: superseded or unsafe

Promotion records should include:
- artifact ids
- who/what promoted the model
- evidence summary
- known limitations

### 5.6 Monitoring Design

Monitoring should track:
- label distribution drift
- pose confidence drift
- ball-observation drift
- action-frequency drift
- evaluation regressions by slice
- deployment-target incompatibilities

## 6. Recommended Near-Term Implementation Order

1. Create dataset manifest and validation conventions for Oracle data.
2. Define explicit lineage between current synthetic and Oracle dataset generations.
3. Add training-run lineage recording for Action Brain checkpoints.
4. Add slice-based evaluation output for current Oracle and real-pose experiments.
5. Add explicit checkpoint lifecycle states and promotion metadata.
6. Separate cloud/x86 and Jetson/ARM64 runtime compatibility reporting.

## 7. Out of Scope for the First MLOps Slice

- full online retraining automation
- automated model rollback orchestration
- generalized feature store infrastructure
- production-grade distributed training

The first priority is artifact discipline and reliable promotion, not platform maximalism.

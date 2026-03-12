# HoopSense MLOps Overview

HoopSense does not treat machine learning as a single training script or a single checkpoint file. It treats ML as a governed product subsystem with explicit data, model, evaluation, and promotion lifecycles.

## Purpose

The purpose of MLOps in HoopSense is to make model behavior:
- reproducible
- debuggable
- promotable
- auditable
- safe to evolve across data, features, and deployment targets

This matters because HoopSense is not just producing model logits. It is producing basketball events, possession state, and stats that users may trust as if they were official.

## What MLOps Covers

In HoopSense, MLOps covers:
- dataset creation and versioning
- feature-contract versioning
- training run orchestration
- evaluation and slice analysis
- checkpoint lineage and promotion
- environment parity across cloud and edge targets
- drift monitoring and feedback loops

## Core Principle

The central MLOps rule is:

`data artifacts, feature contracts, and model artifacts must be versioned and promoted deliberately`

That means:
- a dataset is a product artifact
- a feature schema is a contract
- a checkpoint is a candidate until promoted
- a model is not trusted just because it trained successfully

## Why HoopSense Needs This

HoopSense has unusual failure modes compared with generic image classifiers:
- pose quality can collapse while raw detections still look fine
- ball state can drift independently from player pose
- a model can classify local motion correctly while event attribution is still wrong
- synthetic Oracle data can look statistically healthy while still missing real-game variability
- edge deployment on Jetson/Orin can diverge from cloud/x86 training behavior

These are MLOps problems, not just model-architecture problems.

## Product Boundaries

MLOps sits across the full layered architecture:
- Layer 1 Perception: validate detections, tracks, pose streams, and ball trajectories
- Layer 2 Action Brain: freeze and validate `features_v2`, training inputs, and local action metrics
- Layer 3 Possession Context: validate derived semantic state contracts
- Layer 4 Event Attribution: evaluate event precision/recall, not just model logits
- Layer 5 Stat Generation: verify stat consistency, explainability, and ledger correctness

## Main Artifact Types

The main MLOps artifacts in HoopSense are:
- dataset manifests
- schema versions
- feature validators
- training configs
- evaluation reports
- promotion decisions
- deployment compatibility reports

## Minimum Viable MLOps

The minimum acceptable MLOps system for HoopSense should provide:
1. dataset manifests and lineage for every training dataset
2. feature-contract validation before training and inference
3. reproducible training runs with code/data/config traceability
4. slice-based evaluation, not just aggregate accuracy
5. explicit model promotion states such as `candidate`, `validated`, and `default`
6. environment separation between Jetson/ARM64 runtime and cloud/x86 training

## Relationship to Other Docs

- `docs/architecture/components/FEATURE_SCHEMA_V2.md`
  Defines the current Action Brain input contract.
- `docs/architecture/components/LAYERED_FEATURE_SCHEMA.md`
  Defines where model inputs stop and higher-order basketball reasoning begins.
- `docs/architecture/ML_OPS_PHILOSOPHY.md`
  Defines the operating beliefs that guide model and data decisions.
- `docs/architecture/ML_OPS_STRATEGY.md`
  Defines the concrete MLOps requirements, architecture, and design strategy.

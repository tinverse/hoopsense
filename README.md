# HoopSense OS
Practical game understanding from basketball video.

## Architecture Snapshot
- `features_v2` is the current frozen Action Brain contract for local motion classification.
- The Action Brain is a narrow temporal classifier, not the full game-reasoning system.
- Possession context, event attribution, and stat generation are layered above it.
- MLOps is part of the product architecture: datasets, manifests, evaluations, and promoted checkpoints are first-class artifacts.
- DevOps is also part of the product architecture: Guix is the preferred reproducibility path, with Docker as the fallback packaging/runtime path.
- See `docs/architecture/components/LAYERED_FEATURE_SCHEMA.md` for the current system feature model.
- See `docs/architecture/ML_OPS_STRATEGY.md` for the current ML lifecycle strategy.
- See `docs/architecture/DEV_OPS_STRATEGY.md` for the current build, CI, and deployment strategy.

## Project Structure
- `core/`: Native perception and logic internals.
- `pipelines/`: Deterministic execution stages.
- `docs/`: Design-first documentation (Requirements, Architecture, Book).
- `cli/`: Local orchestration and Gemini Agent tools.
- `data/`: Local storage for training manifests and video links.
- `TASK_STATUS.md`: Current execution frontier.

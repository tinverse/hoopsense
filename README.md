# HoopSense OS
Practical game understanding from basketball video.

## Architecture Snapshot
- `features_v2` is the current frozen Action Brain contract for local motion classification.
- The Action Brain is a narrow temporal classifier, not the full game-reasoning system.
- Possession context, event attribution, and stat generation are layered above it.
- See `docs/architecture/components/LAYERED_FEATURE_SCHEMA.md` for the current system feature model.

## Project Structure
- `core/`: Native perception and logic internals.
- `pipelines/`: Deterministic execution stages.
- `docs/`: Design-first documentation (Requirements, Architecture, Book).
- `cli/`: Local orchestration and Gemini Agent tools.
- `data/`: Local storage for training manifests and video links.
- `TASK_STATUS.md`: Current execution frontier.

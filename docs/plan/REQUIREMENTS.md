# HoopSense Requirements

HoopSense is a "Sports OS" designed to transform raw basketball footage into a professional-grade data, media, and gaming ecosystem.

## Functional Requirements
- **FR-01: Multi-Source Ingestion:** Support video from 720p (Mobile/IMU-fused) to 4K (Fixed HoopBox/DSLR).
- **FR-02: Spatial Resolution:** Map 2D pixel coordinates to 3D court coordinates (X, Y, Z in cm) with <15cm error.
- **FR-03: Temporal Consistency:** Maintain persistent identity (Player ID) across camera cuts, pans, and occlusions.
- **FR-04: Action Recognition:** Identify basketball-specific events (Dribble, Shot, Pass, Crossover, Euro-step).
- **FR-04a: Local Motion Classification:** Classify short-horizon player motion from synchronized pose and ball features.
- **FR-04b: Possession Context Modeling:** Track possession origin, ballhandler, dribble count, pass chains, and offense zone outside the Action Brain.
- **FR-04c: Event Attribution:** Combine local action predictions with possession context and geometry to infer shots, passes, rebounds, steals, fouls, and turnovers.
- **FR-05: Referee API:** Decode official hand signals within a 2-second window to validate points and fouls.
- **FR-06: Game State Ledger:** Implement a retroactive event-rewind system for fouls and score corrections.
- **FR-07: Avatar Generation:** Export skeletal rigs in .gltf format compatible with Roblox/Minecraft/Unity.
- FR-08: Defensive Intelligence: Calculate contest radii, defensive pressure, and "IQ" stats based on defender proximity.
- **FR-10: Layered Feature Contracts:** Maintain a stable Action Brain feature contract (`features_v2`) while allowing richer possession, event, and stat features in higher layers.
- **FR-11: Stat Generation:** Generate box-score events and derived player metrics from attributed events, not directly from neural logits.
- **FR-12: Dataset Governance:** Every training dataset must have a manifest with lineage, schema version, label distribution, and validation summary.
- **FR-13: Training Lineage:** Every meaningful training run must record code revision, dataset version, config, and environment target.
- **FR-14: Model Promotion Workflow:** Model artifacts must move through explicit lifecycle states with validation gates before becoming default inference models.
- **FR-15: Slice-Based Evaluation:** Evaluation must report class-level and slice-level quality, not just aggregate accuracy.
- **FR-16: Drift Monitoring:** The system must detect drift in action distribution, pose quality, ball visibility, and environment characteristics over time.
- **FR-17: Deployment Compatibility Tracking:** Each model artifact must declare and validate compatible runtime targets such as cloud/x86 and Jetson/ARM64.
- FR-09: Action Taxonomy & Signal Decoding:
    - **Scoring Actions:** 1-pt, 2-pt, 3-pt attempts vs. makes.
    - **Dribble Actions:** Crossover, Between-the-legs, Behind-the-back, Hesitation.
    - **Shot Actions:** Jump shot, Layup, Dunk, Hook shot, Step-back.
    - **Official Signals:** 3-pt attempt/success, Time-out, Foul, Substitution, Traveling, Double Dribble.
    - **Violation Actions:** Traveling (visual), Double-dribble (visual), Kicking (visual).


## Non-Functional Requirements
- **NFR-01: Performance:** Process a 1-hour game in <2 hours on standard GPU hardware.
- **NFR-02: Security:** Obfuscate and encrypt model weights in mobile thin clients to protect intellectual property.
- **NFR-03: Resilience:** Self-healing calibration for "HoopBox" mounts subject to vibration or accidental shifts.
- **NFR-04: Privacy:** Automatic blurring of non-consenting bystanders and COPPA compliance for youth athletes.
- **NFR-05: Portability:** Core vision internals (Spatial Resolver, Coordinate Fusion) implemented in Rust for cross-platform efficiency.
- **NFR-06: Auditability:** A promoted model must be traceable to the exact data, config, code revision, and evaluation evidence that produced it.
- **NFR-07: Repeatability:** Re-running the same training configuration on the same dataset should yield materially similar outputs and metrics.
- **NFR-08: Operational Clarity:** Operators must be able to identify the active model, its promotion status, and its known limitations.

## Technical Constraints
- **Primary Languages:** Rust (Performance/IP), Python (AI Orchestration/Training).
- **Rule Hierarchy:** NCAA > NBA > FIBA (Precedence for signal decoding and foul logic).
- **ML Frameworks:** YOLOv11+ for detection, ViTPose/RTMPose for skeletal rigs, BoT-SORT for memory-backed tracking.
- **Environment:** Guix-managed toolchains for deterministic builds.
- **Feature Layering Constraint:** The Action Brain remains a narrow local classifier; possession logic and stats must remain separable and auditable.
- **MLOps Constraint:** Dataset, feature, evaluation, and promotion artifacts must be versioned and reviewable as first-class project outputs.

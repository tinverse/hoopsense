# Action Brain: Temporal Transformer Design

The Action Brain is the learned local-motion classifier in HoopSense. It answers a narrow question: `what motion is this player performing over the next short window?`

It is not the full game-reasoning system. Possession logic, event attribution, and stat generation live in the higher layers documented in `LAYERED_FEATURE_SCHEMA.md`.

## 1. Input Specification (The Multimodal Feature Tensor)
Instead of raw pixels, we feed a 3D tensor of shape `(Batch, 30, D)`.
- **Primary Features (34 dims):** 17 player keypoints (x, y) normalized to bounding box.
- **Kinematic Features (34 dims):** First-order velocity (Δx, Δy) for all joints.
- **Game Context (4 dims):**
    - `ball_dist_wrist_l`, `ball_dist_wrist_r`: Distance to the ball (cm).
    - `court_x`, `court_y`: Global position on the court (cm).
- **Total Dimension (D):** 72.

This input contract should stay narrow. Possession reasoning, event attribution, and stat generation belong in higher layers defined in `LAYERED_FEATURE_SCHEMA.md`.

## 2. Model Architecture
- **Layer 1: Linear Projection:** Maps 72 features to a higher-dimensional embedding space (e.g., 256).
- **Layer 2: Positional Encoding:** Injects temporal sequence information.
- **Layer 3: Transformer Encoder:** 2-4 heads of Self-Attention.
- **Layer 4: Global Pooling:** Aggregates temporal information into a single feature vector.
- **Layer 5: MLP Classifier:** Softmax output over the Action Taxonomy labels.

## 3. Output Contract
- **Raw Output:** Class logits over the action taxonomy.
- **Typical Post-Processing:** `argmax` or calibrated probabilities.
- **Not Included:** Possession state, official events, score changes, or box-score stats.

The Action Brain output must be combined with ball control, player tracks, court context, and ledger state before the system can infer basketball events.

## 4. Training Protocol
- **Source:** `data/training/oracle_dataset_v3.jsonl` when present, otherwise the older synthetic dataset path.
- **Loss Function:** Cross-Entropy.
- **Goal:** Learn stable motion semantics from short pose-plus-ball sequences while keeping the `features_v2` contract fixed.

## 5. Boundaries
- **Inside the model:** local temporal motion cues, pose-ball relationships, coarse court position.
- **Outside the model:** possession origin, dribble chains, pass chains, defender context, event attribution, and stat accumulation.

This separation keeps the neural contract stable while allowing richer basketball reasoning to evolve above it.

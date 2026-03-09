# Action Brain: Temporal Transformer Design

The Action Brain is the learned intelligence layer responsible for classifying basketball moves from skeletal data.

## 1. Input Specification (The Multimodal Feature Tensor)
Instead of raw pixels, we feed a 3D tensor of shape `(Batch, 30, D)`.
- **Primary Features (34 dims):** 17 player keypoints (x, y) normalized to bounding box.
- **Kinematic Features (34 dims):** First-order velocity (Δx, Δy) for all joints.
- **Game Context (4 dims):**
    - `ball_dist_wrist_l`, `ball_dist_wrist_r`: Distance to the ball (cm).
    - `court_x`, `court_y`: Global position on the court (cm).
- **Total Dimension (D):** 72.

## 2. Model Architecture
- **Layer 1: Linear Projection:** Maps 72 features to a higher-dimensional embedding space (e.g., 256).
- **Layer 2: Positional Encoding:** Injects temporal sequence information.
- **Layer 3: Transformer Encoder:** 2-4 heads of Self-Attention.
- **Layer 4: Global Pooling:** Aggregates temporal information into a single feature vector.
- **Layer 5: MLP Classifier:** Softmax output over the Action Taxonomy labels.

## 3. Training Protocol
- **Source:** `synthetic_dataset.jsonl` (Chapter 10).
- **Loss Function:** Cross-Entropy.
- **Goal:** Minimize the "Sim-to-Real" gap by training on jittered synthetic data.

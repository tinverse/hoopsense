# Action Brain: Kinematic Lifting (2D-to-3D Reconstruction)

Kinematic Lifting is the process of estimating the 3D joint coordinates of a player from a single-camera 2D skeletal stream.

## 1. The Challenge: Depth Ambiguity
From a single camera, we lose the "Z" dimension (depth). A player's arm extending toward the camera looks similar to an arm moving laterally.

## 2. The Solution: Geometric & Temporal Constraints
We solve depth ambiguity using three anchors:

1.  **Anthropometric Constants:** Human bones (e.g., femur, humerus) do not change length. If an arm looks shorter in 2D, it MUST be rotating in 3D space.
2.  **Floor Anchoring:** Using the **Spatial Resolver (Chapter 3)**, we know the (X, Y) ground plane coordinate of the feet. This provides the 3D base for the rest of the skeleton.
3.  **Temporal Consistency:** Joint movement must follow smooth velocity vectors. We use a **Kalman Filter** or a **Lifting Transformer** to predict the most likely 3D pose that fits the 2D observation.

## 3. Implementation Path
- **Stage 1:** Simple Geometric Lifting (using known bone lengths).
- **Stage 2:** Learned Lifting (training a model to predict 3D from 2D using our Synthetic Oracle as ground truth).

## 4. Output: The Avatar Rig
The result is a stream of (X, Y, Z) coordinates ready for mapping to standard humanoid bones in `.gltf` format.

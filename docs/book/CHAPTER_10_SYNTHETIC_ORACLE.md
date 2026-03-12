# Chapter 10: The Synthetic Oracle – Training without Labels

The greatest bottleneck in AI development is human labeling. To build HoopSense without a labeling team, we use **Synthetic Data Generation**. Instead of asking humans to watch video, we ask computers to generate it.

## 1. The 3D-to-2D Projection Pipeline

The "Synthetic Oracle" works by running our **Spatial Resolver (Chapter 3)** in reverse.

1.  **The 3D MoCap (Ground Truth):** We start with high-fidelity 3D motion capture data of a basketball player performing a specific move (e.g., a "Step-back"). This data has perfect (X, Y, Z) coordinates for every joint.
2.  **The Virtual Camera:** We place a virtual camera in a 3D gym at various heights and angles (simulating a phone, a HoopBox, or a drone).
3.  **The Projection:** We use Projective Geometry to flatten the 3D skeleton onto the 2D camera plane.
4.  **Auto-Labeling:** Because the computer knows exactly which frames constitute the "Step-back," it writes the label automatically.

## 2. Why Skeletal Training is "Label-Free"

Training on raw pixels (RGB) is hard because of lighting, jersey colors, and blur. However, training on **Skeletal Keypoints** is mathematically clean.

By generating skeletal sequences for "Crossovers" vs. "Normal Dribbles," our **Temporal Transformer** can learn the "Physics of the Move" without ever needing a human to draw a box.

This Oracle feeds the local motion layer. It does not directly generate official events or player stats. Those require the higher-layer possession and attribution logic described in `LAYERED_FEATURE_SCHEMA.md`.

## 3. The Move Taxonomy (Kinematic Signatures)

To generate a complete dataset, we define "Hard" kinematic signatures for every major basketball event.

| Move | Kinematic Signature | AI Goal |
| :--- | :--- | :--- |
| **Jump Shot** | Rising hips + Wrists flickering above head apex. | Spot high-value scoring events. |
| **Crossover** | Lateral hip sway + Rapid wrist-Z oscillation. | Identify ball-handling elite moves. |
| **Rebound** | Maximum vertical extension + Downward wrist snap. | Provide local motion evidence for later possession attribution. |
| **Block** | Lateral jump + High-velocity hand swat at apex. | Measure defensive impact. |
| **Steal** | Forward lunging torso + Low-Z hand reach. | Identify defensive IQ / turnovers. |

## 4. Synthetic Noise & The Completeness Gate

To ensure the AI works in a real, messy gym, we inject **Synthetic Noise** into the generator:
-   **Gaussian Jitter:** Simulates the ±5 pixel "shake" of real pose estimation.
-   **Self-Healing Calibration:** Every synthetic batch is passed through a **Completeness Gate** that verifies if the actor remains in the camera's "Visible Volume."

## 5. The Loop: Synthetic -> Ref-API -> Real

1.  **Synthetic:** Train the baseline Action Brain on Oracle-generated motion windows.
2.  **Ref-API:** Deploy the model in a real gym. Use the human Referee's signals (Chapter 7) to confirm if the model's guess was right.
3.  **Feedback:** Use the Referee-validated "Real" data to fine-tune the model, closing the gap between simulation and reality.
4.  **Attribution:** Combine the model output with possession context and geometry to generate explainable basketball events and stats.

---

**Summary for the Engineer:**
-   **3D MoCap** provides the "Physics."
-   **Projective Geometry** provides the "Perspective."
-   **Auto-labeling** provides the "Scale."
-   **Possession and stats layers** provide the basketball meaning above local motion labels.

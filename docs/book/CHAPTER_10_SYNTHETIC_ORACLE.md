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

By generating millions of skeletal sequences for "Crossovers" vs. "Normal Dribbles," our **Temporal-Transformer (Chapter 6)** can learn the "Physics of the Move" without ever needing a human to draw a box.

## 3. Bridging the "Sim-to-Real" Gap

To ensure the AI works in a real, messy gym, we inject **Synthetic Noise** into the generator:
-   **Jitter:** Simulate a shaky hand-held phone.
-   **Occlusion:** Randomly "hide" keypoints (like an ankle or wrist) to teach the model how to infer missing data.
-   **Projection Error:** Artificially introduce lens distortion to test the robustness of the **Spatial Resolver**.

## 4. The Loop: Synthetic -> Ref-API -> Real

1.  **Synthetic:** Train the baseline model on millions of computer-generated moves.
2.  **Ref-API:** Deploy the model in a real gym. Use the human Referee's signals (Chapter 7) to confirm if the model's guess was right.
3.  **Feedback:** Use the Referee-validated "Real" data to fine-tune the model, closing the gap between simulation and reality.

---

**Summary for the Engineer:**
-   **3D MoCap** provides the "Physics."
-   **Projective Geometry** provides the "Perspective."
-   **Auto-labeling** provides the "Scale."

# Chapter 6: Action Recognition – From Pose to Performance

Basketball is a game of high-speed, multi-agent dynamics. In this chapter, we transition from tracking "where" things are (Chapters 1-3) to understanding **what** is happening. This is the difference between a set of moving points and a "Step-back Three."

## 1. The Semantic Layer: Intent vs. Motion

An "Action" in HoopSense is more than just a skeletal sequence; it is a **physically contextualized intent**. 

### The Context Gap
If we only look at a normalized skeleton, a "Jump" looks identical whether it is a Jump Shot, a Rebound, or a Block. To differentiate, we must feed the model a **Multimodal Feature Tensor** that includes:

1.  **Kinematics (The Motion):** 17 joints + their relative velocities.
2.  **Ball Context (The Object):** Proximity of the ball to the hands. (A jump with the ball at the wrist is likely a shot).
3.  **Global Geometry (The Map):** The player's 3D position on the court. (A jump at the baseline is a corner three; a jump at the rim is a dunk).

## 2. The Input Tensor Specification (D=72)
Our Temporal-Transformer encodes a 1-second window (30 frames) of 72 features:
- **34 Features:** Normalized (x, y) coordinates of 17 joints.
- **34 Features:** Instantaneous velocity (Δx, Δy) of those joints.
- **4 Features:** Global court (x, y) and Ball-relative distance.

### Level 2: Local Action Recognition (The "Move")
We feed a 32-frame window (approx. 1 second) into a **Temporal-Transformer**.
- **Transformer Encoder:** Unlike LSTMs, which process data sequentially, Transformers use **Self-Attention** to look at the entire 1-second clip at once.
- **Attention Map:** The model learns to "attend" to the feet during a travel check, and the hands during a shot check.

### Level 3: Global Game Context (The "Story")
We combine player actions with ball trajectory and official signals.
- **Logic:** If `Action(Player_01) == Shot` AND `Ball_Path == Parabolic_Arc` AND `Ref_Signal == 3pt_Success`, then `Score = 3`.

## 3. The Action Taxonomy (The Dictionary)

Our model is trained to recognize three categories of events:

| Category | Event Examples | Visual Moat |
| :--- | :--- | :--- |
| **Elite Moves** | Step-back, Euro-step, Crossover | Requires high-frequency skeletal tracking. |
| **Violations** | Traveling, Double-dribble | Requires foot-to-ball synchronization. |
| **Official signals** | 3-pt, Foul, Technical | Requires dedicated secondary CNN for the Ref. |

## 4. Training with Synthetic Data

Because real-world amateur footage is noisy and "Step-backs" are relatively rare, we supplement our training data with **Synthetic Motion Capture**.
- We use 3D skeletal data from professional motion capture (like the CMU Panoptic Dataset) and project it into our 2D camera perspectives.
- This allows us to train the AI on "perfect" basketball moves before asking it to recognize them in a blurry high school gym.

---

**Summary for the Engineer:**
- **Action Recognition** is a classification problem over a **temporal window**.
- **Self-Attention** is the key to identifying the defining frame of a move (e.g., the "flick" of the wrist).
- **Rules** define the labels, but **Skeletal Velocity** defines the features.

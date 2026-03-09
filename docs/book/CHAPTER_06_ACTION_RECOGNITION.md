# Chapter 6: Action Recognition – From Pose to Performance

Basketball is a game of high-speed, multi-agent dynamics. In this chapter, we transition from tracking "where" things are (Chapters 1-3) to understanding **what** is happening. This is the difference between a set of moving points and a "Step-back Three."

## 1. The Semantic Layer: What is an Action?

An "Action" in HoopSense is a temporal sequence of skeletal poses that matches a known basketball move or an official signal. Unlike object detection, which works on a single frame, Action Recognition requires **context over time**.

### The Challenge: Temporal Vagueness
- **Where does a shot start?** Is it when the knees bend? When the ball reaches the forehead?
- **Intent vs. Accident:** A player losing their balance might look like a "Euro-step" to a simple classifier.

To solve this, we use a **Hierarchical Action Stack**.

## 2. The Hierarchical Action Stack

### Level 1: Atomic Keypoint Velocities (The "Physics")
We calculate the 1st and 2nd derivatives of our 17 keypoints.
- **Example:** High upward velocity of the `Wrist` keypoints + rising `Center of Mass` = Potential Shot Attempt.

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

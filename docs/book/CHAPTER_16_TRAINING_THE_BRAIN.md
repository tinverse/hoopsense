# Chapter 16: Training the Temporal Brain – Self-Attention & Sim-to-Real

We treat local basketball motion understanding as a high-dimensional sequence classification problem. This chapter details the **Temporal Transformer** architecture used by the Action Brain to classify short motion windows from the multimodal kinematic manifold.

This chapter is intentionally narrow. The Action Brain predicts local motion classes. Possession context, event attribution, and stat generation are separate higher layers.

## 1. The Multi-Head Self-Attention Mechanism

The **Action Brain** utilizes a Transformer Encoder to capture long-range temporal dependencies. Unlike Recurrent Neural Networks (RNNs), which suffer from vanishing gradients, Transformers use **Self-Attention** to weigh the importance of every frame simultaneously.

### The QKV Transform
For a sequence of embeddings $\mathbf{X}$, we compute Queries ($\mathbf{Q}$), Keys ($\mathbf{K}$), and Values ($\mathbf{V}$):
$$ \mathbf{Q} = \mathbf{XW}^Q, \quad \mathbf{K} = \mathbf{XW}^K, \quad \mathbf{V} = \mathbf{XW}^V $$
The attention weights are calculated using the **Scaled Dot-Product Attention**:
$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \mathbf{V} $$
This allows the model to "attend" to the frame where the ball leaves the hand (the release) while suppressing the "noise" of the preceding dribble.

## 2. Positional Encoding & Temporal Topology

Transformers are **Permutation Invariant** (they don't inherently know the order of frames). To preserve the temporal topology of a basketball move, we inject **Sinusoidal Positional Encodings**:
$$ PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}}) $$
This ensures the model can distinguish between a "Jump Shot" (Up then Down) and a "Rebound Landing" (Down then Up).

## 3. Sim-to-Real: Bridging the Domain Gap

The primary challenge in our "Double Hybrid" strategy is the **Domain Shift** between synthetic math-skeletons and real-world pose estimation results.

### Mitigation Strategies:
1.  **Domain Randomization:** During synthetic generation, we inject Gaussian noise $\mathcal{N}(0, \sigma^2)$ and apply **Keypoint Dropout** (simulating occlusions).
2.  **Multimodal Consistency:** By including the **Ball-to-Wrist Distance** (grounded in cm), we provide a feature that is robust to 2D pixel noise. The "Physics of the Ball" acts as a regularizer for the "Noise of the Pose."
3.  **Label Smoothing:** We use Soft-Labels to account for the temporal vagueness of when an action truly "starts" or "ends."

## 4. What This Model Does Not Do
- It does not decide who started the possession.
- It does not decide whether a pass became an assist.
- It does not maintain a game ledger.
- It does not directly output box-score statistics.

Those responsibilities belong to the layers above the Action Brain.

---

**Research Summary:**
- **Attention:** Replaces manual heuristics with learned relevance weights.
- **Topology:** Preserved via harmonic positional signals.
- **Robustness:** Achieved through adversarial noise injection in the Synthetic Oracle.
- **System boundary:** The model is a local classifier inside a larger reasoning stack.

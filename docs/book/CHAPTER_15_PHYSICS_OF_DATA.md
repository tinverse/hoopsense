# Chapter 15: The Projective Manifold – Geometric Grounding & Procedural MoCap

To achieve deterministic understanding from video, we must map the 2D image plane $\mathbb{P}^2$ to the 3D Euclidean world $\mathbb{R}^3$. This chapter explores the high-order mathematics of our **Synthetic Oracle**.

## 1. Procedural Kinematics in State-Space

We model basketball movement as a **Kinematic State-Space Model**. Every joint $j$ at time $t$ is defined by a state vector $\mathbf{s}_{j,t} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^T$. 

### Parametric Motion Generation
Instead of keyframing, we use **Basis Functions** to ensure $C^2$ continuity (smooth acceleration). A "Jump Shot" is modeled as a transformation of the center-of-mass (CoM) through a gravitational parabolic arc:
$$ z_{com}(t) = z_0 + v_0 t - \frac{1}{2} g t^2 $$
Biomechanical constraints are enforced via **Euler Angles** and fixed link-lengths (Anthropometric Invariance), ensuring that the generated skeletons represent valid human topology.

## 2. Projective Geometry & The Direct Linear Transform (DLT)

Mapping pixels $(u, v)$ to court coordinates $(X, Y)$ is a problem of finding the optimal **Homography Matrix** $\mathbf{H} \in PGL(3, \mathbb{R})$.

### The Homogeneous Transform
Using homogeneous coordinates, we solve:
$$ \begin{bmatrix} x \\ y \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} $$
To find $\mathbf{H}$, we employ the **DLT Algorithm**. Given $n \ge 4$ point correspondences, we construct a matrix $\mathbf{A} \in \mathbb{R}^{2n \times 9}$. The solution $\mathbf{h}$ (the flattened $\mathbf{H}$) is the **unit singular vector** corresponding to the smallest singular value of $\mathbf{A}$, found via **Singular Value Decomposition (SVD)**:
$$ \mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T $$
$\mathbf{h}$ is the last column of $\mathbf{V}$. This minimizes the algebraic error $\| \mathbf{Ah} \|$ under the constraint $\| \mathbf{h} \| = 1$.

## 3. The 72-Dimensional Multimodal Tensor

The input to our learned model is a structured embedding $\mathcal{X} \in \mathbb{R}^{T \times D}$ where $D=72$.

### Feature Decomposition:
1.  **Pose Embedding ($\mathbb{R}^{34}$):** $L_2$-normalized coordinates of the 17-point skeleton. Normalization relative to the bounding box $\mathbf{B}$ ensures **Scale Invariance**.
2.  **Temporal Derivative ($\mathbb{R}^{34}$):** The first-order finite difference $\Delta \mathbf{P}_t = \mathbf{P}_t - \mathbf{P}_{t-1}$. This captures the **Momentum Manifold** of the move.
3.  **Interaction Scalar ($\mathbb{R}^{2}$):** The Euclidean distance between the ball $\mathbf{b}$ and the wrists $\mathbf{w}_{l,r}$. This provides the **Semantic Context** necessary to distinguish a shot from a rebound.
4.  **Global Anchor ($\mathbb{R}^{2}$):** The absolute $(X, Y)$ coordinate on the court, providing **Spatial Prior** (e.g., proximity to the 3-point arc).

---

**Research Summary:**
- **Invariance:** We achieve scale and distance invariance through local normalization.
- **Grounding:** We maintain physical reality through global centimeter-scale features.
- **Optimization:** We solve for the projection manifold using SVD to ensure numerical stability.

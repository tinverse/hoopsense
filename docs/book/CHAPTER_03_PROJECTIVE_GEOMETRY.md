# Chapter 3: Projective Geometry – The Math of the Court

To turn a flat 2D video into professional 3D statistics, HoopSense must understand the geometric relationship between the camera lens and the basketball court. This chapter explains the "Geometric Kernel" written in Rust.

## 1. The Homography Matrix (The Map)

A basketball court is a flat plane. In projective geometry, we can map any point on one plane (the video frame) to another plane (the court floor) using a **3x3 Homography Matrix (H)**.

### The Equation:
`[x', y', w]^T = H * [u, v, 1]^T`
- `(u, v)` are the pixel coordinates.
- `(x', y')` are the court coordinates (after dividing by `w`).

### Calibration via DLT:
We use the **Direct Linear Transform (DLT)** algorithm to solve for `H`. By identifying 4 known points (anchors) on the court, such as the corners of the free-throw line, we create a system of equations that defines the entire court's spatial relationship to the camera.

## 2. Lens Undistortion (The "Flattening")

Most cameras (especially phones and GoPros) have "Barrel Distortion," where straight lines appear curved. If we don't fix this, a shot from the corner will appear to be out-of-bounds.

We implement the **Radial-Tangential Model**:
- **Radial (k1, k2):** Fixes the "fisheye" effect.
- **Tangential (p1, p2):** Fixes misalignment between the lens and the sensor.

The Rust core iteratively "un-warps" every pixel before projecting it onto the court floor.

## 3. PnP: Perspective-n-Point (The "Inner Ear")

How does the system know the camera is 15 feet high? It uses the **PnP Solver**.
By comparing the 2D position of the hoop in the video to its known 3D height (10 feet), the system calculates the camera's exact **Rotation (R)** and **Translation (t)**.

## 4. Dynamic SLAM-lite (The "Whip-Pan")

In a real game, the camera moves. We use a **Stateful Motion Model** to track the "Delta H" ($\Delta H$) between frames. 
- If the camera pans right, the system calculates the pixel shift and updates the global Homography matrix smoothly.
- This ensures that stats remain accurate even during high-velocity transitions.

---

**Summary for the Engineer:**
- **Geometry** is our ground truth.
- **Rust** provides the linear algebra (via `nalgebra`) needed for sub-millisecond coordinate projection.
- **Calibration** turns a "picture" of a game into a "measurement" of a game.

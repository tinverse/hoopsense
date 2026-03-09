use nalgebra::{Matrix3, Vector3, Point2};
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};

/// Current state of the resolver, used for temporal smoothing and 
/// dynamic camera tracking (SLAM-lite).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialState {
    pub last_update_t: u64,
    pub confidence: f32,
    pub is_panning: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialResolver {
    /// 3x3 Homography Matrix (H)
    /// Used for mapping: [x', y', w]^T = H * [u, v, 1]^T
    pub h_matrix: Matrix3<f32>,
    
    /// Camera Intrinsic Matrix (K)
    /// [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    pub k_matrix: Matrix3<f32>,

    /// Distortion Coefficients (k1, k2, p1, p2, k3)
    /// Standard OpenCV Radial-Tangential model
    pub distortion_coeffs: Vec<f32>,

    /// Court dimensions in centimeters (e.g., 2800x1500 for FIBA)
    pub court_width: f32,
    pub court_height: f32,

    /// State information for temporal continuity
    pub state: SpatialState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CourtPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl SpatialResolver {
    /// Creates a new resolver with identity matrices and zero distortion.
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            h_matrix: Matrix3::identity(),
            k_matrix: Matrix3::identity(),
            distortion_coeffs: vec![0.0; 5],
            court_width: width,
            court_height: height,
            state: SpatialState {
                last_update_t: 0,
                confidence: 0.0,
                is_panning: false,
            },
        }
    }

    /// Updates the resolver's spatial state based on camera motion deltas.
    /// This is the core of the dynamic SLAM logic.
    /// delta_h: The relative movement of background features between frames.
    pub fn update_from_motion(&mut self, delta_h: Matrix3<f32>, t_ms: u64) {
        // Update the global homography: H_new = H_old * delta_H
        self.h_matrix = self.h_matrix * delta_h;
        self.state.last_update_t = t_ms;
        
        // Detect if we are in a high-velocity pan (heuristic)
        let motion_intensity = (delta_h - Matrix3::identity()).norm();
        self.state.is_panning = motion_intensity > 0.05;
        
        // Decay confidence slightly during pans (until re-anchored)
        if self.state.is_panning {
            self.state.confidence *= 0.99;
        }
    }

    /// Sets the camera intrinsics and distortion coefficients.
    pub fn set_intrinsics(&mut self, fx: f32, fy: f32, cx: f32, cy: f32, coeffs: Vec<f32>) {
        self.k_matrix = Matrix3::new(
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        );
        self.distortion_coeffs = coeffs;
    }

    /// Undistorts a 2D pixel point using the stored intrinsic and distortion parameters.
    /// Iterative method (Newton-Raphson) to invert the distortion model.
    pub fn undistort_point(&self, u: f32, v: f32) -> Point2<f32> {
        if self.distortion_coeffs.iter().all(|&c| c == 0.0) {
            return Point2::new(u, v);
        }

        let fx = self.k_matrix[(0, 0)];
        let fy = self.k_matrix[(1, 1)];
        let cx = self.k_matrix[(0, 2)];
        let cy = self.k_matrix[(1, 2)];
        
        let k1 = self.distortion_coeffs[0];
        let k2 = self.distortion_coeffs[1];
        let p1 = self.distortion_coeffs[2];
        let p2 = self.distortion_coeffs[3];
        let k3 = self.distortion_coeffs.get(4).cloned().unwrap_or(0.0);

        // Normalize the pixel coordinate
        let x0 = (u - cx) / fx;
        let y0 = (v - cy) / fy;
        
        let mut x = x0;
        let mut y = y0;

        // Iteratively undistort (usually converges in 5-10 iterations)
        for _ in 0..5 {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            
            let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            
            let x_dist = x * radial + dx;
            let y_dist = y * radial + dy;
            
            // Error = Calculated_Distorted - Observed_Distorted
            // We want to subtract this error from our current guess
            x = x + (x0 - x_dist);
            y = y + (y0 - y_dist);
        }

        // Project back to pixel coordinates
        Point2::new(x * fx + cx, y * fy + cy)
    }

    /// Transforms a pixel coordinate (u, v) to a 3D court coordinate (X, Y, 0).
    /// Now includes an undistortion step before applying homography.
    pub fn resolve_floor_point(&self, u: f32, v: f32) -> Result<CourtPoint> {
        // 1. Undistort the point first
        let corrected = self.undistort_point(u, v);
        
        // 2. Apply Homography
        let pixel_vec = Vector3::new(corrected.x, corrected.y, 1.0);
        let world_vec = self.h_matrix * pixel_vec;
        
        if world_vec.z.abs() < 1e-6 {
            return Err(anyhow!("Singular point in homography transformation (w=0)"));
        }
        
        // Normalize by the homogeneous coordinate (w)
        Ok(CourtPoint {
            x: world_vec.x / world_vec.z,
            y: world_vec.y / world_vec.z,
            z: 0.0,
        })
    }

    /// Extracts the Camera Extrinsics (Rotation and Translation) from the 
    /// Homography matrix and Camera Intrinsics.
    /// 
    /// This is a planar PnP solution. It returns the rotation matrix R and 
    /// the translation vector t such that: x_pixel = K * [R|t] * X_world
    pub fn solve_extrinsics(&self) -> Result<(nalgebra::Rotation3<f32>, Vector3<f32>)> {
        let k_inv = self.k_matrix.try_inverse().ok_or_else(|| anyhow!("Intrinsics not invertible"))?;
        let mut a = k_inv * self.h_matrix;

        // Columns of the extrinsics matrix [r1, r2, t] are proportional to K^-1 * H
        let s = 1.0 / a.column(0).norm();
        a *= s; // Scale the matrix

        let r1 = a.column(0).into_owned();
        let r2 = a.column(1).into_owned();
        let t = a.column(2).into_owned();
        let r3 = r1.cross(&r2);

        // Form the rotation matrix R = [r1, r2, r3]
        let r_approx = nalgebra::Matrix3::from_columns(&[r1, r2, r3]);
        let svd = r_approx.svd(true, true);
        let u = svd.u.ok_or_else(|| anyhow!("SVD U failed"))?;
        let v_t = svd.v_t.ok_or_else(|| anyhow!("SVD V_t failed"))?;
        let r_final = u * v_t;

        Ok((nalgebra::Rotation3::from_matrix(&r_final), t.into()))
    }

    /// Updates the Homography matrix using a set of known 2D-3D point correspondences
    /// using the Direct Linear Transform (DLT) algorithm.
    pub fn calibrate(&mut self, anchors: Vec<(Point2<f32>, Point2<f32>)>) -> Result<()> {
        if anchors.len() < 4 {
            return Err(anyhow!("At least 4 points are required for calibration"));
        }

        // The DLT system: Ah = 0 where h is the flattened H matrix (9 parameters).
        // Each point correspondence gives 2 equations.
        let mut a = nalgebra::DMatrix::<f32>::zeros(anchors.len() * 2, 9);

        for (i, (pixel, world)) in anchors.iter().enumerate() {
            let u = pixel.x;
            let v = pixel.y;
            let x = world.x;
            let y = world.y;

            // Equation 1: -x_w * w_p + u_p * w_p = 0 -> in terms of h
            a[(i * 2, 0)] = -u;
            a[(i * 2, 1)] = -v;
            a[(i * 2, 2)] = -1.0;
            a[(i * 2, 6)] = x * u;
            a[(i * 2, 7)] = x * v;
            a[(i * 2, 8)] = x;

            // Equation 2: -y_w * w_p + v_p * w_p = 0
            a[(i * 2 + 1, 3)] = -u;
            a[(i * 2 + 1, 4)] = -v;
            a[(i * 2 + 1, 5)] = -1.0;
            a[(i * 2 + 1, 6)] = y * u;
            a[(i * 2 + 1, 7)] = y * v;
            a[(i * 2 + 1, 8)] = y;
        }

        // Solve using SVD: h is the singular vector corresponding to the smallest singular value.
        let svd = a.svd(true, true);
        let v_t = svd.v_t.ok_or_else(|| anyhow!("SVD computation failed (V^T is None)"))?;
        
        // The last row of V^T (the last column of V) is our solution h
        let h_vec = v_t.row(8).transpose();
        
        // Reshape h into the 3x3 matrix H
        self.h_matrix = Matrix3::new(
            h_vec[0], h_vec[1], h_vec[2],
            h_vec[3], h_vec[4], h_vec[5],
            h_vec[6], h_vec[7], h_vec[8]
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration() {
        let mut resolver = SpatialResolver::new(2800.0, 1500.0);
        
        // Define 4 points on the image (pixels) and their 3D court counterparts
        // (u, v) -> (x, y)
        let anchors = vec![
            (Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
            (Point2::new(100.0, 0.0), Point2::new(1000.0, 0.0)),
            (Point2::new(0.0, 100.0), Point2::new(0.0, 1000.0)),
            (Point2::new(100.0, 100.0), Point2::new(1000.0, 1000.0)),
            (Point2::new(50.0, 0.0), Point2::new(500.0, 0.0)),
        ];
        
        resolver.calibrate(anchors).unwrap();
        
        println!("H Matrix: {:?}", resolver.h_matrix);
        
        // Test a point in the middle
        let point = resolver.resolve_floor_point(50.0, 50.0).unwrap();
        println!("Resolved point (50, 50) -> {:?}", point);
        
        // Note: In production, input normalization (Centering and scaling) 
        // is crucial for numerical stability of the DLT algorithm.
        assert!((point.x - 500.0).abs() < 1e-2);
        assert!((point.y - 500.0).abs() < 1e-2);
    }
    #[test]
    fn test_undistortion() {
        let mut resolver = SpatialResolver::new(1920.0, 1080.0);
        
        // Define intrinsics (fx=1000, fy=1000, cx=960, cy=540)
        // and distortion (k1=0.1, others=0) -> Mild barrel/pincushion
        resolver.set_intrinsics(1000.0, 1000.0, 960.0, 540.0, vec![0.1, 0.0, 0.0, 0.0, 0.0]);
        
        // Let's take a point at (1060, 540) -> x_normalized = (1060-960)/1000 = 0.1, y=0
        // r^2 = 0.01
        // radial = 1 + 0.1*0.01 = 1.001
        // x_distorted_norm = 0.1 * 1.001 = 0.1001
        // u_distorted = 0.1001 * 1000 + 960 = 1060.1
        
        // The resolver.undistort_point takes the *distorted* pixel (1060.1, 540)
        // and should return close to the *undistorted* original (1060, 540).
        let undistorted = resolver.undistort_point(1060.1, 540.0);
        
        println!("Undistorted: {:?}", undistorted);
        
        assert!((undistorted.x - 1060.0).abs() < 1e-2);
        assert!((undistorted.y - 540.0).abs() < 1e-2);
    }
}

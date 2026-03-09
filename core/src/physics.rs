use nalgebra::{Point3, Vector3};
use crate::rules::{NCAA_RIM_HEIGHT, NCAA_RIM_RADIUS};

/// Represents a ball's 3D flight path.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub points: Vec<(u64, Point3<f32>)>, // (t_ms, pos)
}

impl Trajectory {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn add_point(&mut self, t_ms: u64, pos: Point3<f32>) {
        self.points.push((t_ms, pos));
        // Keep only recent history for live fit (e.g., 2 seconds)
        if self.points.len() > 60 {
            self.points.remove(0);
        }
    }

    /// Fits a parabola to the Z-axis movement to find the apex.
    /// Returns (apex_z, t_at_apex)
    pub fn solve_apex(&self) -> Option<(f32, u64)> {
        if self.points.len() < 10 { return None; }
        
        // Simplified heuristic: Find max Z in the current buffer
        let max_pt = self.points.iter()
            .max_by(|a, b| a.1.z.partial_cmp(&b.1.z).unwrap())?;
            
        Some((max_pt.1.z, max_pt.0))
    }

    /// Checks if the ball is currently 'interacting' with the rim area.
    pub fn is_rim_intersect(&self, rim_pos: &Point3<f32>) -> bool {
        if let Some((_, last_pos)) = self.points.last() {
            let dist_xy = ((last_pos.x - rim_pos.x).powi(2) + (last_pos.y - rim_pos.y).powi(2)).sqrt();
            let dist_z = (last_pos.z - rim_pos.z).abs();
            
            // Ball is within rim radius and crossing the rim plane
            return dist_xy < NCAA_RIM_RADIUS && dist_z < 10.0;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rim_intersection() {
        let mut traj = Trajectory::new();
        let rim_pos = Point3::new(160.0, 762.0, 304.8);
        
        // Add a point inside the rim
        traj.add_point(1000, Point3::new(160.0, 762.0, 304.0));
        assert!(traj.is_rim_intersect(&rim_pos));
        
        // Add a point far away
        traj.add_point(1100, Point3::new(500.0, 500.0, 304.0));
        assert!(!traj.is_rim_intersect(&rim_pos));
    }
}

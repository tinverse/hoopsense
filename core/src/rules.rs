use nalgebra::{Point2, Point3};

/// Standard NCAA Court Dimensions (in centimeters)
/// Citations refer to NCAA Men's Basketball 2023-24 Rules.

/// Rule 1, Section 3: The court shall be 94 feet long and 50 feet wide.
pub const NCAA_COURT_WIDTH: f32 = 1524.0;  // 50 feet
pub const NCAA_COURT_LENGTH: f32 = 2865.0; // 94 feet

/// Rule 1, Section 15: The upper edge of the ring shall be 10 feet above the floor.
pub const NCAA_RIM_HEIGHT: f32 = 304.8;    // 10 feet

/// Rule 1, Section 14: The cord net shall be 18 inches in diameter at the ring.
pub const NCAA_RIM_RADIUS: f32 = 22.86;    // 9 inches (radius)

/// Rule 1, Section 7: The 3-point field-goal line is 22 feet 1.75 inches from the center of the basket.
pub const NCAA_3PT_RADIUS: f32 = 675.0;    // ~22.15 feet

/// Rule 1, Section 13: The center of the basket shall be 5 feet 3 inches from the baseline.
pub const NCAA_BASELINE_TO_RIM: f32 = 160.0; // 5.33 feet (rounded for metric)

pub struct GeometricReferee {
    pub rim_pos_left: Point3<f32>,
    pub rim_pos_right: Point3<f32>,
}

impl GeometricReferee {
    pub fn new() -> Self {
        Self {
            // Rims are centered on the baselines
            rim_pos_left: Point3::new(NCAA_BASELINE_TO_RIM, NCAA_COURT_WIDTH / 2.0, NCAA_RIM_HEIGHT),
            rim_pos_right: Point3::new(NCAA_COURT_LENGTH - NCAA_BASELINE_TO_RIM, NCAA_COURT_WIDTH / 2.0, NCAA_RIM_HEIGHT),
        }
    }

    /// Predicate: Is the ball currently passing through the rim?
    pub fn is_ball_in_rim(&self, ball_pos: &Point3<f32>) -> bool {
        let dist_left = (Point2::new(ball_pos.x, ball_pos.y) - Point2::new(self.rim_pos_left.x, self.rim_pos_left.y)).norm();
        let dist_right = (Point2::new(ball_pos.x, ball_pos.y) - Point2::new(self.rim_pos_right.x, self.rim_pos_right.y)).norm();
        
        let in_left = dist_left < NCAA_RIM_RADIUS && (ball_pos.z - NCAA_RIM_HEIGHT).abs() < 10.0;
        let in_right = dist_right < NCAA_RIM_RADIUS && (ball_pos.z - NCAA_RIM_HEIGHT).abs() < 10.0;
        
        in_left || in_right
    }

    /// Predicate: Is the coordinate out of bounds?
    pub fn is_out_of_bounds(&self, pos: &Point2<f32>) -> bool {
        pos.x < 0.0 || pos.x > NCAA_COURT_LENGTH || pos.y < 0.0 || pos.y > NCAA_COURT_WIDTH
    }

    /// Predicate: Was the shot a 3-pointer based on feet location?
    pub fn is_3pt_attempt(&self, feet_pos: &Point2<f32>, target_left: bool) -> bool {
        let rim = if target_left { self.rim_pos_left } else { self.rim_pos_right };
        let dist = (feet_pos - Point2::new(rim.x, rim.y)).norm();
        dist > NCAA_3PT_RADIUS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basket_detection() {
        let ref_bot = GeometricReferee::new();
        // Ball exactly at the left rim center and height
        let ball_in = Point3::new(NCAA_BASELINE_TO_RIM, NCAA_COURT_WIDTH / 2.0, NCAA_RIM_HEIGHT);
        assert!(ref_bot.is_ball_in_rim(&ball_in));
        
        // Ball high above the rim
        let ball_high = Point3::new(157.5, 762.0, 400.0);
        assert!(!ref_bot.is_ball_in_rim(&ball_high));
    }

    #[test]
    fn test_out_of_bounds() {
        let ref_bot = GeometricReferee::new();
        assert!(ref_bot.is_out_of_bounds(&Point2::new(-1.0, 10.0)));
        assert!(!ref_bot.is_out_of_bounds(&Point2::new(100.0, 100.0)));
    }
}

use nalgebra::{Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Standard NCAA Court Dimensions (in centimeters)
/// Citations refer to NCAA Men's Basketball 2023-24 Rules.

pub const NCAA_COURT_WIDTH: f32 = 1524.0;
pub const NCAA_COURT_LENGTH: f32 = 2865.0;
pub const NCAA_RIM_HEIGHT: f32 = 304.8;
pub const NCAA_RIM_RADIUS: f32 = 22.86;
pub const NCAA_3PT_RADIUS: f32 = 675.0;
pub const NCAA_BASELINE_TO_RIM: f32 = 160.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CourtZone {
    Paint,
    WingLeft,
    WingRight,
    CornerLeft,
    CornerRight,
    TopOfKey,
    Backcourt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PossessionOrigin {
    Inbound,
    Rebound,
    Steal,
    Turnover,
    StartOfPeriod,
}

pub struct GeometricReferee {
    pub rim_pos_left: Point3<f32>,
    pub rim_pos_right: Point3<f32>,
}

impl GeometricReferee {
    pub fn new() -> Self {
        Self {
            rim_pos_left: Point3::new(NCAA_BASELINE_TO_RIM, NCAA_COURT_WIDTH / 2.0, NCAA_RIM_HEIGHT),
            rim_pos_right: Point3::new(NCAA_COURT_LENGTH - NCAA_BASELINE_TO_RIM, NCAA_COURT_WIDTH / 2.0, NCAA_RIM_HEIGHT),
        }
    }

    pub fn is_ball_in_rim(&self, ball_pos: &Point3<f32>) -> bool {
        let dist_left = (Point2::new(ball_pos.x, ball_pos.y) - Point2::new(self.rim_pos_left.x, self.rim_pos_left.y)).norm();
        let dist_right = (Point2::new(ball_pos.x, ball_pos.y) - Point2::new(self.rim_pos_right.x, self.rim_pos_right.y)).norm();
        let in_left = dist_left < NCAA_RIM_RADIUS && (ball_pos.z - NCAA_RIM_HEIGHT).abs() < 10.0;
        let in_right = dist_right < NCAA_RIM_RADIUS && (ball_pos.z - NCAA_RIM_HEIGHT).abs() < 10.0;
        in_left || in_right
    }

    pub fn is_out_of_bounds(&self, pos: &Point2<f32>) -> bool {
        pos.x < 0.0 || pos.x > NCAA_COURT_LENGTH || pos.y < 0.0 || pos.y > NCAA_COURT_WIDTH
    }

    pub fn is_3pt_attempt(&self, feet_pos: &Point2<f32>, target_left: bool) -> bool {
        let rim = if target_left { self.rim_pos_left } else { self.rim_pos_right };
        let dist = (feet_pos - Point2::new(rim.x, rim.y)).norm();
        dist > NCAA_3PT_RADIUS
    }

    pub fn resolve_zone(&self, pos: &Point2<f32>, target_left: bool) -> CourtZone {
        let half_court = NCAA_COURT_LENGTH / 2.0;
        let is_backcourt = if target_left { pos.x > half_court } else { pos.x < half_court };
        if is_backcourt { return CourtZone::Backcourt; }

        let rim = if target_left { self.rim_pos_left } else { self.rim_pos_right };
        let dist_to_baseline = if target_left { pos.x } else { NCAA_COURT_LENGTH - pos.x };
        let dist_to_centerline = (pos.y - NCAA_COURT_WIDTH / 2.0).abs();
        if dist_to_baseline < 580.0 && dist_to_centerline < 245.0 { return CourtZone::Paint; }

        let dist_to_rim = (pos - Point2::new(rim.x, rim.y)).norm();
        let is_outside_3pt = dist_to_rim > NCAA_3PT_RADIUS;
        if is_outside_3pt {
            if dist_to_baseline < 300.0 {
                if pos.y < NCAA_COURT_WIDTH / 2.0 { return CourtZone::CornerLeft; }
                else { return CourtZone::CornerRight; }
            }
            if dist_to_centerline < 300.0 { return CourtZone::TopOfKey; }
            if pos.y < NCAA_COURT_WIDTH / 2.0 { return CourtZone::WingLeft; }
            else { return CourtZone::WingRight; }
        }
        if pos.y < NCAA_COURT_WIDTH / 2.0 { return CourtZone::WingLeft; }
        return CourtZone::WingRight;
    }

    pub fn is_blocking_path(&self, hand_pos: &Point3<f32>, ball_pos: &Point3<f32>) -> bool {
        let dist_xy = (Point2::new(hand_pos.x, hand_pos.y) - Point2::new(ball_pos.x, ball_pos.y)).norm();
        let z_overlap = hand_pos.z >= ball_pos.z - 10.0;
        dist_xy < 20.0 && z_overlap
    }

    pub fn is_shot_release(&self, hand_vel: &Vector3<f32>, ball_vel: &Vector3<f32>, z_height: f32) -> bool {
        let vel_delta = (ball_vel.z - hand_vel.z).abs();
        vel_delta > 100.0 && z_height > 150.0
    }

    pub fn is_rebound_opportunity(&self, ball_pos: &Point3<f32>) -> bool {
        (ball_pos.z - NCAA_RIM_HEIGHT).abs() < 50.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_resolution() {
        let ref_bot = GeometricReferee::new();
        let bc_pos = Point2::new(2000.0, 762.0);
        assert_eq!(ref_bot.resolve_zone(&bc_pos, true), CourtZone::Backcourt);
        let paint_pos = Point2::new(200.0, 762.0);
        assert_eq!(ref_bot.resolve_zone(&paint_pos, true), CourtZone::Paint);
        let top_pos = Point2::new(1000.0, 762.0);
        assert_eq!(ref_bot.resolve_zone(&top_pos, true), CourtZone::TopOfKey);
    }
}

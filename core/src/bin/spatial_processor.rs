use hoopsense_core::{
    ledger::GameEvent, GameStateLedger, GeometricReferee, SpatialResolver, Trajectory,
};
use nalgebra::{Point2, Point3};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{self, BufRead, Write};

fn main() -> anyhow::Result<()> {
    let mut resolver = SpatialResolver::new(1524.0, 2865.0);
    let referee = GeometricReferee::new();
    let mut ledger = GameStateLedger::new();
    let mut ball_trajectory = Trajectory::new();

    // Calibration: Set camera to same position as integration_probe.py
    resolver.set_intrinsics(1000.0, 1000.0, 960.0, 540.0, vec![0.0; 5]);
    resolver.h_matrix = nalgebra::Matrix3::identity();

    let input_path = "data/intelligent_game_dna.jsonl";
    let output_path = "data/validated_game_dna.jsonl";

    if !std::path::Path::new(input_path).exists() {
        return Ok(());
    }

    let file = File::open(input_path)?;
    let mut output = File::create(output_path)?;
    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let mut row: Value = serde_json::from_str(&line)?;
        let kind = row["kind"].as_str().unwrap_or("unknown").to_string();
        let t_ms = row["t_ms"].as_u64().unwrap_or(0);

        if let (Some(x), Some(y)) = (row["x"].as_f64(), row["y"].as_f64()) {
            let cp = resolver.resolve_floor_point(x as f32, y as f32)?;
            row["court_x"] = json!(cp.x);
            row["court_y"] = json!(cp.y);
            row["court_z"] = json!(cp.z);

            if kind == "ball" {
                let ball_height = referee.rim_pos_left.z;
                ball_trajectory.add_point(t_ms, Point3::new(cp.x, cp.y, ball_height));
                if ball_trajectory.is_rim_intersect(&referee.rim_pos_left)
                    || ball_trajectory.is_rim_intersect(&referee.rim_pos_right)
                {
                    ledger.propose_event(
                        GameEvent::MadeBasket {
                            player_id: 0,
                            team_id: 1, // Added missing team_id
                            points: 2,
                            t_ms,
                            is_official: false,
                        },
                        2000,
                    );
                }
            }
        }

        match kind.as_str() {
            "referee" => {
                if let Some("ref_3pt_success") = row["signal"].as_str() {
                    ledger.validate_event(t_ms, 3);
                }
            }
            _ => (),
        }

        let output_line = serde_json::to_string(&row)?;
        writeln!(output, "{}", output_line)?;
    }

    println!("[SCORE] Final Score: {:?}", ledger.official_score);
    Ok(())
}

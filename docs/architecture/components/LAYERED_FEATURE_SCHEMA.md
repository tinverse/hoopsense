# Layered Feature Schema

This document defines the feature layers required for trustworthy basketball reasoning and stat generation.

`FEATURE_SCHEMA_V2.md` remains the current neural contract for the Action Brain. This document defines the broader system feature graph around it.

## Design Rule

- Keep the Action Brain narrow and stable.
- Put possession, sequence, and stat logic in higher layers.
- Prefer derived semantic state over forcing every concern into one neural tensor.

## Layer 1: Perception Frame

Purpose:
Convert raw video into synchronized per-frame observations for players, ball, referee, and court geometry.

Primary owner:
- detectors
- trackers
- pose estimator
- spatial resolver

Canonical fields:
- `t_ms`
- `frame_idx`
- `track_id`
- `entity_type`
- `bbox_xywh`
- `keypoints_2d`
- `keypoints_3d`
- `court_x`
- `court_y`
- `ball_x`, `ball_y`, `ball_z`
- `ref_signal`
- `confidence_bps`

Outputs:
- frame-aligned trajectories
- pose streams
- ball trajectory
- court-relative player locations

## Layer 2: Action Brain Input

Purpose:
Represent short-horizon body motion and ball interaction for local action classification.

Primary owner:
- `ActionBrain`

Current contract:
- `FEATURE_SCHEMA_V2.md`
- shape `(Batch, 30, 72)`

Canonical fields:
- `pose_xy[17]`
- `pose_dxy[17]`
- `ball_dist_left_wrist`
- `ball_dist_right_wrist`
- `court_x`
- `court_y`

Near-term extension candidates:
- `ball_height`
- `ball_speed`
- `ball_release_separation`
- `nearest_defender_dist`

Non-goals:
- possession attribution
- stat generation
- rules reasoning

## Layer 3: Possession Context

Purpose:
Track who controls the ball, how the possession started, and what movement pattern the offense is currently in.

Primary owner:
- possession context engine
- game-state ledger

Canonical fields:
- `possession_id`
- `team_in_possession`
- `possession_origin`
- `ballhandler_id`
- `touch_start_t_ms`
- `dribble_count`
- `pass_count`
- `paint_touch_flag`
- `transition_flag`
- `offense_zone`
- `defender_pressure`
- `recent_action_labels`

Derived semantics:
- live-ball vs dead-ball
- inbound vs rebound vs steal vs turnover start
- drive started
- perimeter swing
- kick-out candidate

## Layer 4: Event Attribution

Purpose:
Transform trajectories, local action predictions, and possession state into concrete basketball events.

Primary owner:
- event inference
- rules engine
- ledger

Canonical fields:
- `event_id`
- `event_type`
- `actor_id`
- `secondary_actor_id`
- `start_t_ms`
- `end_t_ms`
- `event_confidence_bps`
- `source_features`
- `ball_control_state`
- `court_zone`
- `action_label`
- `action_confidence_bps`

Typical events:
- `pass`
- `catch`
- `dribble`
- `pickup`
- `shot_attempt`
- `ball_released`
- `rebound`
- `block`
- `steal`
- `turnover`

## Layer 5: Stat Generation

Purpose:
Accumulate event-level truth into box score, shot chart, possession summaries, and derived player metrics.

Primary owner:
- stat ledger
- report generation

Canonical fields:
- `game_id`
- `player_id`
- `team_id`
- `possessions_used`
- `fga`
- `fgm`
- `three_pa`
- `three_pm`
- `assists`
- `rebounds_off`
- `rebounds_def`
- `steals`
- `blocks`
- `turnovers`
- `shot_locations`

Derived products:
- `events.jsonl`
- `stats.csv`
- `stats.json`
- `report.md`

## MVP Implementation Order

1. Keep `FEATURE_SCHEMA_V2` as the current Action Brain contract.
2. Add a `PossessionContext` object in the ledger.
3. Track `possession_origin`, `ballhandler_id`, `dribble_count`, `pass_count`, and coarse `offense_zone`.
4. Define event attribution rules using Action Brain output plus possession context.
5. Generate stats from attributed events, not directly from neural logits.

## Ownership Summary

- Perception answers: `what is where`
- Action Brain answers: `what motion is this`
- Possession Context answers: `what phase of basketball is happening`
- Event Attribution answers: `what event occurred`
- Stat Generation answers: `what should be counted`

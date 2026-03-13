use crate::rules::{CourtZone, GeometricReferee, PossessionOrigin};
use nalgebra::Point2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Represents a validated basketball event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GameEvent {
    MadeBasket {
        player_id: u32,
        team_id: u8,
        points: u8,
        t_ms: u64,
        is_official: bool,
    },
    MissedBasket {
        player_id: u32,
        team_id: u8,
        t_ms: u64,
    },
    Foul {
        player_id: u32,
        team_id: u8,
        target_id: Option<u32>,
        t_ms: u64,
    },
    PossessionChange {
        player_id: u32,
        team_id: u8,
        origin: PossessionOrigin,
        t_ms: u64,
    },
    Dribble {
        player_id: u32,
        team_id: u8,
        t_ms: u64,
        x: f32,
        y: f32,
    },
    Pass {
        from_id: u32,
        to_id: u32,
        team_id: u8,
        t_ms: u64,
        x: f32,
        y: f32,
    },
    Steal {
        player_id: u32,
        team_id: u8,
        t_ms: u64,
    },
    Rebound {
        player_id: u32,
        team_id: u8,
        t_ms: u64,
    },
}

/// Layer 3: Possession Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PossessionContext {
    pub possession_id: u64,
    pub team_id: u8,
    pub ballhandler_id: Option<u32>,
    pub start_t_ms: u64,
    pub dribble_count: u32,
    pub pass_count: u32,
    pub offense_zone: CourtZone,
    pub is_transition: bool,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct PlayerStats {
    pub points: u32,
    pub fga: u32,
    pub fgm: u32,
    pub rebounds: u32,
    pub steals: u32,
    pub fouls: u32,
}

pub struct GameStateLedger {
    pub official_score: (u16, u16),
    pub history: Vec<GameEvent>,
    pub pending_buffer: VecDeque<PendingEvent>,
    pub current_possession: Option<PossessionContext>,
    pub possession_counter: u64,
    pub referee: GeometricReferee,
}

pub struct PendingEvent {
    pub event: GameEvent,
    pub expiration_t: u64,
}

impl GameStateLedger {
    pub fn new() -> Self {
        Self {
            official_score: (0, 0),
            history: Vec::new(),
            pending_buffer: VecDeque::new(),
            current_possession: None,
            possession_counter: 0,
            referee: GeometricReferee::new(),
        }
    }

    pub fn propose_event(&mut self, event: GameEvent, window_ms: u64) {
        self.update_possession(&event);

        let expiration = match &event {
            GameEvent::MadeBasket { t_ms, .. } => t_ms + window_ms,
            _ => 0,
        };

        if expiration == 0 {
            self.update_score(&event);
            self.history.push(event);
        } else {
            self.pending_buffer.push_back(PendingEvent {
                event,
                expiration_t: expiration,
            });
        }
    }

    fn update_possession(&mut self, event: &GameEvent) {
        match event {
            GameEvent::PossessionChange {
                player_id,
                team_id,
                origin,
                t_ms,
            } => {
                self.possession_counter += 1;
                let is_transition = match origin {
                    PossessionOrigin::Steal
                    | PossessionOrigin::Rebound
                    | PossessionOrigin::Turnover => true,
                    _ => false,
                };
                self.current_possession = Some(PossessionContext {
                    possession_id: self.possession_counter,
                    team_id: *team_id,
                    ballhandler_id: Some(*player_id),
                    start_t_ms: *t_ms,
                    dribble_count: 0,
                    pass_count: 0,
                    offense_zone: CourtZone::Backcourt,
                    is_transition,
                });
            }
            GameEvent::Dribble {
                player_id, x, y, ..
            } => {
                if let Some(ref mut ctx) = self.current_possession {
                    if ctx.ballhandler_id == Some(*player_id) {
                        ctx.dribble_count += 1;
                        let target_left = ctx.team_id == 1; // Simplification: Team 1 attacks Left
                        ctx.offense_zone =
                            self.referee.resolve_zone(&Point2::new(*x, *y), target_left);
                        if ctx.offense_zone == CourtZone::Paint {
                            ctx.is_transition = false;
                        }
                    }
                }
            }
            GameEvent::Pass { to_id, x, y, .. } => {
                if let Some(ref mut ctx) = self.current_possession {
                    ctx.pass_count += 1;
                    ctx.ballhandler_id = Some(*to_id);
                    let target_left = ctx.team_id == 1;
                    ctx.offense_zone = self.referee.resolve_zone(&Point2::new(*x, *y), target_left);
                }
            }
            _ => {}
        }
    }

    pub fn validate_event(&mut self, signal_t: u64, points_hint: u8) {
        if let Some(pos) = self.pending_buffer.iter().position(|p| {
            if let GameEvent::MadeBasket { t_ms, .. } = p.event {
                signal_t >= t_ms && signal_t <= p.expiration_t
            } else {
                false
            }
        }) {
            let mut pending = self.pending_buffer.remove(pos).unwrap();
            if let GameEvent::MadeBasket {
                ref mut points,
                ref mut is_official,
                ..
            } = pending.event
            {
                *points = points_hint;
                *is_official = true;
            }
            self.update_score(&pending.event);
            self.history.push(pending.event);
        }
    }

    fn update_score(&mut self, event: &GameEvent) {
        if let GameEvent::MadeBasket {
            points, team_id, ..
        } = event
        {
            if *team_id == 1 {
                self.official_score.0 += *points as u16;
            } else {
                self.official_score.1 += *points as u16;
            }
        }
    }

    pub fn generate_box_score(&self) -> HashMap<u32, PlayerStats> {
        let mut stats: HashMap<u32, PlayerStats> = HashMap::new();
        for event in &self.history {
            match event {
                GameEvent::MadeBasket {
                    player_id, points, ..
                } => {
                    let s = stats.entry(*player_id).or_default();
                    s.points += *points as u32;
                    s.fgm += 1;
                    s.fga += 1;
                }
                GameEvent::MissedBasket { player_id, .. } => {
                    stats.entry(*player_id).or_default().fga += 1;
                }
                GameEvent::Rebound { player_id, .. } => {
                    stats.entry(*player_id).or_default().rebounds += 1;
                }
                GameEvent::Steal { player_id, .. } => {
                    stats.entry(*player_id).or_default().steals += 1;
                }
                GameEvent::Foul { player_id, .. } => {
                    stats.entry(*player_id).or_default().fouls += 1;
                }
                _ => {}
            }
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_and_zone() {
        let mut ledger = GameStateLedger::new();

        // 1. Live ball turnover (Steal) -> Transition = True
        ledger.propose_event(
            GameEvent::PossessionChange {
                player_id: 10,
                team_id: 1,
                origin: PossessionOrigin::Steal,
                t_ms: 1000,
            },
            0,
        );

        let ctx = ledger.current_possession.as_ref().unwrap();
        assert!(ctx.is_transition);
        assert_eq!(ctx.offense_zone, CourtZone::Backcourt);

        // 2. Dribble in Paint -> Transition = False, Zone = Paint
        // Target Left (Team 1), Paint is near x=0
        ledger.propose_event(
            GameEvent::Dribble {
                player_id: 10,
                team_id: 1,
                t_ms: 2000,
                x: 100.0,
                y: 762.0,
            },
            0,
        );

        let ctx = ledger.current_possession.as_ref().unwrap();
        assert!(!ctx.is_transition);
        assert_eq!(ctx.offense_zone, CourtZone::Paint);
    }
}

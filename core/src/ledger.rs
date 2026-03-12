use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};

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
    },
    Pass {
        from_id: u32,
        to_id: u32,
        team_id: u8,
        t_ms: u64,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PossessionOrigin {
    Inbound,
    Rebound,
    Steal,
    Turnover,
    StartOfPeriod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CourtZone {
    Paint,
    WingLeft,
    WingRight,
    CornerLeft,
    CornerRight,
    TopOfKey,
    Backcourt,
}

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

/// L3.13: MVP Box Score Summary
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

    pub fn validate_event(&mut self, signal_t: u64, points_hint: u8) {
        if let Some(pos) = self.pending_buffer.iter().position(|p| {
            if let GameEvent::MadeBasket { t_ms, .. } = p.event {
                signal_t >= t_ms && signal_t <= p.expiration_t
            } else {
                false
            }
        }) {
            let mut pending = self.pending_buffer.remove(pos).unwrap();
            if let GameEvent::MadeBasket { ref mut points, ref mut is_official, .. } = pending.event {
                *points = points_hint;
                *is_official = true;
            }
            self.update_score(&pending.event);
            self.history.push(pending.event);
        }
    }

    fn update_possession(&mut self, event: &GameEvent) {
        match event {
            GameEvent::PossessionChange { player_id, team_id, origin: _, t_ms } => {
                self.possession_counter += 1;
                self.current_possession = Some(PossessionContext {
                    possession_id: self.possession_counter,
                    team_id: *team_id,
                    ballhandler_id: Some(*player_id),
                    start_t_ms: *t_ms,
                    dribble_count: 0,
                    pass_count: 0,
                    offense_zone: CourtZone::Backcourt,
                    is_transition: true,
                });
            },
            _ => {}
        }
    }

    fn update_score(&mut self, event: &GameEvent) {
        if let GameEvent::MadeBasket { points, team_id, .. } = event {
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
                GameEvent::MadeBasket { player_id, points, .. } => {
                    let s = stats.entry(*player_id).or_default();
                    s.points += *points as u32;
                    s.fgm += 1;
                    s.fga += 1;
                },
                GameEvent::MissedBasket { player_id, .. } => {
                    let s = stats.entry(*player_id).or_default();
                    s.fga += 1;
                },
                GameEvent::Rebound { player_id, .. } => {
                    stats.entry(*player_id).or_default().rebounds += 1;
                },
                GameEvent::Steal { player_id, .. } => {
                    stats.entry(*player_id).or_default().steals += 1;
                },
                GameEvent::Foul { player_id, .. } => {
                    stats.entry(*player_id).or_default().fouls += 1;
                },
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
    fn test_box_score_generation() {
        let mut ledger = GameStateLedger::new();
        
        // Player 5 makes a 2-pt shot, needs 2s window
        ledger.propose_event(GameEvent::MadeBasket {
            player_id: 5, team_id: 1, points: 2, t_ms: 1000, is_official: false,
        }, 2000);
        
        // Validate it
        ledger.validate_event(1500, 2);
        
        // Player 5 misses a shot (No window needed for misses usually)
        ledger.propose_event(GameEvent::MissedBasket {
            player_id: 5, team_id: 1, t_ms: 2000,
        }, 0);
        
        // Player 10 gets a rebound
        ledger.propose_event(GameEvent::Rebound {
            player_id: 10, team_id: 1, t_ms: 2100,
        }, 0);
        
        let box_score = ledger.generate_box_score();
        
        let p5 = box_score.get(&5).expect("Player 5 should have stats");
        assert_eq!(p5.points, 2);
        assert_eq!(p5.fga, 2);
        assert_eq!(p5.fgm, 1);
        
        let p10 = box_score.get(&10).expect("Player 10 should have stats");
        assert_eq!(p10.rebounds, 1);
    }

    #[test]
    fn test_possession_change_updates_current_context() {
        let mut ledger = GameStateLedger::new();
        ledger.propose_event(
            GameEvent::PossessionChange {
                player_id: 12,
                team_id: 2,
                origin: PossessionOrigin::Steal,
                t_ms: 5000,
            },
            0,
        );

        let possession = ledger.current_possession.as_ref().expect("possession should be tracked");
        assert_eq!(possession.possession_id, 1);
        assert_eq!(possession.team_id, 2);
        assert_eq!(possession.ballhandler_id, Some(12));
        assert_eq!(possession.start_t_ms, 5000);
        assert_eq!(possession.dribble_count, 0);
        assert_eq!(possession.pass_count, 0);
        match possession.offense_zone {
            CourtZone::Backcourt => {}
            _ => panic!("expected backcourt default"),
        }
        assert!(possession.is_transition);
    }
}

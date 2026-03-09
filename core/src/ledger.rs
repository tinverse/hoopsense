use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Represents a validated basketball event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameEvent {
    MadeBasket {
        player_id: u32,
        points: u8,
        t_ms: u64,
        is_official: bool, // Validated by Ref-API
    },
    Foul {
        player_id: u32,
        target_id: Option<u32>,
        t_ms: u64,
    },
    PossessionChange {
        player_id: u32,
        t_ms: u64,
    },
}

/// The GameStateLedger is the source of truth for the game's official record.
/// It maintains a buffer of "Pending" events that are awaiting validation
/// from the Ref-API (The 2-second window).
pub struct GameStateLedger {
    pub official_score: (u16, u16), // (Home, Away)
    pub history: Vec<GameEvent>,
    pub pending_buffer: VecDeque<PendingEvent>,
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
        }
    }

    /// Adds a potential event to the ledger.
    pub fn propose_event(&mut self, event: GameEvent, window_ms: u64) {
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

    /// Validates a pending event based on an official signal.
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

    fn update_score(&mut self, event: &GameEvent) {
        if let GameEvent::MadeBasket { points, .. } = event {
            // Simplified: adding to 'Home' score for now
            self.official_score.0 += *points as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ledger_validation_window() {
        let mut ledger = GameStateLedger::new();
        
        // 1. Propose a basket at T=1000ms with a 2s window
        ledger.propose_event(GameEvent::MadeBasket {
            player_id: 5,
            points: 2,
            t_ms: 1000,
            is_official: false,
        }, 2000);
        
        assert_eq!(ledger.pending_buffer.len(), 1);
        
        // 2. Validate with signal at T=2500ms (Within window: 1000 to 3000)
        ledger.validate_event(2500, 3);
        
        assert_eq!(ledger.pending_buffer.len(), 0);
        assert_eq!(ledger.history.len(), 1);
        
        if let GameEvent::MadeBasket { points, is_official, .. } = &ledger.history[0] {
            assert_eq!(*points, 3); // Upgraded to 3pt by Ref
            assert!(*is_official);
        } else {
            panic!("Event should be a MadeBasket");
        }
    }

    #[test]
    fn test_expired_validation() {
        let mut ledger = GameStateLedger::new();
        ledger.propose_event(GameEvent::MadeBasket {
            player_id: 5,
            points: 2,
            t_ms: 1000,
            is_official: false,
        }, 2000);
        
        // Validate with signal at T=4000ms (Outside 2s window)
        ledger.validate_event(4000, 3);
        
        assert_eq!(ledger.pending_buffer.len(), 1); // Still pending/expired
        assert_eq!(ledger.history.len(), 0);
    }
}

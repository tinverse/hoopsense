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
    /// It stays "pending" until validated or the window expires.
    pub fn propose_event(&mut self, event: GameEvent, window_ms: u64) {
        let expiration = match &event {
            GameEvent::MadeBasket { t_ms, .. } => t_ms + window_ms,
            _ => 0, // Other events might be immediate
        };
        
        if expiration == 0 {
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
        // Find the most recent pending basket within the window
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
            self.history.push(pending.event);
        }
    }
}

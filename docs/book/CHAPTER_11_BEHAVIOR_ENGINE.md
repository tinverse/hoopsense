# Chapter 11: The Behavior Engine – From Heuristics to State Machines

To build a professional sports analyzer, the AI must do more than just "see" frames; it must maintain a consistent mental model of every person on the court. This chapter explains the transition from simple `if` statements to a robust **Behavior Engine**.

## 1. The Problem with Heuristics

Simple heuristics (e.g., "if hand > head, then shoot") are fragile. They fail when a player is jumping for a rebound, scratching their head, or seen from an unusual angle. They also lack **Persistence**—the system forgets the context of the previous frame.

## 2. The Solution: The Behavior State Machine

We implement a **Finite State Machine (FSM)** for every `track_id`. 

### Possible States:
- **IDLE:** Standing or moving without a specific basketball action.
- **SHOOTING:** Middle of a jump-shot kinematic sequence.
- **PASSING:** Arm extension and ball release.
- **DRIBBLING:** Rhythmic hand-to-floor motion.
- **OFFICIAL_SIGNALING:** Referee performing NCAA signals.

### State Transitions:
Transitions are triggered by a **Rules Engine** that evaluates the last 15-60 frames of skeletal data. 

## 3. The Rules Engine (Kinematic Predicates)

Instead of hardcoding transitions, we define a library of **Kinematic Rules**.
- **Rule:** `JumpShotRule`
- **Predicate:** `(Velocity(Hips) > Threshold) AND (Y(Wrist) < Y(Head))`
- **Action:** If Predicate == TRUE, Transition to `SHOOTING`.

## 4. Why This Architecture?

1.  **Testability:** Each `Rule` can be unit-tested against synthetic MoCap data.
2.  **Rust-Ready:** This pattern (State/Rules) translates directly into Rust Traits and Enums, fulfilling our "Rust-First" long-term goal.
3.  **Hierarchy:** It respects the rule precedence (NCAA > NBA) by allowing us to swap the `RulesEngine` based on the game type.

---

**Summary for the Engineer:**
- **State** provides context.
- **Rules** provide logic.
- **Engines** provide the bridge between perception and game sense.

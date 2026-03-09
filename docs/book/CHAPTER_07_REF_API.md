# Chapter 7: The Ref-API – The Human Ground Truth

Even the most advanced AI can make mistakes in a chaotic game. The **Ref-API** is HoopSense's "Auditor," using the official signals of the human referee to validate and correct inferred basketball statistics.

## 1. The Referee as a Visual API

Referees follow a standardized visual language (NCAA > NBA > FIBA). By tracking the official, we gain access to a "Ground Truth" data stream that is already validated by the game's official rules.

### Key Visual Cues:
- **High Variance:** Unlike players in solid jerseys, NCAA officials wear stripes. This makes them easy to identify using **Contrast Variance** heuristics.
- **Pose Archetypes:** Most referee signals involve specific upper-body poses (e.g., both arms raised for a successful 3-pointer).

## 2. The 2-Second Temporal Window

Referee signals are reactive; they occur *after* the physical event.
- **The Physical Event (T=0):** A player shoots and the ball goes through the hoop.
- **The Official Signal (T+1.5s):** The referee raises both arms to confirm the 3 points.

The system maintains a **Circular Buffer** of the last 2 seconds of "Potential Events." When a Ref signal is detected, the AI looks back in time to match the signal to the physical action.

## 3. Conflict Resolution & The Rule Hierarchy

HoopSense follows a strict hierarchy for data validation:
1. **Referee Signal:** If the Ref signals a 3-point success, the system records 3 points, even if the geometric inference was uncertain.
2. **Geometric Inference:** If no whistle is blown, the system relies on the **Spatial Resolver** to determine if the ball went in.
3. **Action Recognition:** Provides the context (e.g., "Was it a jump shot or a dunk?").

## 4. Why This is a "Moat"

Most consumer basketball apps only track the ball. By tracking the **Referee**, HoopSense provides **Trustworthy Stats** that can be used for official scouting reports and league leaderboards. We aren't just "watching" the game; we are "understanding" the official ruling.

---

**Summary for the Engineer:**
- The Ref is our **Synchronous Auditor**.
- We use **Temporal Reconciliation** to bridge the gap between action and ruling.
- **Stripes** are a feature, not a bug.

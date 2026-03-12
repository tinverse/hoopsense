# HoopSense Use Case Model

## Primary Personas

### 1. The Athlete (Rohan)
- **Goal:** Level up and get recruited.
- **Value:** Receives an NBA 2K-style profile and a custom Roblox avatar based on real-world performance.
- **Key Use Case:** "View Personal Growth Over 6 Months."

### 2. The Parent / Fan
- **Goal:** Cherish and share highlights.
- **Value:** Automated video clips and high-quality photography extracted from footage.
- **Key Use Case:** "Share Automated Highlight Reel to Social Media."

### 3. The Scout / Coach
- **Goal:** Evaluate talent and tactical efficiency.
- **Value:** Professional-grade shot charts, defensive IQ stats, and practice monitoring.
- **Key Use Case:** "Analyze Player X's Shooting Percentage Under Heavy Contest."

### 4. The Advertiser / Sponsor
- **Goal:** Targeted branding and engagement.
- **Value:** Programmatic jersey branding on digital avatars based on real-world popularity (NIL).
- **Key Use Case:** "Bid for Digital Patch Placement on Popular Local Players."

## Core Interaction Scenarios

### UC-01: Game-Day Automated Tracking
- **Actor:** Coach / HoopBox
- **Description:** The system ingests a fixed or panning feed, identifies all players, classifies local motion, tracks possession context, and maintains a "Game DNA" ledger.
- **Outcome:** A JSONL stream of movements, attributed events, and stat-ready possession context.

### UC-02: Referee Signal Reconciliation
- **Actor:** System / Referee
- **Description:** The AI identifies a "3-point attempt" signal from the Ref and looks back 2 seconds to confirm the shot origin.
- **Outcome:** Official validation of the inferred statistic.

### UC-04: Possession Reasoning
- **Actor:** Coach / Analyst
- **Description:** The system tracks how a possession started, who handled the ball, how many dribbles occurred, and where the offense moved on the floor before the event.
- **Outcome:** A possession timeline that supports pass-chain analysis, drive identification, and transition vs half-court reasoning.

### UC-05: Stat Attribution
- **Actor:** Coach / Analyst / Athlete
- **Description:** The system combines local action predictions, ball trajectory, court geometry, and possession state to decide whether a sequence is a shot attempt, assist opportunity, rebound, steal, or turnover.
- **Outcome:** Box-score stats and explainable event records rather than raw neural labels.

### UC-03: Virtual Avatar Export
- **Actor:** Athlete
- **Description:** Skeletal pose data is mapped to a humanoid rig and textured with jersey crops.
- **Outcome:** A .gltf file ready for digital metaverses.

# Chapter 1: Identity Fusion – From Pixels to People

In computer vision, identifying a basketball player is not a single task; it is a **probabilistic fusion** of tracking, OCR, and re-identification. This chapter explains the bridge between seeing a "person" and knowing it is "Rohan."

## 1. The Fundamental Gap: Tracking vs. Identification

A standard tracker (like BoT-SORT) is designed for **temporal consistency**. It sees a bounding box in Frame 1 and predicts where it will be in Frame 2. If it is right, it assigns the same `track_id`.

**The Problem:**
- **Occlusion:** When two players cross paths, the bounding boxes overlap. The tracker often swaps their IDs.
- **Camera Cuts:** If the video cuts to a replay and back, the tracker starts from scratch. All history is lost.

To solve this, we must map **Temporary Track IDs** to **Persistent Player IDs**.

## 2. The Logic of Buffered Voting

We don't run OCR (Optical Character Recognition) on every frame. It is computationally expensive and error-prone due to motion blur and jersey folds. Instead, we use a **Majority Consensus Voting** system.

### The Algorithm:
1. **Torso Buffer:** For every `track_id`, we store a `deque` (double-ended queue) of the last 30 frames of their torso crop.
2. **Confidence Gating:** We only run OCR on the 5 clearest crops (calculated by Laplacian variance to ensure sharpness).
3. **Regex Normalization:** OCR engines often misread digits (e.g., "1O" instead of "10"). We normalize all strings to digits and filter out invalid jersey numbers (>99 or empty).
4. **Majority Vote:**
   - Votes: `["24", "24", "2A", "24", "84"]`
   - Winner: `"24"`
   - Confidence: `0.80`

## 3. Team Color Clustering (Reducing the Search Space)

To prevent the AI from guessing a number that isn't on the roster, we use color as a **prior**.

Instead of simple grayscale, we use **HSV (Hue, Saturation, Value)** clustering.
- **Hue:** Handles the color family (Red team vs. Blue team).
- **Value:** Handles lighting variations (shadows on one side of the court).

By knowing a player is on the "Dark" team, we can throw out OCR guesses for numbers that only exist on the "Light" team roster.

## 4. Re-ID: The Final Anchor

Even if OCR fails (e.g., a player's back is to the camera for 30 seconds), we can use a **Re-ID Embedding**.

A Re-ID model (like a ResNet-50 trained on the Market-1501 dataset) converts a player's appearance—height, skin tone, shoe color, arm sleeves—into a **vector of 512 numbers**.

When a player disappears and reappears, we compare their new vector to the "Gallery" of known vectors using **Cosine Similarity**. If the similarity is >0.9, we know it's the same person, even without seeing their jersey.

---

**Summary for the Engineer:**
- **Track IDs** are for movement (shorter-term).
- **Player IDs** are for stats (long-term).
- **Fusion** is the math that links them using OCR votes and visual embeddings.

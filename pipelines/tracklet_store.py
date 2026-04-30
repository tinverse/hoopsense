"""Runtime tracklet state and low-cost ReID evidence collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ReIDEvidence:
    """One appearance observation that can later be replaced by learned embeddings."""

    source: str
    vector: list[float]
    confidence: float
    frame_idx: int


class ReIDEvidenceExtractor(Protocol):
    source: str

    def extract(self, frame_bgr, bbox_xywh, *, frame_idx: int) -> ReIDEvidence | None:
        """Return appearance evidence for one player crop, or None when unusable."""


class ColorHistogramReIDExtractor:
    """Cheap baseline ReID extractor for real-time runtime wiring.

    This is not meant to be final ReID. It gives the pipeline a stable evidence
    contract using a very low-cost feature so tracklet logic can be built and
    measured before a learned embedding model is selected.
    """

    source = "torso_color_histogram_v1"

    def __init__(self, bins_per_channel: int = 4):
        self.bins_per_channel = int(bins_per_channel)

    def extract(self, frame_bgr, bbox_xywh, *, frame_idx: int) -> ReIDEvidence | None:
        crop = self._crop_torso(frame_bgr, bbox_xywh)
        if crop is None or crop.size == 0:
            return None
        pixels = crop.reshape(-1, 3).astype(np.float32)
        if pixels.size == 0:
            return None
        bins = np.clip((pixels / 256.0 * self.bins_per_channel).astype(np.int32), 0, self.bins_per_channel - 1)
        bin_ids = (
            bins[:, 0] * self.bins_per_channel * self.bins_per_channel
            + bins[:, 1] * self.bins_per_channel
            + bins[:, 2]
        )
        hist = np.bincount(bin_ids, minlength=self.bins_per_channel ** 3).astype(np.float32)
        total = float(hist.sum())
        if total <= 0.0:
            return None
        hist /= total
        confidence = min(1.0, total / 2000.0)
        return ReIDEvidence(
            source=self.source,
            vector=[round(float(value), 6) for value in hist.tolist()],
            confidence=round(float(confidence), 4),
            frame_idx=int(frame_idx),
        )

    @staticmethod
    def _crop_torso(frame_bgr, bbox_xywh):
        if frame_bgr is None:
            return None
        height, width = frame_bgr.shape[:2]
        cx, cy, bw, bh = [float(v) for v in bbox_xywh]
        if bw <= 1.0 or bh <= 1.0:
            return None
        x1 = int(max(0, np.floor(cx - 0.30 * bw)))
        x2 = int(min(width, np.ceil(cx + 0.30 * bw)))
        y1 = int(max(0, np.floor(cy - 0.35 * bh)))
        y2 = int(min(height, np.ceil(cy + 0.10 * bh)))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]


@dataclass
class TrackletState:
    track_id: int
    first_frame_idx: int
    last_frame_idx: int
    last_bbox_xywh: list[float]
    last_court_xy: list[float]
    last_confidence: float
    observation_count: int = 1
    missing_gap_frames: int = 0
    velocity_court_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])
    reid_sample_count: int = 0
    reid_source: str | None = None
    reid_prototype: list[float] | None = None
    reid_confidence: float = 0.0
    last_reid_frame_idx: int | None = None

    @property
    def age_frames(self) -> int:
        return int(self.last_frame_idx - self.first_frame_idx + 1)

    def update(
        self,
        *,
        frame_idx: int,
        bbox_xywh,
        court_xy,
        confidence: float,
        reid_evidence: ReIDEvidence | None = None,
    ) -> None:
        previous_court = np.array(self.last_court_xy, dtype=np.float32)
        next_court = np.array([float(court_xy[0]), float(court_xy[1])], dtype=np.float32)
        velocity = next_court - previous_court
        self.last_frame_idx = int(frame_idx)
        self.last_bbox_xywh = [round(float(v), 3) for v in bbox_xywh]
        self.last_court_xy = [round(float(v), 3) for v in next_court.tolist()]
        self.last_confidence = round(float(confidence), 4)
        self.observation_count += 1
        self.missing_gap_frames = 0
        self.velocity_court_xy = [round(float(v), 3) for v in velocity.tolist()]
        if reid_evidence is not None:
            self._update_reid(reid_evidence)

    def mark_missing(self, frame_idx: int) -> None:
        self.missing_gap_frames = max(0, int(frame_idx) - int(self.last_frame_idx))

    def _update_reid(self, evidence: ReIDEvidence) -> None:
        vector = np.array(evidence.vector, dtype=np.float32)
        if vector.size == 0:
            return
        if self.reid_prototype is None:
            prototype = vector
        else:
            previous = np.array(self.reid_prototype, dtype=np.float32)
            prototype = 0.80 * previous + 0.20 * vector
            total = float(prototype.sum())
            if total > 0.0:
                prototype = prototype / total
        self.reid_sample_count += 1
        self.reid_source = evidence.source
        self.reid_prototype = [round(float(value), 6) for value in prototype.tolist()]
        self.reid_confidence = round(max(float(self.reid_confidence), float(evidence.confidence)), 4)
        self.last_reid_frame_idx = int(evidence.frame_idx)

    def to_payload(self) -> dict:
        return {
            "track_id": int(self.track_id),
            "first_frame_idx": int(self.first_frame_idx),
            "last_frame_idx": int(self.last_frame_idx),
            "age_frames": int(self.age_frames),
            "observation_count": int(self.observation_count),
            "missing_gap_frames": int(self.missing_gap_frames),
            "last_confidence": round(float(self.last_confidence), 4),
            "last_bbox_xywh": list(self.last_bbox_xywh),
            "last_court_xy": list(self.last_court_xy),
            "velocity_court_xy": list(self.velocity_court_xy),
            "reid": {
                "source": self.reid_source,
                "sample_count": int(self.reid_sample_count),
                "confidence": round(float(self.reid_confidence), 4),
                "prototype_dim": len(self.reid_prototype or []),
            },
        }


class TrackletStore:
    """Own runtime tracklets independently of final player identity."""

    def __init__(self, *, reid_extractor: ReIDEvidenceExtractor | None = None, reid_sample_interval_frames: int = 15):
        self.reid_extractor = reid_extractor
        self.reid_sample_interval_frames = max(1, int(reid_sample_interval_frames))
        self.tracklets: dict[int, TrackletState] = {}

    def update(
        self,
        *,
        track_id: int,
        frame_idx: int,
        bbox_xywh,
        court_xy,
        confidence: float,
        frame_bgr=None,
    ) -> TrackletState:
        track_key = int(track_id)
        existing = self.tracklets.get(track_key)
        should_sample_reid = (
            existing is None
            or existing.last_reid_frame_idx is None
            or (int(frame_idx) - int(existing.last_reid_frame_idx)) >= self.reid_sample_interval_frames
        )
        evidence = None
        if should_sample_reid and self.reid_extractor is not None and frame_bgr is not None:
            evidence = self.reid_extractor.extract(frame_bgr, bbox_xywh, frame_idx=int(frame_idx))
        if existing is None:
            state = TrackletState(
                track_id=track_key,
                first_frame_idx=int(frame_idx),
                last_frame_idx=int(frame_idx),
                last_bbox_xywh=[round(float(v), 3) for v in bbox_xywh],
                last_court_xy=[round(float(court_xy[0]), 3), round(float(court_xy[1]), 3)],
                last_confidence=round(float(confidence), 4),
            )
            if evidence is not None:
                state._update_reid(evidence)
            self.tracklets[track_key] = state
            return state
        existing.update(
            frame_idx=int(frame_idx),
            bbox_xywh=bbox_xywh,
            court_xy=court_xy,
            confidence=float(confidence),
            reid_evidence=evidence,
        )
        return existing

    def mark_missing_except(self, visible_track_ids, *, frame_idx: int) -> None:
        visible = {int(track_id) for track_id in visible_track_ids}
        for track_id, state in self.tracklets.items():
            if track_id not in visible:
                state.mark_missing(int(frame_idx))

    def get(self, track_id: int) -> TrackletState | None:
        return self.tracklets.get(int(track_id))

    def summary(self) -> dict:
        return {
            "kind": "runtime_tracklet_store_v1",
            "tracklet_count": len(self.tracklets),
            "reid_source": getattr(self.reid_extractor, "source", None),
            "reid_sample_interval_frames": int(self.reid_sample_interval_frames),
            "reid_tracklet_count": sum(1 for state in self.tracklets.values() if state.reid_sample_count > 0),
        }

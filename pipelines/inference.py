import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from pipelines.behavior_engine import BehaviorStateMachine, PossessionEngine
from pipelines.ball_tracking import BallSearchScheduler, BallSearchSchedulerConfig, BallStateTracker
from pipelines.court_pose import CourtPoseTracker
from pipelines.geometry import (
    geometry_evidence_gate,
    lift_keypoints_to_3d,
    play_region_prior_for_point,
    project_pixel_to_court,
)
from pipelines.mvp_event_engine import MvpEventRuleEngine
from pipelines.mvp_stat_accumulator import MvpStatAccumulator
from pipelines.resource_policy import resolve_torch_device
from pipelines.tracklet_store import ColorHistogramReIDExtractor, TrackletStore

DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_GROUNDING_DINO_PROMPT = "basketball court. basketball player. basketball hoop. basketball referee."
try:
    from tools.review.labeller.grounding_dino_bootstrap import (
        DEFAULT_GROUNDING_DINO_MODEL as _IMPORTED_GROUNDING_DINO_MODEL,
        DEFAULT_GROUNDING_DINO_PROMPT as _IMPORTED_GROUNDING_DINO_PROMPT,
        GroundingDinoBootstrapper,
    )

    DEFAULT_GROUNDING_DINO_MODEL = _IMPORTED_GROUNDING_DINO_MODEL
    DEFAULT_GROUNDING_DINO_PROMPT = _IMPORTED_GROUNDING_DINO_PROMPT
except Exception:  # pragma: no cover - fail-closed when optional deps are missing
    GroundingDinoBootstrapper = None


def get_label_map(spec_path="specs/basketball_ncaa.yaml"):
    """Load the Action Brain output label map from the basketball spec.

    The inference path emits the string label chosen by Action Brain, so we
    resolve the integer-to-string mapping once at module load. When the spec is
    missing, keep a small built-in default so local smoke runs still work.
    """
    if not os.path.exists(spec_path):
        return {
            0: "jump_shot",
            1: "crossover",
            2: "rebound",
            3: "block",
            4: "steal",
        }
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)
    labels = []
    for cat in spec.get("categories", []):
        labels.extend(cat.get("rules", []))
    labels = sorted(list(set(labels)))
    return {i: label for i, label in enumerate(labels)}


LABEL_MAP_INV = get_label_map()
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32
BALL_DEFAULT_Z = 50.0
BOOTSTRAP_DISCONTINUITY_THRESHOLD = 22.0
PAN_DRIFT_MIN_SHARED_TRACKS = 4
PAN_DRIFT_MEDIAN_DISPLACEMENT_PX = 55.0
PAN_DRIFT_ALIGNMENT_SPREAD_PX = 22.0
PLAY_REGION_GEOMETRY_MIN_PRIOR = 0.5
BOOTSTRAP_PAN_RECOMPUTE_COOLDOWN_FRAMES = 30


class KalmanFilter:
    """Minimal 1D Kalman filter used to smooth court-space x/y coordinates.

    This is intentionally tiny: the current runtime path only smooths the
    projected court center for each track before handing it to downstream game
    logic. More complex multi-dimensional smoothing now lives in the review
    pipeline, but this class preserves the existing runtime behavior.
    """

    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.posteri_estimate = measurement
            self.initialized = True
            return self.posteri_estimate
        pri_est = self.posteri_estimate
        pri_err = self.posteri_error_estimate + self.process_variance
        blending = pri_err / (pri_err + self.measurement_variance)
        self.posteri_estimate = pri_est + blending * (measurement - pri_est)
        self.posteri_error_estimate = (1 - blending) * pri_err
        return self.posteri_estimate


class TrackManager:
    """State container for one tracked player across frames.

    Responsibilities are deliberately narrow:
    - smooth projected court position
    - retain the last 30 pose frames for Action Brain
    - maintain the local behavior state machine

    It does not own possession logic or output writing.
    """

    def __init__(self, tid):
        self.tid = tid
        self.kf_x = KalmanFilter()
        self.kf_y = KalmanFilter()
        self.kpt_history = deque(maxlen=30)
        self.state_machine = BehaviorStateMachine()
        self.court_x = 0.0
        self.court_y = 0.0
        self.pos_3d = np.zeros(3)
        self.team = 1  # Default to Home for now
        self.play_region_prior = 0.0
        self.geometry_region_ok = True

    def update_position(self, x, y):
        self.court_x = self.kf_x.update(x)
        self.court_y = self.kf_y.update(y)
        return self.court_x, self.court_y

    def add_keypoints(self, kpts, h_matrix):
        """Append normalized keypoints and keep a rough lifted 3D player anchor.

        The lifted position is intentionally approximate: we average the hips so
        possession logic can reason about coarse player locations without
        building a richer skeletal model here.
        """
        if kpts is not None:
            self.kpt_history.append(kpts)
            kpts_np = np.array(kpts)
            if np.any(kpts_np[11:13] > 0):
                lifted = lift_keypoints_to_3d(kpts_np, h_matrix)
                self.pos_3d = lifted[11:13].mean(axis=0)

    def is_ready(self):
        return len(self.kpt_history) == 30


class RuntimeDiscontinuityDetector:
    """Detect abrupt scene shifts that should invalidate runtime bootstrap priors."""

    def __init__(self, threshold=BOOTSTRAP_DISCONTINUITY_THRESHOLD):
        self.threshold = float(threshold)
        self.last_signature = None

    def update(self, frame):
        frame_np = np.asarray(frame, dtype=np.float32)
        if frame_np.ndim == 3:
            signature = float(frame_np.mean(axis=2).mean())
        else:
            signature = float(frame_np.mean())
        score = 0.0 if self.last_signature is None else abs(signature - self.last_signature)
        triggered = self.last_signature is not None and score >= self.threshold
        self.last_signature = signature
        return {
            "score": round(float(score), 4),
            "triggered": bool(triggered),
            "signature": round(float(signature), 4),
        }


class RuntimePanDriftDetector:
    """Detect coherent track motion that likely reflects camera pan/layout drift."""

    def __init__(
        self,
        *,
        min_shared_tracks=PAN_DRIFT_MIN_SHARED_TRACKS,
        median_displacement_px=PAN_DRIFT_MEDIAN_DISPLACEMENT_PX,
        alignment_spread_px=PAN_DRIFT_ALIGNMENT_SPREAD_PX,
    ):
        self.min_shared_tracks = int(min_shared_tracks)
        self.median_displacement_px = float(median_displacement_px)
        self.alignment_spread_px = float(alignment_spread_px)
        self.previous_centers = {}

    def update(self, boxes_xywh, tids, classes):
        current_centers = {}
        for bbox_xywh, tid, class_id in zip(boxes_xywh, tids, classes):
            if int(class_id) != PERSON_CLASS_ID:
                continue
            current_centers[int(tid)] = np.array(
                [float(bbox_xywh[0]), float(bbox_xywh[1])],
                dtype=np.float32,
            )

        shared_ids = sorted(set(self.previous_centers).intersection(current_centers))
        if len(shared_ids) < self.min_shared_tracks:
            self.previous_centers = current_centers
            return {
                "triggered": False,
                "shared_track_count": int(len(shared_ids)),
                "median_displacement_px": 0.0,
                "alignment_spread_px": 0.0,
                "median_dx": 0.0,
                "median_dy": 0.0,
            }

        displacements = np.asarray(
            [current_centers[tid] - self.previous_centers[tid] for tid in shared_ids],
            dtype=np.float32,
        )
        median_vector = np.median(displacements, axis=0)
        residuals = displacements - median_vector
        median_displacement = float(np.linalg.norm(median_vector))
        alignment_spread = float(np.median(np.linalg.norm(residuals, axis=1)))
        triggered = (
            median_displacement >= self.median_displacement_px
            and alignment_spread <= self.alignment_spread_px
        )
        self.previous_centers = current_centers
        return {
            "triggered": bool(triggered),
            "shared_track_count": int(len(shared_ids)),
            "median_displacement_px": round(median_displacement, 4),
            "alignment_spread_px": round(alignment_spread, 4),
            "median_dx": round(float(median_vector[0]), 4),
            "median_dy": round(float(median_vector[1]), 4),
        }


@dataclass
class RuntimeBootstrapState:
    """Mutable segment-scoped runtime bootstrap context."""

    segment_id: int = 0
    context: dict | None = None


def construct_features_v2(
    kpt_history,
    last_ball_2d,
    h_matrix,
    player_court_pos,
    kpts_3d_approx,
    ball_3d_approx,
    device,
):
    """Build the fixed `(1, 30, 72)` Action Brain tensor for one player.

    This remains the narrow neural contract:
    - 2D pose
    - first-order pose velocity
    - wrist-to-ball distances
    - coarse court position

    Possession, event attribution, and stats are intentionally outside this
    tensor and are handled in higher layers.
    """
    import torch

    if len(kpt_history) < 30:
        return None
    kpts_2d = np.array(kpt_history)
    features = []
    ball_court_xy = project_pixel_to_court(last_ball_2d[0], last_ball_2d[1], h_matrix)
    ball_z = ball_3d_approx[2] if ball_3d_approx is not None else 100.0
    ball_3d = np.array([ball_court_xy[0], ball_court_xy[1], ball_z])
    for t in range(30):
        pose = kpts_2d[t].flatten()
        delta = kpts_2d[t] - kpts_2d[max(0, t - 1)]
        vel = delta.flatten() * 0.1
        kpts_3d = lift_keypoints_to_3d(kpts_2d[t], h_matrix)
        dist_l = np.linalg.norm(kpts_3d[9] - ball_3d) * 0.01
        dist_r = np.linalg.norm(kpts_3d[10] - ball_3d) * 0.01
        court_context = player_court_pos * 0.001
        row = np.concatenate([pose, vel, [dist_l, dist_r], court_context])
        features.append(row)
    return torch.FloatTensor(np.array(features)).unsqueeze(0).to(device)


@dataclass
class InferenceConfig:
    """Runtime configuration for one inference invocation.

    Kept as plain data so the top-level orchestration can be inspected and
    tested without dragging model objects or file handles along with it.
    """

    video_path: str
    output_dir: str = "data"
    smoke_test: bool = False
    player_model_name: str = "yolov8n.pt"
    player_tracker_backend: str = "default"
    pose_model_name: str = "yolov8n-pose.pt"
    ball_model_name: str = "yolov8n.pt"
    player_conf_threshold: float = 0.12
    pose_conf_threshold: float = 0.20
    calibration_path: str = "data/training/camera_calibration.json"
    fallback_calibration_path: str = "data/calibration.json"
    brain_path: str = "data/models/action_brain.pt"
    output_filename: str = "intelligent_game_dna.jsonl"
    bootstrap_foreground_backend: str = "none"
    bootstrap_foreground_model: str = DEFAULT_GROUNDING_DINO_MODEL
    bootstrap_foreground_prompt: str = DEFAULT_GROUNDING_DINO_PROMPT
    bootstrap_discontinuity_threshold: float = BOOTSTRAP_DISCONTINUITY_THRESHOLD
    play_region_geometry_min_prior: float = PLAY_REGION_GEOMETRY_MIN_PRIOR
    bootstrap_pan_recompute_cooldown_frames: int = BOOTSTRAP_PAN_RECOMPUTE_COOLDOWN_FRAMES
    reid_sample_interval_frames: int = 15
    ball_full_frame_interval_frames: int = 1
    ball_missing_full_frame_interval_frames: int = 1
    ball_roi_search_enabled: bool = False
    ball_roi_max_count: int = 4
    device: str = "auto"


class CalibrationResolver:
    """Resolve clip-specific homography data for the current video.

    The active inference path supports two calibration storage shapes:
    - per-clip data under `data/training/camera_calibration.json`
    - a fallback single global matrix under `data/calibration.json`

    The resolver normalizes both into a single `homography_for_frame(...)`
    interface.
    """

    def __init__(self, calibration_path, fallback_calibration_path):
        self.calibration_path = Path(calibration_path)
        self.fallback_calibration_path = Path(fallback_calibration_path)
        self.global_h = np.eye(3)
        self.h_sequence = None

    def load_for_clip(self, clip_id):
        """Load the best available calibration data for one clip id."""
        self.global_h = np.eye(3)
        self.h_sequence = None
        if self.calibration_path.exists():
            with open(self.calibration_path, "r") as f:
                try:
                    calibrations = json.load(f)
                except Exception:
                    calibrations = {}
            calibration = calibrations.get(clip_id)
            if calibration:
                if "h_sequence" in calibration:
                    self.h_sequence = {
                        int(k): np.array(v) for k, v in calibration["h_sequence"].items()
                    }
                elif "h_matrix" in calibration:
                    self.global_h = np.array(calibration["h_matrix"])
        elif self.fallback_calibration_path.exists():
            with open(self.fallback_calibration_path, "r") as f:
                fallback = json.load(f)
            self.global_h = np.array(fallback.get("h_matrix", np.eye(3)))
        return self

    def homography_for_frame(self, frame_idx):
        """Return the per-frame homography if available, else the global one."""
        if self.h_sequence:
            return self.h_sequence.get(frame_idx, self.global_h)
        return self.global_h


class ModelBundle:
    """Lazy owner of runtime ML resources used by inference.

    Separating model loading from the frame loop keeps `extract_game_dna(...)`
    readable and isolates optional Action Brain setup from the rest of the
    pipeline.
    """

    def __init__(self, player_model_name, pose_model_name, ball_model_name, brain_path, label_map, requested_device="auto"):
        self.player_model_name = player_model_name
        self.pose_model_name = pose_model_name
        self.ball_model_name = ball_model_name
        self.brain_path = brain_path
        self.label_map = label_map
        self.requested_device = requested_device
        self.device = None
        self.resource_policy = None
        self.player_model = None
        self.pose_model = None
        self.ball_model = None
        self.brain = None

    def load(self):
        """Load YOLO unconditionally and Action Brain only if a checkpoint exists."""
        import torch
        from ultralytics import YOLO
        from core.vision.action_brain import ActionBrain

        self.device, self.resource_policy = resolve_torch_device(self.requested_device)
        self.player_model = YOLO(self.player_model_name)
        self.pose_model = YOLO(self.pose_model_name)
        self.ball_model = YOLO(self.ball_model_name)
        if os.path.exists(self.brain_path):
            brain = ActionBrain(num_classes=len(self.label_map)).to(self.device)
            brain.load_state_dict(torch.load(self.brain_path, map_location=self.device))
            brain.eval()
            self.brain = brain
        return self

    def smoke_test(self):
        """Exercise YOLO initialization without running the full pipeline."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.player_model(dummy)
        self.pose_model(dummy)
        self.ball_model(dummy)


class JsonlEventWriter:
    """Small context-managed JSONL sink.

    The existing output contract is append-one-record-per-line, so we preserve
    that shape while moving file ownership out of the main loop.
    """

    def __init__(self, output_path):
        self.output_path = output_path
        self._fh = None

    def __enter__(self):
        self._fh = open(self.output_path, "w")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.close()
        return False

    def write(self, payload):
        self._fh.write(json.dumps(payload) + "\n")


class MvpEventAdapter:
    """Translate runtime events into MVP-attribution payloads.

    The current inference path already emits simple `pass` / `catch` / `steal`
    events. This adapter preserves those rows and adds a parallel
    `attributed_event` contract shaped for the deterministic MVP event-rule
    engine.
    """

    def __init__(self, rule_engine=None):
        self.rule_engine = rule_engine or MvpEventRuleEngine()

    def adapt(self, raw_event, player_map):
        kind = raw_event.get("kind")
        if kind == "catch":
            return [self._build_and_count(
                event_type="catch",
                actor_id=raw_event.get("player_id"),
                team_id=self._team_for(player_map, raw_event.get("player_id")),
                t_ms=raw_event.get("t_ms"),
                evidence={},
            )]
        if kind == "pass":
            return [self._build_and_count(
                event_type="pass",
                actor_id=raw_event.get("from"),
                secondary_actor_id=raw_event.get("to"),
                team_id=raw_event.get("team_id") or self._team_for(player_map, raw_event.get("from")),
                t_ms=raw_event.get("t_ms"),
                next_event_type="catch",
                evidence={},
            )]
        if kind == "steal":
            outputs = []
            offender_id = raw_event.get("from")
            if offender_id is not None:
                outputs.append(self._build_and_count(
                    event_type="turnover",
                    actor_id=offender_id,
                    secondary_actor_id=raw_event.get("player_id"),
                    team_id=self._team_for(player_map, offender_id),
                    t_ms=raw_event.get("t_ms"),
                    next_event_type="steal",
                    evidence={"loss_of_team_control": True},
                ))
            outputs.append(self._build_and_count(
                event_type="steal",
                actor_id=raw_event.get("player_id"),
                secondary_actor_id=offender_id,
                team_id=raw_event.get("team_id") or self._team_for(player_map, raw_event.get("player_id")),
                t_ms=raw_event.get("t_ms"),
                evidence={"possession_gain": True},
            ))
            return outputs
        return []

    def build_shot_attempt_candidate(self, *, track_id, team_id, t_ms, evidence, shot_value=None):
        return self._build_and_count(
            event_type="shot_attempt",
            actor_id=track_id,
            team_id=team_id,
            t_ms=t_ms,
            evidence=evidence,
            shot_value=shot_value,
        )

    def _build_and_count(
        self,
        *,
        event_type,
        actor_id,
        team_id,
        t_ms,
        evidence,
        secondary_actor_id=None,
        preceding_event_type=None,
        next_event_type=None,
        terminal_event_type=None,
        shot_value=None,
    ):
        payload = {
            "kind": "attributed_event",
            "event_type": event_type,
            "actor_id": int(actor_id) if actor_id is not None else None,
            "secondary_actor_id": int(secondary_actor_id) if secondary_actor_id is not None else None,
            "team_id": int(team_id) if team_id is not None else None,
            "t_ms": int(t_ms) if t_ms is not None else None,
            "live_play": True,
            "preceding_event_type": preceding_event_type,
            "next_event_type": next_event_type,
            "terminal_event_type": terminal_event_type,
            "shot_value": shot_value,
            "evidence": evidence or {},
        }
        payload["rule_validation"] = self.rule_engine.validate_event(payload)
        payload["stat_deltas"] = (
            {}
            if payload["rule_validation"]
            else self.rule_engine.stat_deltas_for_event(payload)
        )
        return payload

    @staticmethod
    def _team_for(player_map, player_id):
        if player_id is None:
            return None
        player_state = player_map.get(player_id)
        if not player_state:
            return None
        return player_state.get("team")


class FrameResultAdapter:
    """Translate raw Ultralytics frame results into pipeline-friendly pieces.

    This keeps the main extractor from having to know about the low-level YOLO
    tensor layout and centralizes the ball-extraction policy.
    """

    @staticmethod
    def has_track_ids(result):
        """Guard the rest of the pipeline against empty/untracked detections."""
        return (
            result
            and result[0].boxes is not None
            and result[0].boxes.id is not None
        )

    @staticmethod
    def extract_arrays(result):
        """Extract the raw arrays needed by the rest of the runtime pipeline."""
        result0 = result[0]
        boxes_xywh = result0.boxes.xywh
        cls = result0.boxes.cls
        conf = result0.boxes.conf
        tids = result0.boxes.id.int().cpu().numpy()
        classes = result0.boxes.cls.int().cpu().numpy()
        keypoints = (
            result0.keypoints.xyn.cpu().numpy()
            if result0.keypoints is not None
            else None
        )
        return boxes_xywh, cls, conf, tids, classes, keypoints

    @staticmethod
    def extract_tracked_arrays(result):
        """Extract tracked detector arrays when the source has no keypoints."""
        result0 = result[0]
        boxes_xywh = result0.boxes.xywh
        cls = result0.boxes.cls
        conf = result0.boxes.conf
        tids = result0.boxes.id.int().cpu().numpy()
        classes = result0.boxes.cls.int().cpu().numpy()
        return boxes_xywh, cls, conf, tids, classes

    @staticmethod
    def _bbox_iou_xyxy(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    @staticmethod
    def _center_distance_ratio(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        center_a = np.array([(ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5], dtype=np.float32)
        center_b = np.array([(bx1 + bx2) * 0.5, (by1 + by2) * 0.5], dtype=np.float32)
        distance = float(np.linalg.norm(center_a - center_b))
        norm = max(ay2 - ay1, by2 - by1, 1.0)
        return distance / norm

    @staticmethod
    def match_pose_keypoints(tracked_result, pose_result):
        """Project pose detections onto tracked person boxes by IoU and center distance."""
        if not tracked_result or tracked_result[0].boxes is None or len(tracked_result[0].boxes) == 0:
            return None
        if (
            not pose_result
            or pose_result[0].boxes is None
            or len(pose_result[0].boxes) == 0
            or pose_result[0].keypoints is None
        ):
            return [None for _ in range(len(tracked_result[0].boxes))]
        tracked_boxes = tracked_result[0].boxes.xyxy.cpu().numpy()
        pose_boxes = pose_result[0].boxes.xyxy.cpu().numpy()
        pose_keypoints = pose_result[0].keypoints.xyn.cpu().numpy()
        matched = [None for _ in range(len(tracked_boxes))]
        used_pose_indices = set()
        for tracked_idx, tracked_box in enumerate(tracked_boxes):
            best_idx = None
            best_score = None
            for pose_idx, pose_box in enumerate(pose_boxes):
                if pose_idx in used_pose_indices:
                    continue
                iou = FrameResultAdapter._bbox_iou_xyxy(tracked_box, pose_box)
                distance_ratio = FrameResultAdapter._center_distance_ratio(tracked_box, pose_box)
                if iou < 0.35 and distance_ratio > 0.45:
                    continue
                score = float(iou) - 0.2 * float(distance_ratio)
                if best_score is None or score > best_score:
                    best_idx = pose_idx
                    best_score = score
            if best_idx is None:
                continue
            matched[tracked_idx] = pose_keypoints[best_idx]
            used_pose_indices.add(best_idx)
        return matched

    @staticmethod
    def extract_ball_arrays(result):
        """Extract raw ball detector arrays from a detection-model result."""
        if not result or result[0].boxes is None or len(result[0].boxes) == 0:
            return [], [], []
        result0 = result[0]
        boxes_xywh = result0.boxes.xywh.cpu().numpy()
        cls = result0.boxes.cls.int().cpu().numpy()
        conf = result0.boxes.conf.cpu().numpy()
        return boxes_xywh, cls, conf

    @staticmethod
    def extract_ball_state(
        boxes_xywh,
        cls,
        conf,
        ball_tracker,
        h_matrix,
        t_ms,
        writer,
        bootstrap_context=None,
        bootstrap_segment_id=None,
        play_region_geometry_min_prior=PLAY_REGION_GEOMETRY_MIN_PRIOR,
        ball_search_plan=None,
        ball_candidate_metadata=None,
    ):
        """Select one auditable ball state and emit it as the runtime ball row."""
        ball_state = ball_tracker.update(
            boxes_xywh,
            cls,
            conf,
            bootstrap_context=bootstrap_context,
            candidate_metadata=ball_candidate_metadata,
        )
        center_xy = ball_state.get("center_xy")
        updated_ball_2d = np.array(center_xy, dtype=np.float32) if center_xy else np.array([0.0, 0.0], dtype=np.float32)
        ball_3d = None
        if center_xy:
            ball_court_xy = project_pixel_to_court(updated_ball_2d[0], updated_ball_2d[1], h_matrix)
            ball_3d = np.array([ball_court_xy[0], ball_court_xy[1], BALL_DEFAULT_Z])
            ball_state["court_xy"] = [round(float(ball_court_xy[0]), 3), round(float(ball_court_xy[1]), 3)]
        writer.write(
            {
                "kind": "ball",
                "t_ms": t_ms,
                "state": ball_state["state"],
                "confidence_bps": int(float(ball_state.get("confidence") or 0.0) * 10000),
                "x": float(center_xy[0]) if center_xy else None,
                "y": float(center_xy[1]) if center_xy else None,
                "missing_gap_frames": int(ball_state.get("missing_gap_frames") or 0),
                "candidate_count": int(ball_state.get("candidate_count") or 0),
                "candidate_scores": ball_state.get("candidate_scores") or [],
                "selected_candidate": ball_state.get("selected_candidate"),
                "rejected_candidates": ball_state.get("rejected_candidates") or [],
                "source": ball_state.get("source"),
                "bootstrap_segment_id": int(bootstrap_segment_id) if bootstrap_segment_id is not None else None,
                "bootstrap_enabled": bool(bootstrap_context and bootstrap_context.get("enabled")),
                "play_region_prior": round(
                    play_region_prior_for_point(bootstrap_context, center_xy[0], center_xy[1]),
                    4,
                ) if center_xy is not None else 0.0,
                "geometry_region_ok": bool(
                    geometry_evidence_gate(
                        bootstrap_context,
                        center_xy[0],
                        center_xy[1],
                        min_prior=play_region_geometry_min_prior,
                    )["geometry_region_ok"]
                ) if center_xy is not None else False,
                "search_plan": ball_search_plan,
            }
        )
        return ball_state, updated_ball_2d, ball_3d


class GameDNAExtractor:
    """Orchestrate one full game-DNA extraction run for a single video.

    This class owns the mutable per-run state:
    - frame counter
    - live track table
    - current possession engine
    - last observed ball center

    The goal is to make the frame pipeline explicit while keeping the original
    output contract and high-level behavior intact.
    """

    def __init__(self, config, models, calibration):
        self.config = config
        self.models = models
        self.calibration = calibration
        self.frame_idx = 0
        self.tracks = {}
        self.possession_engine = PossessionEngine()
        self.mvp_event_adapter = MvpEventAdapter()
        self.mvp_stat_accumulator = MvpStatAccumulator()
        self.tracklet_store = TrackletStore(
            reid_extractor=ColorHistogramReIDExtractor(),
            reid_sample_interval_frames=self.config.reid_sample_interval_frames,
        )
        self.ball_search_scheduler = BallSearchScheduler(
            BallSearchSchedulerConfig(
                full_frame_interval_frames=self.config.ball_full_frame_interval_frames,
                missing_full_frame_interval_frames=self.config.ball_missing_full_frame_interval_frames,
                roi_search_enabled=self.config.ball_roi_search_enabled,
                max_roi_count=self.config.ball_roi_max_count,
            )
        )
        self.ball_tracker = BallStateTracker()
        self.ball_state = None
        self.last_ball_2d = np.array([0.0, 0.0])
        self.last_shot_attempt_t_ms_by_track = {}
        self.bootstrap_state = RuntimeBootstrapState()
        self.discontinuity_detector = RuntimeDiscontinuityDetector(
            threshold=self.config.bootstrap_discontinuity_threshold
        )
        self.pan_drift_detector = RuntimePanDriftDetector()
        self.bootstrapper = None
        self.last_bootstrap_frame_idx = None
        self.court_pose_tracker = CourtPoseTracker()

    def run(self):
        """Drive the frame loop and stream JSONL output for the whole video."""
        import cv2

        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, self.config.output_filename)
        capture = cv2.VideoCapture(self.config.video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        with JsonlEventWriter(output_path) as writer:
            writer.write(
                {
                    "kind": "resource_context",
                    "device": str(self.models.device),
                    "resource_policy": (
                        self.models.resource_policy.to_payload()
                        if self.models.resource_policy is not None
                        else None
                    ),
                }
            )
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break
                self.frame_idx += 1
                # All downstream outputs are indexed by the same frame clock.
                t_ms = int(self.frame_idx / fps * 1000)
                h_matrix = self.calibration.homography_for_frame(self.frame_idx)
                self._process_frame(frame, t_ms, h_matrix, writer)
            writer.write(
                self.mvp_stat_accumulator.terminal_game_snapshot(
                    game_id=Path(self.config.video_path).stem,
                    t_ms=int(self.frame_idx / fps * 1000) if self.frame_idx else 0,
                )
            )
        capture.release()
        print(f"[INFO] Complete. Output: {output_path}")

    def _process_frame(self, frame, t_ms, h_matrix, writer):
        """Run one frame through perception, tracking, logic, and writing."""
        self._maybe_refresh_bootstrap_context(frame, t_ms, writer)
        player_track_kwargs = {
            "persist": True,
            "classes": [PERSON_CLASS_ID],
            "conf": self.config.player_conf_threshold,
            "device": self.models.device,
            "verbose": False,
        }
        if self.config.player_tracker_backend != "default":
            player_track_kwargs["tracker"] = self.config.player_tracker_backend
        player_result = self.models.player_model.track(frame, **player_track_kwargs)
        pose_result = self.models.pose_model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.config.pose_conf_threshold,
            device=self.models.device,
            verbose=False,
        )
        if not FrameResultAdapter.has_track_ids(player_result):
            self._write_court_pose_event(frame, t_ms, h_matrix, [], writer)
            ball_boxes_xywh, ball_cls, ball_conf, ball_search_plan, ball_candidate_metadata = self._run_ball_search(frame, [])
            self.ball_state, self.last_ball_2d, _ = FrameResultAdapter.extract_ball_state(
                ball_boxes_xywh,
                ball_cls,
                ball_conf,
                self.ball_tracker,
                h_matrix,
                t_ms,
                writer,
                bootstrap_context=self.bootstrap_state.context,
                bootstrap_segment_id=self.bootstrap_state.segment_id,
                play_region_geometry_min_prior=self.config.play_region_geometry_min_prior,
                ball_search_plan=ball_search_plan,
                ball_candidate_metadata=ball_candidate_metadata,
            )
            return

        boxes_xywh, cls, conf, tids, classes = FrameResultAdapter.extract_tracked_arrays(player_result)
        keypoints = FrameResultAdapter.match_pose_keypoints(player_result, pose_result)
        self._maybe_refresh_bootstrap_context_after_tracking(
            frame,
            t_ms,
            writer,
            boxes_xywh,
            tids,
            classes,
        )
        ball_search_players = self._ball_search_player_detections(boxes_xywh, tids, classes, keypoints, frame.shape)
        self._write_court_pose_event(frame, t_ms, h_matrix, ball_search_players, writer)
        ball_boxes_xywh, ball_cls, ball_conf, ball_search_plan, ball_candidate_metadata = self._run_ball_search(frame, ball_search_players)
        # Ball state is updated before player features are built so Action Brain
        # sees the freshest available ball position for this frame.
        self.ball_state, self.last_ball_2d, ball_3d = FrameResultAdapter.extract_ball_state(
            ball_boxes_xywh,
            ball_cls,
            ball_conf,
            self.ball_tracker,
            h_matrix,
            t_ms,
            writer,
            bootstrap_context=self.bootstrap_state.context,
            bootstrap_segment_id=self.bootstrap_state.segment_id,
            play_region_geometry_min_prior=self.config.play_region_geometry_min_prior,
            ball_search_plan=ball_search_plan,
            ball_candidate_metadata=ball_candidate_metadata,
        )
        player_map = self._update_tracks(boxes_xywh, conf, tids, classes, keypoints, h_matrix, frame)
        self._write_possession_events(player_map, ball_3d, t_ms, writer)
        self._write_player_events(player_map, h_matrix, t_ms, writer)

    def _ball_search_player_detections(self, boxes_xywh, tids, classes, keypoints=None, frame_shape=None):
        players = []
        frame_height = float(frame_shape[0]) if frame_shape is not None else 1.0
        frame_width = float(frame_shape[1]) if frame_shape is not None else 1.0
        for idx, track_id in enumerate(tids):
            if int(classes[idx]) != PERSON_CLASS_ID:
                continue
            keypoints_xy = None
            if keypoints is not None and idx < len(keypoints) and keypoints[idx] is not None:
                keypoint_entry = keypoints[idx]
                keypoint_list = keypoint_entry.tolist() if hasattr(keypoint_entry, "tolist") else keypoint_entry
                keypoints_xy = [
                    [float(point[0]) * frame_width, float(point[1]) * frame_height]
                    if point is not None and len(point) >= 2
                    else [0.0, 0.0]
                    for point in keypoint_list
                ]
                keypoints_conf = [
                    1.0
                    if point is not None and len(point) >= 2 and (float(point[0]) > 0.0 or float(point[1]) > 0.0)
                    else 0.0
                    for point in keypoint_list
                ]
            else:
                keypoints_conf = None
            players.append(
                {
                    "track_id": int(track_id),
                    "bbox_xywh": [float(v) for v in boxes_xywh[idx]],
                    "bbox_xyxy": self._bbox_xywh_to_xyxy(boxes_xywh[idx]),
                    "class_id": PERSON_CLASS_ID,
                    "keypoints_xy": keypoints_xy,
                    "keypoints_conf": keypoints_conf,
                }
            )
        return players

    @staticmethod
    def _bbox_xywh_to_xyxy(bbox_xywh):
        cx, cy, width, height = [float(v) for v in bbox_xywh]
        return [
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ]

    def _runtime_scene_prior(self):
        context = self.bootstrap_state.context or {}
        return {
            "prior_status": "ready" if context.get("enabled") and context.get("status") == "ready" else "inactive",
            "region_mask_shape": context.get("mask_shape"),
            "region_mask_grid": context.get("mask_grid"),
            "source_backend": context.get("backend"),
            "source_model": context.get("model_name"),
            "trigger_reason": context.get("reason"),
        }

    def _write_court_pose_event(self, frame, t_ms, h_matrix, player_detections, writer):
        payload_frame = {
            "frame_idx": int(self.frame_idx),
            "_h_matrix": h_matrix,
            "scene_prior": self._runtime_scene_prior(),
            "detections": player_detections,
        }
        pose = self.court_pose_tracker.update(
            payload_frame,
            video_meta={"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        )
        writer.write(
            {
                **pose.to_payload(),
                "kind": "court_pose",
                "t_ms": int(t_ms),
                "bootstrap_segment_id": int(self.bootstrap_state.segment_id),
            }
        )

    def _run_ball_search(self, frame, player_detections):
        plan = self.ball_search_scheduler.plan(
            frame.shape,
            frame_idx=int(self.frame_idx),
            ball_state=self.ball_state,
            player_detections=player_detections,
        )
        boxes_xywh = []
        classes = []
        confidences = []
        candidate_metadata = []
        if plan.run_full_frame:
            ball_result = self.models.ball_model.predict(
                frame,
                classes=[BALL_CLASS_ID],
                conf=0.05,
                device=self.models.device,
                verbose=False,
            )
            self._extend_ball_arrays(
                boxes_xywh,
                classes,
                confidences,
                candidate_metadata,
                ball_result,
                source="full_frame",
            )
        if plan.rois and not boxes_xywh:
            crops = []
            offsets = []
            roi_records = []
            for roi in plan.rois:
                x1, y1, x2, y2 = roi.bbox_xyxy
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    continue
                crops.append(crop)
                offsets.append((x1, y1))
                roi_records.append(roi)
            if crops:
                roi_results = self.models.ball_model.predict(
                    crops,
                    classes=[BALL_CLASS_ID],
                    conf=0.05,
                    device=self.models.device,
                    verbose=False,
                )
                for result, offset, roi in zip(roi_results, offsets, roi_records):
                    self._extend_ball_arrays(
                        boxes_xywh,
                        classes,
                        confidences,
                        candidate_metadata,
                        [result],
                        offset_xy=offset,
                        source=roi.source,
                        roi=roi,
                    )
        return boxes_xywh, classes, confidences, plan.to_payload(), candidate_metadata

    @staticmethod
    def _extend_ball_arrays(boxes_xywh, classes, confidences, candidate_metadata, result, *, offset_xy=(0, 0), source="unknown", roi=None):
        detected_boxes, detected_classes, detected_confidences = FrameResultAdapter.extract_ball_arrays(result)
        offset_x, offset_y = [float(v) for v in offset_xy]
        for bbox_xywh, class_id, confidence in zip(detected_boxes, detected_classes, detected_confidences):
            adjusted = np.array(bbox_xywh, dtype=np.float32).copy()
            adjusted[0] += offset_x
            adjusted[1] += offset_y
            boxes_xywh.append(adjusted)
            classes.append(int(class_id))
            confidences.append(float(confidence))
            candidate_metadata.append(
                {
                    "source": source,
                    "roi_bbox_xyxy": list(roi.bbox_xyxy) if roi is not None else None,
                    "track_id": roi.track_id if roi is not None else None,
                    "motion_mode": roi.motion_mode if roi is not None else None,
                }
            )

    def _maybe_refresh_bootstrap_context(self, frame, t_ms, writer):
        """Run the optional bootstrap pre-pass only on the first frame or after cuts."""
        discontinuity = self.discontinuity_detector.update(frame)
        reason = None
        if self.bootstrap_state.context is None:
            reason = "initial"
        elif discontinuity["triggered"]:
            reason = "discontinuity"
        if reason is None:
            return
        self.bootstrap_state.segment_id += 1
        self.bootstrap_state.context = self._run_bootstrap(frame, t_ms, reason, discontinuity, writer)

    def _run_bootstrap(self, frame, t_ms, reason, discontinuity, writer):
        payload = {
            "enabled": False,
            "status": "disabled",
            "backend": self.config.bootstrap_foreground_backend,
            "model_name": self.config.bootstrap_foreground_model,
            "frame_idx": int(self.frame_idx),
        }
        if self.config.bootstrap_foreground_backend == "grounding_dino":
            if GroundingDinoBootstrapper is None:
                payload["status"] = "bootstrap_unavailable"
            elif self.bootstrapper is None:
                self.bootstrapper = GroundingDinoBootstrapper(
                    model_name=self.config.bootstrap_foreground_model,
                    text_prompt=self.config.bootstrap_foreground_prompt,
                    device=str(self.models.device),
                )
            if self.bootstrapper is not None:
                payload = self.bootstrapper.run_on_frame(frame, frame_idx=self.frame_idx).to_payload()
        payload["segment_id"] = int(self.bootstrap_state.segment_id)
        payload["reason"] = reason
        payload["discontinuity_score"] = round(float(discontinuity.get("score") or 0.0), 4)
        payload["source_frame_idx"] = int(self.frame_idx)
        self.last_bootstrap_frame_idx = int(self.frame_idx)
        writer.write(
            {
                "kind": "bootstrap_context",
                "t_ms": int(t_ms),
                "segment_id": int(self.bootstrap_state.segment_id),
                "reason": reason,
                "enabled": bool(payload.get("enabled")),
                "status": payload.get("status"),
                "backend": payload.get("backend"),
                "foreground_ratio": payload.get("foreground_ratio"),
                "discontinuity_score": payload["discontinuity_score"],
                "source_frame_idx": payload["source_frame_idx"],
            }
        )
        return payload

    def _maybe_refresh_bootstrap_context_after_tracking(
        self,
        frame,
        t_ms,
        writer,
        boxes_xywh,
        tids,
        classes,
    ):
        """Invalidate stale play-region priors when coherent camera pan is detected."""
        pan_drift = self.pan_drift_detector.update(boxes_xywh, tids, classes)
        if not self.bootstrap_state.context or not pan_drift["triggered"]:
            return
        cooldown = int(self.config.bootstrap_pan_recompute_cooldown_frames)
        if (
            self.last_bootstrap_frame_idx is not None
            and (self.frame_idx - self.last_bootstrap_frame_idx) < cooldown
        ):
            self.bootstrap_state.context["bootstrap_stale"] = True
            self.bootstrap_state.context["pan_drift"] = pan_drift
            return
        self.bootstrap_state.segment_id += 1
        self.bootstrap_state.context = self._run_bootstrap(
            frame,
            t_ms,
            "pan_drift",
            {"score": pan_drift["median_displacement_px"]},
            writer,
        )
        self.bootstrap_state.context["pan_drift"] = pan_drift

    def _update_tracks(self, boxes_xywh, conf, tids, classes, keypoints, h_matrix, frame):
        """Project and update all player tracks visible in the current frame."""
        player_map = {}
        visible_track_ids = []
        for idx, tid in enumerate(tids):
            if classes[idx] != PERSON_CLASS_ID:
                continue
            visible_track_ids.append(int(tid))
            # Track objects are created lazily so stable tracker ids become the
            # runtime identity handle for subsequent smoothing and history use.
            track = self.tracks.setdefault(tid, TrackManager(tid))
            geometry_gate = geometry_evidence_gate(
                self.bootstrap_state.context,
                float(boxes_xywh[idx][0]),
                float(boxes_xywh[idx][1]),
                min_prior=self.config.play_region_geometry_min_prior,
            )
            court_xy = project_pixel_to_court(
                float(boxes_xywh[idx][0]), float(boxes_xywh[idx][1]), h_matrix
            )
            track.update_position(court_xy[0], court_xy[1])
            track.play_region_prior = float(geometry_gate["play_region_prior"])
            track.geometry_region_ok = bool(geometry_gate["geometry_region_ok"])
            tracklet = self.tracklet_store.update(
                track_id=int(tid),
                frame_idx=int(self.frame_idx),
                bbox_xywh=boxes_xywh[idx],
                court_xy=court_xy,
                confidence=float(conf[idx]),
                frame_bgr=frame,
            )
            current_keypoints = None
            if keypoints is not None:
                keypoint_entry = keypoints[idx]
                current_keypoints = keypoint_entry.tolist() if hasattr(keypoint_entry, "tolist") else keypoint_entry
            track.add_keypoints(current_keypoints, h_matrix)
            player_map[tid] = {
                "pos_3d": track.pos_3d,
                "team": track.team,
                "play_region_prior": track.play_region_prior,
                "geometry_region_ok": track.geometry_region_ok,
                "tracklet": tracklet.to_payload(),
            }
        self.tracklet_store.mark_missing_except(visible_track_ids, frame_idx=int(self.frame_idx))
        return player_map

    def _write_possession_events(self, player_map, ball_3d, t_ms, writer):
        """Ask the possession engine for newly triggered events and stream them."""
        possession_events = self.possession_engine.update(player_map, ball_3d, t_ms)
        for event in possession_events:
            writer.write(event)
            for attributed_event in self.mvp_event_adapter.adapt(event, player_map):
                writer.write(attributed_event)
                stat_update = self.mvp_stat_accumulator.apply_attributed_event(attributed_event)
                if stat_update is not None:
                    writer.write(stat_update)
                    stat_snapshot = self.mvp_stat_accumulator.snapshot_for_player(
                        stat_update["player_id"],
                        team_id=stat_update.get("team_id"),
                        t_ms=stat_update.get("t_ms"),
                    )
                    if stat_snapshot is not None:
                        writer.write(stat_snapshot)

    def _write_player_events(self, player_map, h_matrix, t_ms, writer):
        """Emit one player-state row per currently visible tracked player."""
        for tid in player_map.keys():
            track = self.tracks[tid]
            learned_label = self._infer_action_label(track, h_matrix)
            has_possession = self.possession_engine.current_handler == tid
            track.state_machine.update(
                track.kpt_history,
                learned_label=learned_label,
                context={"has_possession": has_possession},
            )
            writer.write(
                {
                    "kind": "player",
                    "track_id": int(tid),
                    "t_ms": t_ms,
                    "action": track.state_machine.get_label(),
                    "court_x": track.court_x,
                    "court_y": track.court_y,
                    "bootstrap_segment_id": int(self.bootstrap_state.segment_id),
                    "play_region_prior": round(float(track.play_region_prior), 4),
                    "geometry_region_ok": bool(track.geometry_region_ok),
                    "runtime_identity": {
                        "kind": "tracklet_store_v1",
                        "tracklet_id": int(tid),
                        "final_identity_committed": False,
                        "tracklet": player_map[tid].get("tracklet"),
                    },
                }
            )
            shot_attempt = self._maybe_build_shot_attempt_candidate(
                tid=tid,
                learned_label=learned_label,
                has_possession=has_possession,
                t_ms=t_ms,
            )
            if shot_attempt is not None:
                writer.write(shot_attempt)
                stat_update = self.mvp_stat_accumulator.apply_attributed_event(shot_attempt)
                if stat_update is not None:
                    writer.write(stat_update)
                    stat_snapshot = self.mvp_stat_accumulator.snapshot_for_player(
                        stat_update["player_id"],
                        team_id=stat_update.get("team_id"),
                        t_ms=stat_update.get("t_ms"),
                    )
                    if stat_snapshot is not None:
                        writer.write(stat_snapshot)

    def _maybe_build_shot_attempt_candidate(self, *, tid, learned_label, has_possession, t_ms):
        """Emit a conservative unresolved shot-attempt payload when evidence aligns.

        This path does not overclaim made/missed outcomes. It only surfaces a
        shot-attempt candidate in MVP event shape when the current runtime has:
        - the same player as current ballhandler
        - a shot-like Action Brain / declarative label
        - recent ball visibility

        The deterministic rule engine will explicitly report any remaining
        unsatisfied constraints such as missing shot value.
        """
        if learned_label != "jump_shot":
            return None
        if not has_possession:
            return None
        last_emitted_t_ms = self.last_shot_attempt_t_ms_by_track.get(tid)
        if last_emitted_t_ms is not None and (t_ms - last_emitted_t_ms) < 1200:
            return None
        ball_visible = self.ball_state is not None and self.ball_state.get("state") != "missing"
        evidence = {
            "ball_release": ball_visible,
            "action_label": learned_label,
        }
        payload = self.mvp_event_adapter.build_shot_attempt_candidate(
            track_id=tid,
            team_id=self.tracks[tid].team,
            t_ms=t_ms,
            evidence=evidence,
            shot_value=None,
        )
        payload["candidate_only"] = True
        self.last_shot_attempt_t_ms_by_track[tid] = t_ms
        return payload

    def _infer_action_label(self, track, h_matrix):
        """Run Action Brain only when enough pose history is available."""
        if not track.is_ready() or self.models.brain is None:
            return None
        import torch

        player_court_pos = np.array([track.court_x, track.court_y])
        feature_tensor = construct_features_v2(
            track.kpt_history,
            self.last_ball_2d,
            h_matrix,
            player_court_pos,
            None,
            None,
            self.models.device,
        )
        with torch.no_grad():
            output = self.models.brain(feature_tensor)
            return LABEL_MAP_INV[int(torch.argmax(output))]


def resolve_video_path(video_path):
    """Resolve the requested video path, falling back to `hoops_config.yaml`."""
    if video_path is not None:
        return video_path
    with open("hoops_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get("local_video_path", "data/sample.mp4")


def extract_game_dna(
    video_path=None,
    output_dir="data",
    smoke_test=False,
    bootstrap_foreground_backend="none",
    bootstrap_foreground_model=DEFAULT_GROUNDING_DINO_MODEL,
    bootstrap_foreground_prompt=DEFAULT_GROUNDING_DINO_PROMPT,
):
    """Backward-compatible entrypoint for the current inference pipeline.

    The refactor keeps this function as the public API and CLI target, but the
    actual work is delegated to focused helpers so the control flow is readable:
    config -> models -> calibration -> extractor -> JSONL output.
    """
    resolved_video_path = resolve_video_path(video_path)
    config = InferenceConfig(
        video_path=resolved_video_path,
        output_dir=output_dir,
        smoke_test=smoke_test,
        bootstrap_foreground_backend=bootstrap_foreground_backend,
        bootstrap_foreground_model=bootstrap_foreground_model,
        bootstrap_foreground_prompt=bootstrap_foreground_prompt,
    )
    models = ModelBundle(
        config.player_model_name,
        config.pose_model_name,
        config.ball_model_name,
        config.brain_path,
        LABEL_MAP_INV,
        requested_device=config.device,
    ).load()
    if config.smoke_test:
        models.smoke_test()
        return

    clip_id = Path(config.video_path).stem
    calibration = CalibrationResolver(
        config.calibration_path,
        config.fallback_calibration_path,
    ).load_for_clip(clip_id)
    extractor = GameDNAExtractor(config, models, calibration)
    extractor.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HoopSense Inference Pipeline")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input video file")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick model initialization check")
    parser.add_argument("--output-dir", default="data", help="Output directory for results")
    parser.add_argument("--player-model", default="yolov8n.pt", help="Detection model used for player proposals")
    parser.add_argument(
        "--player-tracker-backend",
        default="default",
        help="Ultralytics tracker backend for player tracking: default, bytetrack.yaml, or botsort.yaml",
    )
    parser.add_argument("--player-conf", type=float, default=0.12, help="Player proposal confidence threshold")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Pose extraction confidence threshold")
    parser.add_argument(
        "--reid-sample-interval-frames",
        type=int,
        default=15,
        help="Frame cadence for cheap runtime ReID evidence extraction on active tracklets",
    )
    parser.add_argument(
        "--ball-full-frame-interval-frames",
        type=int,
        default=1,
        help="Run full-frame ball detection every N frames; 1 preserves every-frame detection",
    )
    parser.add_argument(
        "--ball-missing-full-frame-interval-frames",
        type=int,
        default=1,
        help="Run full-frame ball detection every N frames while no recent ball exists; 1 preserves every-frame reacquisition",
    )
    parser.add_argument(
        "--ball-roi-search",
        action="store_true",
        help="Enable ROI ball search around last ball state and nearby players between full-frame scans",
    )
    parser.add_argument(
        "--ball-roi-max-count",
        type=int,
        default=4,
        help="Maximum number of runtime ball-search ROIs per frame",
    )
    parser.add_argument(
        "--bootstrap-foreground-backend",
        choices=["none", "grounding_dino"],
        default="none",
        help="Optional runtime bootstrap foreground prior backend",
    )
    parser.add_argument(
        "--bootstrap-foreground-model",
        default=DEFAULT_GROUNDING_DINO_MODEL,
        help="Model name for runtime Grounding DINO bootstrap pre-pass",
    )
    parser.add_argument(
        "--bootstrap-foreground-prompt",
        default=DEFAULT_GROUNDING_DINO_PROMPT,
        help="Prompt text for runtime Grounding DINO bootstrap pre-pass",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Runtime device selector: auto, cpu, cuda, or cuda:0",
    )
    args = parser.parse_args()
    config_override = InferenceConfig(
        video_path=resolve_video_path(args.video_path),
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        player_model_name=args.player_model,
        player_tracker_backend=args.player_tracker_backend,
        player_conf_threshold=args.player_conf,
        pose_conf_threshold=args.pose_conf,
        reid_sample_interval_frames=args.reid_sample_interval_frames,
        ball_full_frame_interval_frames=args.ball_full_frame_interval_frames,
        ball_missing_full_frame_interval_frames=args.ball_missing_full_frame_interval_frames,
        ball_roi_search_enabled=args.ball_roi_search,
        ball_roi_max_count=args.ball_roi_max_count,
        bootstrap_foreground_backend=args.bootstrap_foreground_backend,
        bootstrap_foreground_model=args.bootstrap_foreground_model,
        bootstrap_foreground_prompt=args.bootstrap_foreground_prompt,
        device=args.device,
    )
    models = ModelBundle(
        config_override.player_model_name,
        config_override.pose_model_name,
        config_override.ball_model_name,
        config_override.brain_path,
        LABEL_MAP_INV,
        requested_device=config_override.device,
    ).load()
    if config_override.smoke_test:
        models.smoke_test()
    else:
        clip_id = Path(config_override.video_path).stem
        calibration = CalibrationResolver(
            config_override.calibration_path,
            config_override.fallback_calibration_path,
        ).load_for_clip(clip_id)
        extractor = GameDNAExtractor(config_override, models, calibration)
        extractor.run()

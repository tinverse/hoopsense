import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from pipelines.behavior_engine import BehaviorStateMachine, PossessionEngine
from pipelines.geometry import lift_keypoints_to_3d, project_pixel_to_court
from pipelines.mvp_event_engine import MvpEventRuleEngine


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


class KalmanFilter:
    """Minimal 1D Kalman filter used to smooth court-space x/y coordinates.

    This is intentionally tiny: the legacy inference path only smooths the
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
    pose_model_name: str = "yolov8n-pose.pt"
    calibration_path: str = "data/training/camera_calibration.json"
    legacy_calibration_path: str = "data/calibration.json"
    brain_path: str = "data/models/action_brain.pt"
    output_filename: str = "intelligent_game_dna.jsonl"


class CalibrationResolver:
    """Resolve clip-specific homography data for the current video.

    The legacy inference path supports two calibration storage shapes:
    - per-clip data under `data/training/camera_calibration.json`
    - a fallback single global matrix under `data/calibration.json`

    The resolver normalizes both into a single `homography_for_frame(...)`
    interface.
    """

    def __init__(self, calibration_path, legacy_calibration_path):
        self.calibration_path = Path(calibration_path)
        self.legacy_calibration_path = Path(legacy_calibration_path)
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
        elif self.legacy_calibration_path.exists():
            with open(self.legacy_calibration_path, "r") as f:
                legacy = json.load(f)
            self.global_h = np.array(legacy.get("h_matrix", np.eye(3)))
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

    def __init__(self, pose_model_name, brain_path, label_map):
        self.pose_model_name = pose_model_name
        self.brain_path = brain_path
        self.label_map = label_map
        self.device = None
        self.pose_model = None
        self.brain = None

    def load(self):
        """Load YOLO unconditionally and Action Brain only if a checkpoint exists."""
        import torch
        from ultralytics import YOLO
        from core.vision.action_brain import ActionBrain

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_model = YOLO(self.pose_model_name)
        if os.path.exists(self.brain_path):
            brain = ActionBrain(num_classes=len(self.label_map)).to(self.device)
            brain.load_state_dict(torch.load(self.brain_path, map_location=self.device))
            brain.eval()
            self.brain = brain
        return self

    def smoke_test(self):
        """Exercise YOLO initialization without running the full pipeline."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.pose_model(dummy)


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
    """Translate legacy runtime events into MVP-attribution payloads.

    The legacy inference path already emits simple `pass` / `catch` / `steal`
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
        """Extract the raw arrays needed by the rest of the legacy pipeline."""
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
    def extract_ball_state(boxes_xywh, cls, conf, last_ball_2d, h_matrix, t_ms, writer):
        """Update the last-seen ball state and emit the corresponding JSONL rows.

        This preserves existing behavior: we treat every class-32 detection as a
        fresh frame-level observation and keep only the most recent projected
        center as the input to downstream player features.
        """
        ball_3d = None
        updated_ball_2d = last_ball_2d
        for bbox_xywh, class_id, confidence in zip(boxes_xywh, cls, conf):
            if int(class_id) != BALL_CLASS_ID:
                continue
            updated_ball_2d = np.array([float(bbox_xywh[0]), float(bbox_xywh[1])])
            ball_court_xy = project_pixel_to_court(
                updated_ball_2d[0], updated_ball_2d[1], h_matrix
            )
            ball_3d = np.array([ball_court_xy[0], ball_court_xy[1], BALL_DEFAULT_Z])
            writer.write(
                {
                    "kind": "ball",
                    "t_ms": t_ms,
                    "x": float(bbox_xywh[0]),
                    "y": float(bbox_xywh[1]),
                    "confidence_bps": int(confidence * 10000),
                }
            )
        return updated_ball_2d, ball_3d


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
        self.last_ball_2d = np.array([0.0, 0.0])

    def run(self):
        """Drive the frame loop and stream JSONL output for the whole video."""
        import cv2

        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, self.config.output_filename)
        capture = cv2.VideoCapture(self.config.video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        with JsonlEventWriter(output_path) as writer:
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break
                self.frame_idx += 1
                # All downstream outputs are indexed by the same frame clock.
                t_ms = int(self.frame_idx / fps * 1000)
                h_matrix = self.calibration.homography_for_frame(self.frame_idx)
                self._process_frame(frame, t_ms, h_matrix, writer)
        capture.release()
        print(f"[INFO] Complete. Output: {output_path}")

    def _process_frame(self, frame, t_ms, h_matrix, writer):
        """Run one frame through perception, tracking, logic, and writing."""
        result = self.models.pose_model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
            verbose=False,
        )
        if not FrameResultAdapter.has_track_ids(result):
            return

        boxes_xywh, cls, conf, tids, classes, keypoints = FrameResultAdapter.extract_arrays(result)
        # Ball state is updated before player features are built so Action Brain
        # sees the freshest available ball position for this frame.
        self.last_ball_2d, ball_3d = FrameResultAdapter.extract_ball_state(
            boxes_xywh, cls, conf, self.last_ball_2d, h_matrix, t_ms, writer
        )
        player_map = self._update_tracks(boxes_xywh, tids, classes, keypoints, h_matrix)
        self._write_possession_events(player_map, ball_3d, t_ms, writer)
        self._write_player_events(player_map, h_matrix, t_ms, writer)

    def _update_tracks(self, boxes_xywh, tids, classes, keypoints, h_matrix):
        """Project and update all player tracks visible in the current frame."""
        player_map = {}
        for idx, tid in enumerate(tids):
            if classes[idx] != PERSON_CLASS_ID:
                continue
            # Track objects are created lazily so stable tracker ids become the
            # runtime identity handle for subsequent smoothing and history use.
            track = self.tracks.setdefault(tid, TrackManager(tid))
            court_xy = project_pixel_to_court(
                float(boxes_xywh[idx][0]), float(boxes_xywh[idx][1]), h_matrix
            )
            track.update_position(court_xy[0], court_xy[1])
            current_keypoints = keypoints[idx].tolist() if keypoints is not None else None
            track.add_keypoints(current_keypoints, h_matrix)
            player_map[tid] = {"pos_3d": track.pos_3d, "team": track.team}
        return player_map

    def _write_possession_events(self, player_map, ball_3d, t_ms, writer):
        """Ask the possession engine for newly triggered events and stream them."""
        possession_events = self.possession_engine.update(player_map, ball_3d, t_ms)
        for event in possession_events:
            writer.write(event)
            for attributed_event in self.mvp_event_adapter.adapt(event, player_map):
                writer.write(attributed_event)

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
                }
            )

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


def extract_game_dna(video_path=None, output_dir="data", smoke_test=False):
    """Backward-compatible entrypoint for the legacy inference pipeline.

    The refactor keeps this function as the public API and CLI target, but the
    actual work is delegated to focused helpers so the control flow is readable:
    config -> models -> calibration -> extractor -> JSONL output.
    """
    resolved_video_path = resolve_video_path(video_path)
    config = InferenceConfig(
        video_path=resolved_video_path,
        output_dir=output_dir,
        smoke_test=smoke_test,
    )
    models = ModelBundle(config.pose_model_name, config.brain_path, LABEL_MAP_INV).load()
    if config.smoke_test:
        models.smoke_test()
        return

    clip_id = Path(config.video_path).stem
    calibration = CalibrationResolver(
        config.calibration_path,
        config.legacy_calibration_path,
    ).load_for_clip(clip_id)
    extractor = GameDNAExtractor(config, models, calibration)
    extractor.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HoopSense Inference Pipeline")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to input video file")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick model initialization check")
    parser.add_argument("--output-dir", default="data", help="Output directory for results")
    args = parser.parse_args()
    extract_game_dna(
        video_path=args.video_path,
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
    )

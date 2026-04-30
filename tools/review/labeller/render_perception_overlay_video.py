import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp_runs" / "perception_overlays"
DEFAULT_FFMPEG_DOCKER_IMAGE = "hoopsense-orin:sam3-exp1"


def resolve_ffmpeg_binary():
    system_ffmpeg = Path('/usr/bin/ffmpeg')
    if system_ffmpeg.exists() and not system_ffmpeg.is_symlink():
        return str(system_ffmpeg)
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path and '/.guix' not in ffmpeg_path and '/guix-' not in ffmpeg_path:
        return ffmpeg_path
    return None


def build_docker_ffmpeg_command(raw_path, output_path, image=DEFAULT_FFMPEG_DOCKER_IMAGE):
    raw_path = Path(raw_path).resolve()
    output_path = Path(output_path).resolve()
    return [
        'docker',
        'run',
        '--rm',
        '--user',
        f'{os.getuid()}:{os.getgid()}',
        '-v',
        f'{raw_path.parent}:/input',
        '-v',
        f'{output_path.parent}:/output',
        image,
        'bash',
        '-lc',
        (
            'ffmpeg -y '
            f'-i /input/{raw_path.name} '
            '-c:v libx264 -pix_fmt yuv420p -movflags +faststart -an '
            f'/output/{output_path.name}'
        ),
    ]


def resolve_docker_ffmpeg_command(raw_path, output_path, image=DEFAULT_FFMPEG_DOCKER_IMAGE):
    docker_binary = shutil.which('docker')
    if not docker_binary:
        return None
    inspect = subprocess.run(
        [docker_binary, 'image', 'inspect', image],
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        return None
    return build_docker_ffmpeg_command(raw_path, output_path, image=image)


def load_artifact(artifact_path):
    with Path(artifact_path).open() as handle:
        return json.load(handle)


def build_detection_label(detection):
    on_court_candidate = bool(detection.get("on_court_candidate"))
    active_candidate = bool(detection.get("active_player_candidate"))
    track_id = detection.get("track_id")
    class_name = detection.get("class_name") or "det"
    track_label = f"P{track_id}" if track_id is not None else class_name
    identity_track_id = detection.get("identity_track_id")
    identity_label = (
        f"/I{identity_track_id}"
        if identity_track_id is not None and identity_track_id != track_id
        else ""
    )
    jersey_number = detection.get("identity_jersey_number")
    jersey_confidence = detection.get("identity_jersey_number_confidence")
    jersey_label = f" #{jersey_number}" if jersey_number not in (None, "", "unknown") else ""
    jersey_conf_label = (
        f"({int(round(float(jersey_confidence) * 100))})"
        if jersey_label and jersey_confidence is not None
        else ""
    )
    uniform_bucket = detection.get("uniform_bucket")
    uniform_label = f" {str(uniform_bucket).upper()}" if uniform_bucket and uniform_bucket != "unknown" else ""
    conf_label = f"{int(round(float(detection.get('confidence') or 0.0) * 100))}%"
    on_court_score = detection.get("on_court_score")
    on_court_score_label = f" C{int(round(float(on_court_score) * 100))}" if on_court_score is not None else ""
    active_score = detection.get("active_player_score")
    active_score_label = f" A{int(round(float(active_score) * 100))}" if active_score is not None else ""
    motion_speed = detection.get("motion_speed_px")
    motion_label = f" M{float(motion_speed):.1f}" if motion_speed is not None else ""
    repair_label = " SYN" if detection.get("synthesized") else ""
    status_label = " ACTIVE" if active_candidate else (" COURT" if on_court_candidate else " RAW")
    return f"{track_label}{identity_label}{jersey_label}{jersey_conf_label}{uniform_label}{status_label} {conf_label}{on_court_score_label}{active_score_label}{motion_label}{repair_label}".strip()


def frame_summary(frame):
    detections = list(frame.get("detections") or [])
    on_court_count = sum(1 for detection in detections if detection.get("on_court_candidate"))
    active_count = sum(1 for detection in detections if detection.get("active_player_candidate"))
    demoted_count = len(detections) - on_court_count
    ball_detection = frame.get("ball_state") or frame.get("ball_detection") or None
    ball_text = "ball none"
    if ball_detection:
        ball_text = f"ball {ball_detection.get('state') or 'observed'} {int(round(float(ball_detection.get('confidence') or 0.0) * 100))}%"
    return (
        f"f{int(frame.get('frame_idx') or 0)}  "
        f"t={float(frame.get('t_ms') or 0.0)/1000.0:.2f}s  "
        f"det={len(detections)} on={on_court_count} act={active_count} raw={demoted_count}  "
        f"live={frame.get('live_play_label') or 'unknown'} {float(frame.get('live_play_score') or 0.0):.2f}  "
        f"disc={frame.get('discontinuity_label') or 'continuous'} {float(frame.get('discontinuity_score') or 0.0):.2f}  "
        f"seg={frame.get('continuity_segment_id') if frame.get('continuity_segment_id') is not None else 'na'}  "
        f"{ball_text}"
    )


def overlay_mask(frame, mask, color=(0, 190, 220), alpha=0.2):
    if mask is None:
        return frame
    binary = mask.astype(bool)
    if not np.any(binary):
        return frame
    color_arr = np.array(color, dtype=np.float32)
    frame_float = frame.astype(np.float32)
    frame_float[binary] = frame_float[binary] * (1.0 - alpha) + color_arr * alpha
    np.clip(frame_float, 0, 255, out=frame_float)
    return frame_float.astype(np.uint8)


def mask_from_grid(grid, frame_shape):
    if not grid:
        return None
    mask = np.array(grid, dtype=np.uint8)
    if mask.ndim != 2:
        return None
    return cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)


def draw_grounding_context(frame, frame_payload):
    grounding_context = frame_payload.get("grounding_context") or {}
    bootstrap_context = frame_payload.get("bootstrap_context") or {}
    mask_grid = grounding_context.get("yolo_search_region_mask_grid") or bootstrap_context.get("mask_grid")
    mask = mask_from_grid(mask_grid, frame.shape)
    if mask is not None:
        frame = overlay_mask(frame, mask, color=(18, 168, 210), alpha=0.16)

    search_bbox = grounding_context.get("yolo_search_region_bbox_xyxy") or bootstrap_context.get("foreground_bbox_xyxy")
    if search_bbox and len(search_bbox) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in search_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (18, 168, 210), 2)
        cv2.putText(frame, 'GROUNDING', (max(12, x1), max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (215, 250, 255), 2, cv2.LINE_AA)

    return frame


def draw_keypoints(frame, detection):
    keypoints = detection.get("keypoints_xy") or []
    confidences = detection.get("keypoints_conf") or []
    for idx, point in enumerate(keypoints):
        confidence = float(confidences[idx]) if idx < len(confidences) and confidences[idx] is not None else 1.0
        if not point or confidence < 0.15:
            continue
        color = (102, 255, 209) if idx >= 5 else (102, 214, 255)
        cv2.circle(frame, (int(round(point[0])), int(round(point[1]))), 3, color, -1, lineType=cv2.LINE_AA)


def draw_detection(frame, detection):
    bbox = detection.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    on_court_candidate = bool(detection.get("on_court_candidate"))
    active_candidate = bool(detection.get("active_player_candidate"))
    demoted = not on_court_candidate
    synthesized = bool(detection.get("synthesized"))
    candidate_color = (139, 242, 53) if active_candidate else ((255, 210, 105) if on_court_candidate else (112, 143, 255))
    box_color = (102, 209, 255) if synthesized else candidate_color
    line_type = cv2.LINE_4 if demoted else cv2.LINE_AA
    thickness = 2 if demoted else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, lineType=line_type)
    if demoted:
        for dash_start in range(x1, x2, 18):
            cv2.line(frame, (dash_start, y1), (min(dash_start + 10, x2), y1), box_color, 2)
            cv2.line(frame, (dash_start, y2), (min(dash_start + 10, x2), y2), box_color, 2)
        for dash_start in range(y1, y2, 18):
            cv2.line(frame, (x1, dash_start), (x1, min(dash_start + 10, y2)), box_color, 2)
            cv2.line(frame, (x2, dash_start), (x2, min(dash_start + 10, y2)), box_color, 2)

    label = build_detection_label(detection)
    label_width = max(120, 10 * len(label) + 20)
    label_height = 24
    label_x = max(10, min(frame.shape[1] - label_width - 10, x1))
    label_y = max(10, y1 - label_height - 6)
    fill = (48, 24, 19) if demoted else (15, 39, 48)
    text_color = (200, 245, 255) if not demoted else (200, 213, 255)
    cv2.rectangle(frame, (label_x, label_y), (label_x + label_width, label_y + label_height), fill, -1)
    cv2.rectangle(frame, (label_x, label_y), (label_x + label_width, label_y + label_height), candidate_color, 2)
    cv2.putText(frame, label, (label_x + 8, label_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    draw_keypoints(frame, detection)


def draw_ball(frame, ball_detection):
    if not ball_detection:
        return
    bbox = ball_detection.get("bbox_xyxy")
    center = ball_detection.get("center_xy")
    if center is None and bbox and len(bbox) == 4:
        center = [0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))]
    if center is None:
        return
    cx, cy = int(round(center[0])), int(round(center[1]))
    radius = 10
    if bbox and len(bbox) == 4:
        radius = max(7, min(18, int(round(max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) / 2.0))))
    cv2.circle(frame, (cx, cy), radius, (0, 176, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius - 2, (0, 176, 255), 1, lineType=cv2.LINE_AA)
    state_label = 'PRED' if ball_detection.get('state') == 'predicted_short_gap' else 'BALL'
    label = f"{state_label} {int(round(float(ball_detection.get('confidence') or 0.0) * 100))}%"
    label_width = max(92, len(label) * 10 + 12)
    label_height = 22
    y1 = int(round(float(bbox[1]))) if bbox and len(bbox) == 4 else cy - radius
    label_x = max(10, min(frame.shape[1] - label_width - 10, cx - (label_width // 2)))
    label_y = max(10, y1 - label_height - 6)
    cv2.rectangle(frame, (label_x, label_y), (label_x + label_width, label_y + label_height), (28, 22, 0), -1)
    cv2.rectangle(frame, (label_x, label_y), (label_x + label_width, label_y + label_height), (102, 209, 255), 2)
    cv2.putText(frame, label, (label_x + 8, label_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 242, 255), 1, cv2.LINE_AA)


def draw_header(frame, text):
    cv2.rectangle(frame, (12, 12), (min(frame.shape[1] - 12, 1260), 54), (18, 18, 18), -1)
    cv2.putText(frame, text[:180], (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)


def render_overlay_video(video_path, artifact_path, output_path, start_frame=0, max_frames=None):
    artifact = load_artifact(artifact_path)
    frame_payloads = {int(frame.get("frame_idx") or idx): frame for idx, frame in enumerate(artifact.get("frames") or [])}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine frame size for video: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='perception_overlay_') as tmpdir:
        raw_path = Path(tmpdir) / 'overlay_raw.mp4'
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open temp writer: {raw_path}")

        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if max_frames is not None and written >= max_frames:
                break

            payload = frame_payloads.get(frame_idx)
            if payload:
                frame = draw_grounding_context(frame, payload)
                for detection in payload.get("detections") or []:
                    draw_detection(frame, detection)
                draw_ball(frame, payload.get("ball_state") or payload.get("ball_detection"))
                draw_header(frame, frame_summary(payload))
            else:
                draw_header(frame, f"f{frame_idx}  no artifact frame payload")
            writer.write(frame)
            written += 1
            frame_idx += 1

        writer.release()
        cap.release()

        ffmpeg_binary = resolve_ffmpeg_binary()
        docker_ffmpeg_cmd = resolve_docker_ffmpeg_command(raw_path, output_path)
        transcode_mode = 'raw_mp4v_fallback'
        if ffmpeg_binary:
            cmd = [
                ffmpeg_binary, '-y', '-i', str(raw_path),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an',
                str(output_path),
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode == 0:
                transcode_mode = 'h264_faststart'
            else:
                shutil.copy2(raw_path, output_path)
                transcode_mode = 'raw_mp4v_fallback'
        elif docker_ffmpeg_cmd:
            completed = subprocess.run(docker_ffmpeg_cmd, capture_output=True, text=True)
            if completed.returncode == 0:
                transcode_mode = 'h264_faststart_docker'
            else:
                shutil.copy2(raw_path, output_path)
                transcode_mode = 'raw_mp4v_fallback'
        else:
            shutil.copy2(raw_path, output_path)

    return {
        'output_path': str(output_path),
        'frames_rendered': int(written),
        'fps': float(fps),
        'width': int(width),
        'height': int(height),
        'transcode_mode': transcode_mode,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Render a web-safe overlay video from a Layer 1 perception artifact.')
    parser.add_argument('video', help='Input source video path')
    parser.add_argument('artifact', help='Layer 1 perception artifact JSON path')
    parser.add_argument('--output', help='Output MP4 path')
    parser.add_argument('--start-frame', type=int, default=0, help='First frame to render')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to render')
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    artifact_path = Path(args.artifact)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{video_path.stem}.perception_overlay.mp4"
    summary = render_overlay_video(
        video_path,
        artifact_path,
        output_path,
        start_frame=max(0, int(args.start_frame)),
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

import argparse
import subprocess
from pathlib import Path

import cv2

from tools.review.labeller.sam_refiner import (
    DEFAULT_SAM3_BALL_PROMPT,
    DEFAULT_SAM3_REPO_MODEL,
    Sam3BallDetector,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _draw_detection(frame, detection, *, color=(0, 176, 255), rank=0):
    x1, y1, x2, y2 = [int(v) for v in detection.bbox_xyxy]
    cx, cy = [int(round(v)) for v in detection.center_xy]
    thickness = 3 if rank == 0 else 1
    radius = max(6, min(18, int(max(x2 - x1, y2 - y1) * 0.5)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(frame, (cx, cy), radius, color, thickness)
    label = f"SAM3 BALL {int(round(detection.score * 100))}%"
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (min(frame.shape[1] - 1, x1 + 170), y1), (24, 24, 24), -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(14, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def render_sam3_ball_video(
    clip_path,
    output_path,
    *,
    model_name=DEFAULT_SAM3_REPO_MODEL,
    prompt=DEFAULT_SAM3_BALL_PROMPT,
    device="cuda:0",
    top_k=3,
):
    clip_path = Path(clip_path)
    if not clip_path.is_absolute():
        clip_path = (REPO_ROOT / clip_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    detector = Sam3BallDetector(
        model_name=model_name,
        text_prompt=prompt,
        device=device,
    )
    detector.load()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        overlay = frame.copy()
        for rank, detection in enumerate(detections[:top_k]):
            color = (0, 176, 255) if rank == 0 else (80, 120, 180)
            _draw_detection(overlay, detection, color=color, rank=rank)
        cv2.rectangle(overlay, (12, 12), (350, 48), (24, 24, 24), -1)
        status = f"SAM3 prompt='{prompt}' frame {frame_idx + 1}/{frame_count} detections {len(detections)}"
        cv2.putText(overlay, status, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        writer.write(overlay)
        frame_idx += 1

    writer.release()
    cap.release()
    return output_path


def transcode_websafe(input_path, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render a SAM3 basketball-detection overlay video.")
    parser.add_argument("clip_path", help="Absolute or repo-relative path to the clip")
    parser.add_argument("--output", required=True, help="Rendered MP4 output path")
    parser.add_argument("--websafe-output", default=None, help="Optional H.264 web-safe MP4 output path")
    parser.add_argument("--model", default=DEFAULT_SAM3_REPO_MODEL, help="SAM3 repo id")
    parser.add_argument("--prompt", default=DEFAULT_SAM3_BALL_PROMPT, help="SAM3 text prompt")
    parser.add_argument("--device", default="cuda:0", help="Runtime device")
    parser.add_argument("--top-k", type=int, default=3, help="How many ranked SAM3 candidates to draw per frame")
    args = parser.parse_args()

    raw_output = render_sam3_ball_video(
        args.clip_path,
        args.output,
        model_name=args.model,
        prompt=args.prompt,
        device=args.device,
        top_k=max(1, int(args.top_k)),
    )
    print(raw_output)
    if args.websafe_output:
        websafe_path = transcode_websafe(raw_output, args.websafe_output)
        print(websafe_path)


if __name__ == "__main__":
    main()

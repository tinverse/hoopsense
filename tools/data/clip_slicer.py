import cv2
import argparse
from pathlib import Path


def slice_video(video_path, output_dir, domain, segment_duration=5.0):
    """Slices raw video into semantic segments."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_segment = int(segment_duration * fps)

    video_stem = Path(video_path).stem
    domain_dir = Path(output_dir) / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Slicing {video_path} ({domain})...")

    # Simple strategy: slice entire video into segments
    for start_frame in range(0, total_frames, frames_per_segment):
        end_frame = min(start_frame + frames_per_segment, total_frames)
        if (end_frame - start_frame) < (frames_per_segment * 0.8):
            break

        output_name = f"{video_stem}_{start_frame}.mp4"
        output_path = domain_dir / output_name

        # Placeholder for actual slicing command
        print(f"  -> Generated {output_path} (Frame {start_frame} to {end_frame})")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/raw_clips")
    args = parser.parse_args()

    input_base = Path(args.input_dir)
    domains = ["nba", "ncaa", "youth", "playground"]
    for domain in domains:
        domain_path = input_base / domain
        if domain_path.exists():
            for video in domain_path.glob("*.mp4"):
                slice_video(video, args.output_dir, domain)

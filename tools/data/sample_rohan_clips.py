import cv2
import random
import argparse
import subprocess
from pathlib import Path


def get_random_clips(video_path, output_dir, domain="youth",
                     num_clips=5, duration=5.0):
    """Extracts random segments from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_frames = int(duration * fps)

    video_stem = Path(video_path).stem
    dest_dir = Path(output_dir) / domain
    dest_dir.mkdir(parents=True, exist_ok=True)

    if total_frames <= duration_frames:
        print(f"[WARN] {video_path} too short for sampling.")
        return

    print(f"[INFO] Sampling {num_clips} clips from {video_path}...")

    for i in range(num_clips):
        start_frame = random.randint(0, total_frames - duration_frames)
        start_time = start_frame / fps

        output_name = f"{video_stem}_sample_{i}.mp4"
        output_path = dest_dir / output_name

        # Use ffmpeg for accurate, fast slicing
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
            "-i", str(video_path), "-c", "copy", "-an", str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            print(f"  + Created: {output_path}")
        except subprocess.CalledProcessError:
            print(f"  ! Error generating clip for {video_path}")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing rohan_games")
    parser.add_argument("--output-dir", default="data/raw_clips")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    input_base = Path(args.input_dir)
    for video in input_base.glob("*.mp4"):
        get_random_clips(video, args.output_dir, num_clips=args.num_samples)

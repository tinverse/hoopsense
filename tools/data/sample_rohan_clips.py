import cv2
import random
import argparse
import subprocess
from pathlib import Path


def get_single_clip(video_path, output_dir, clip_idx, domain="youth", duration=5.0):
    """Extracts a single random segment from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_frames = int(duration * fps)

    if total_frames <= duration_frames:
        cap.release()
        return False

    start_frame = random.randint(0, total_frames - duration_frames)
    start_time = start_frame / fps
    
    video_stem = Path(video_path).stem
    dest_dir = Path(output_dir) / domain
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = f"{video_stem}_sample_{clip_idx}.mp4"
    output_path = dest_dir / output_name

    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
        "-i", str(video_path), "-c", "copy", "-an", str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cap.release()
        return True
    except Exception:
        cap.release()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data/raw_clips")
    parser.add_argument("--max-total", type=int, default=50)
    args = parser.parse_args()

    input_base = Path(args.input_dir)
    video_files = list(input_base.rglob("*.mp4")) + list(input_base.rglob("*.MP4"))
    random.shuffle(video_files)

    if not video_files:
        print(f"[ERROR] No videos found.")
        exit(1)

    print(f"[INFO] Found {len(video_files)} videos. Sampling up to {args.max_total} total clips...")
    
    sampled_count = 0
    while sampled_count < args.max_total:
        # Pick a random video from the shuffled list
        video = random.choice(video_files)
        if get_single_clip(video, args.output_dir, sampled_count):
            sampled_count += 1
            print(f"  [{sampled_count}/{args.max_total}] Sampled from {video.name}")
        
        # Safety break if we can't find more valid clips
        if sampled_count >= args.max_total or sampled_count > len(video_files) * 2:
            break

    print(f"[SUCCESS] {sampled_count} clips ready in {args.output_dir}/youth/")

"""Evaluate frame quality against runtime ball miss evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelines.frame_quality import (
    apply_frame_quality_raws,
    compute_frame_quality_raw,
    summarize_detection_misses_by_quality,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_runtime_ball_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "ball":
            rows.append(row)
    return rows


def evaluate_frame_quality(video_path: Path, runtime_jsonl: Path, output_path: Path, *, max_frames=None, long_side=320) -> dict:
    import cv2

    video_path = video_path if video_path.is_absolute() else (REPO_ROOT / video_path).resolve()
    runtime_jsonl = runtime_jsonl if runtime_jsonl.is_absolute() else (REPO_ROOT / runtime_jsonl).resolve()
    output_path = output_path if output_path.is_absolute() else (REPO_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ball_rows = _load_runtime_ball_rows(runtime_jsonl)
    if max_frames is not None:
        ball_rows = ball_rows[: max(0, int(max_frames))]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    raws = []
    previous_gray = None
    frame_idx = 0
    while frame_idx < len(ball_rows):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        ball_row = ball_rows[frame_idx]
        raw, previous_gray = compute_frame_quality_raw(
            frame_bgr,
            frame_idx=frame_idx + 1,
            previous_gray=previous_gray,
            long_side=long_side,
        )
        frames.append(
            {
                "frame_idx": frame_idx + 1,
                "t_ms": ball_row.get("t_ms"),
                "ball_state": {"state": ball_row.get("state")},
                "detections": [],
            }
        )
        raws.append(raw)
        frame_idx += 1
    cap.release()

    quality_summary = apply_frame_quality_raws(frames, raws, long_side=long_side)
    miss_summary = summarize_detection_misses_by_quality(frames)
    report = {
        "kind": "frame_quality_eval_v1",
        "video_path": str(video_path),
        "runtime_jsonl": str(runtime_jsonl),
        "frame_count": len(frames),
        "quality_summary": quality_summary,
        "quality_miss_summary": miss_summary,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate frame quality against runtime ball misses.")
    parser.add_argument("video")
    parser.add_argument("runtime_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--long-side", type=int, default=320)
    args = parser.parse_args(argv)
    report = evaluate_frame_quality(
        Path(args.video),
        Path(args.runtime_jsonl),
        Path(args.output),
        max_frames=args.max_frames,
        long_side=args.long_side,
    )
    print(args.output)
    print(json.dumps({
        "frame_count": report["frame_count"],
        "label_counts": report["quality_summary"]["label_counts"],
        "quality_miss_summary": report["quality_miss_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

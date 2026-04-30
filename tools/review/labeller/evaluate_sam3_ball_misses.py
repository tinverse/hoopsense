"""Run SAM3 ball detection on runtime-ball misses and classify miss causes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAM3_REPO_MODEL = "facebook/sam3"
DEFAULT_SAM3_BALL_PROMPT = "basketball"


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


def _bbox_center_xy(bbox_xywh):
    if not bbox_xywh or len(bbox_xywh) < 2:
        return None
    return [float(bbox_xywh[0]), float(bbox_xywh[1])]


def _distance(a, b):
    if a is None or b is None:
        return None
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _nearest_runtime_candidate(sam_detection, ball_row):
    sam_center = sam_detection.center_xy
    candidates = []
    for candidate in ball_row.get("candidate_scores") or []:
        center = _bbox_center_xy(candidate.get("bbox_xywh"))
        distance = _distance(sam_center, center)
        if distance is None:
            continue
        candidates.append((distance, candidate))
    for candidate in ball_row.get("rejected_candidates") or []:
        center = _bbox_center_xy(candidate.get("bbox_xywh"))
        distance = _distance(sam_center, center)
        if distance is None:
            continue
        candidates.append((distance, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    distance, candidate = candidates[0]
    payload = dict(candidate)
    payload["center_distance_px"] = round(float(distance), 3)
    return payload


def classify_sam_miss(ball_row, sam_detections, *, nearby_px=80.0):
    if not sam_detections:
        return {
            "classification": "sam_no_detection",
            "nearest_runtime_candidate": None,
            "sam_detection_count": 0,
        }
    best = sam_detections[0]
    nearest = _nearest_runtime_candidate(best, ball_row)
    if nearest is None:
        classification = "runtime_detector_absence"
    elif float(nearest.get("center_distance_px") or 0.0) <= nearby_px:
        classification = "tracker_rejected_nearby_candidate"
    else:
        classification = "runtime_detector_absence"
    return {
        "classification": classification,
        "nearest_runtime_candidate": nearest,
        "sam_detection_count": len(sam_detections),
    }


def _target_missed_rows(ball_rows, start_ms: int, end_ms: int) -> list[dict]:
    return [
        row
        for row in ball_rows
        if start_ms <= int(row.get("t_ms") or 0) < end_ms and row.get("state") == "missing"
    ]


def run_sam3_miss_eval(
    clip_path: Path,
    runtime_jsonl: Path,
    output_path: Path,
    *,
    start_s: float,
    end_s: float,
    model_name=DEFAULT_SAM3_REPO_MODEL,
    prompt=DEFAULT_SAM3_BALL_PROMPT,
    device="cuda:0",
    top_k=3,
    nearby_px=80.0,
    progress_every=25,
) -> dict:
    clip_path = clip_path if clip_path.is_absolute() else (REPO_ROOT / clip_path).resolve()
    runtime_jsonl = runtime_jsonl if runtime_jsonl.is_absolute() else (REPO_ROOT / runtime_jsonl).resolve()
    output_path = output_path if output_path.is_absolute() else (REPO_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ball_rows = _load_runtime_ball_rows(runtime_jsonl)
    start_ms = int(float(start_s) * 1000.0)
    end_ms = int(float(end_s) * 1000.0)
    missed_rows = _target_missed_rows(ball_rows, start_ms, end_ms)

    import cv2
    from tools.review.labeller.sam_refiner import Sam3BallDetector

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = Sam3BallDetector(model_name=model_name, text_prompt=prompt, device=device)
    detector.load()

    frames = []
    classifications = Counter()
    accepted_sources = Counter()
    total = len(missed_rows)
    for row_idx, ball_row in enumerate(missed_rows, start=1):
        if progress_every and (row_idx == 1 or row_idx % progress_every == 0 or row_idx == total):
            print(f"[sam3-miss-eval] frame {row_idx}/{total}", flush=True)
        t_ms = int(ball_row.get("t_ms") or 0)
        frame_idx_1based = max(1, int(round((t_ms / 1000.0) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_1based - 1)
        ok, frame = cap.read()
        if not ok:
            frames.append({"t_ms": t_ms, "frame_idx": frame_idx_1based, "status": "frame_read_failed"})
            classifications["frame_read_failed"] += 1
            continue
        detections = detector.detect(frame)
        detections = detections[: max(1, int(top_k))]
        result = classify_sam_miss(ball_row, detections, nearby_px=nearby_px)
        classifications[result["classification"]] += 1
        if result["nearest_runtime_candidate"]:
            accepted_sources[result["nearest_runtime_candidate"].get("source")] += 1
        frames.append(
            {
                "t_ms": t_ms,
                "frame_idx": frame_idx_1based,
                "runtime_state": ball_row.get("state"),
                "runtime_candidate_count": int(ball_row.get("candidate_count") or 0),
                "classification": result["classification"],
                "nearest_runtime_candidate": result["nearest_runtime_candidate"],
                "sam_detections": [
                    {
                        "bbox_xyxy": detection.bbox_xyxy,
                        "bbox_xywh": detection.bbox_xywh,
                        "center_xy": detection.center_xy,
                        "mask_area_px": detection.mask_area_px,
                        "mask_area_ratio": detection.mask_area_ratio,
                        "score": detection.score,
                        "prompt": detection.prompt,
                    }
                    for detection in detections
                ],
            }
        )
    cap.release()

    report = {
        "kind": "sam3_ball_missed_frame_eval_v1",
        "clip_path": str(clip_path),
        "runtime_jsonl": str(runtime_jsonl),
        "window_s": [float(start_s), float(end_s)],
        "prompt": prompt,
        "model_name": model_name,
        "device": device,
        "missed_frame_count": len(missed_rows),
        "classification_counts": dict(sorted(classifications.items())),
        "nearest_runtime_candidate_source_counts": dict(sorted(accepted_sources.items())),
        "frames": frames,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 detections on runtime ball misses.")
    parser.add_argument("clip_path")
    parser.add_argument("runtime_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--model", default=DEFAULT_SAM3_REPO_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_SAM3_BALL_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nearby-px", type=float, default=80.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)

    report = run_sam3_miss_eval(
        Path(args.clip_path),
        Path(args.runtime_jsonl),
        Path(args.output),
        start_s=args.start_s,
        end_s=args.end_s,
        model_name=args.model,
        prompt=args.prompt,
        device=args.device,
        top_k=args.top_k,
        nearby_px=args.nearby_px,
        progress_every=args.progress_every,
    )
    print(args.output)
    print(json.dumps({k: report[k] for k in ("missed_frame_count", "classification_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

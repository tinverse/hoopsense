"""Compare SAM3 basketball detections against existing runtime/artifact ball states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAM3_REPO_MODEL = "facebook/sam3"
DEFAULT_SAM3_BALL_PROMPT = "basketball"


def _center_distance(a, b):
    if a is None or b is None:
        return None
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _state_from_artifact_frame(frame):
    ball = frame.get("ball_state") or frame.get("ball_detection") or {}
    center = ball.get("center_xy")
    state = ball.get("state") or ("observed" if center else "missing")
    return {
        "state": state,
        "center_xy": center,
        "confidence": ball.get("confidence"),
        "source": ball.get("source") or ball.get("ball_detection_source"),
    }


def _load_artifact_states(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [_state_from_artifact_frame(frame) for frame in data.get("frames") or []]


def _load_jsonl_states(path: Path) -> list[dict]:
    states = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") != "ball":
            continue
        center = None
        if row.get("x") is not None and row.get("y") is not None:
            center = [float(row["x"]), float(row["y"])]
        states.append(
            {
                "state": row.get("state") or ("observed" if center else "missing"),
                "center_xy": center,
                "confidence": (float(row.get("confidence_bps") or 0.0) / 10000.0),
                "source": row.get("source"),
            }
        )
    return states


def _load_reference_states(path: Path | None, kind: str | None) -> list[dict]:
    if path is None:
        return []
    if kind == "jsonl" or path.suffix == ".jsonl":
        return _load_jsonl_states(path)
    return _load_artifact_states(path)


def _ranked_sam_candidates(sam_detections, runtime_center):
    ranked = []
    for rank, detection in enumerate(sam_detections, start=1):
        distance = _center_distance(detection.center_xy, runtime_center)
        ranked.append((float("inf") if distance is None else float(distance), rank, detection))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked


def _detection_payload(detection):
    if detection is None:
        return None
    return {
        "bbox_xyxy": detection.bbox_xyxy,
        "bbox_xywh": detection.bbox_xywh,
        "center_xy": detection.center_xy,
        "score": detection.score,
        "mask_area_px": detection.mask_area_px,
        "mask_area_ratio": detection.mask_area_ratio,
    }


def _classify_frame(sam_detections, runtime_state, *, match_distance_px):
    runtime_center = runtime_state.get("center_xy")
    if not sam_detections and runtime_center is None:
        return {
            "classification": "both_missing",
            "center_distance_px": None,
            "best_sam_rank": None,
            "best_sam_detection": None,
        }
    if sam_detections and runtime_center is None:
        return {
            "classification": "sam_only",
            "center_distance_px": None,
            "best_sam_rank": 1,
            "best_sam_detection": _detection_payload(sam_detections[0]),
        }
    if not sam_detections and runtime_center is not None:
        return {
            "classification": "runtime_only",
            "center_distance_px": None,
            "best_sam_rank": None,
            "best_sam_detection": None,
        }

    distance, rank, detection = _ranked_sam_candidates(sam_detections, runtime_center)[0]
    if distance is not None and distance <= float(match_distance_px):
        classification = "matched"
    else:
        classification = "disagree"
    return {
        "classification": classification,
        "center_distance_px": distance,
        "best_sam_rank": rank,
        "best_sam_detection": _detection_payload(detection),
    }


def compare_sam3_ball_to_runtime(
    video_path: Path,
    output_path: Path,
    *,
    reference_path: Path | None = None,
    reference_kind: str | None = None,
    model_name=DEFAULT_SAM3_REPO_MODEL,
    prompt=DEFAULT_SAM3_BALL_PROMPT,
    device="cuda:0",
    max_frames=120,
    start_frame=0,
    top_k=3,
    match_distance_px=80.0,
    progress_every=25,
) -> dict:
    import cv2
    from tools.review.labeller.sam_refiner import Sam3BallDetector

    video_path = video_path if video_path.is_absolute() else (REPO_ROOT / video_path).resolve()
    output_path = output_path if output_path.is_absolute() else (REPO_ROOT / output_path).resolve()
    if reference_path is not None and not reference_path.is_absolute():
        reference_path = (REPO_ROOT / reference_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_states = _load_reference_states(reference_path, reference_kind)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))

    detector = Sam3BallDetector(model_name=model_name, text_prompt=prompt, device=device)
    detector.load()

    frames = []
    counts = Counter()
    for offset in range(max(0, int(max_frames))):
        frame_idx = int(start_frame) + offset
        ok, frame = cap.read()
        if not ok:
            break
        if progress_every and (offset == 0 or (offset + 1) % progress_every == 0):
            print(f"[sam3-compare] frame {offset + 1}/{max_frames}", flush=True)
        sam_detections = detector.detect(frame)[: max(1, int(top_k))]
        runtime_state = reference_states[frame_idx] if frame_idx < len(reference_states) else {"state": "missing", "center_xy": None}
        frame_match = _classify_frame(sam_detections, runtime_state, match_distance_px=match_distance_px)
        classification = frame_match["classification"]
        distance = frame_match["center_distance_px"]
        counts[classification] += 1
        frames.append(
            {
                "frame_idx": frame_idx,
                "t_ms": int(round((frame_idx / max(float(fps), 1.0)) * 1000.0)),
                "classification": classification,
                "center_distance_px": round(float(distance), 3) if distance is not None else None,
                "best_sam_rank": frame_match["best_sam_rank"],
                "best_sam_detection": frame_match["best_sam_detection"],
                "runtime_ball": runtime_state,
                "sam3_detections": [_detection_payload(detection) for detection in sam_detections],
            }
        )
    cap.release()

    compared = len(frames)
    sam_positive = sum(1 for frame in frames if frame["sam3_detections"])
    runtime_positive = sum(1 for frame in frames if frame["runtime_ball"].get("center_xy") is not None)
    report = {
        "kind": "sam3_ball_runtime_comparison_v1",
        "video_path": str(video_path),
        "reference_path": str(reference_path) if reference_path else None,
        "model_name": model_name,
        "prompt": prompt,
        "device": device,
        "start_frame": int(start_frame),
        "requested_max_frames": int(max_frames),
        "compared_frames": compared,
        "video_frame_count": frame_count,
        "fps": float(fps),
        "match_distance_px": float(match_distance_px),
        "classification_counts": dict(sorted(counts.items())),
        "sam_positive_frames": int(sam_positive),
        "runtime_positive_frames": int(runtime_positive),
        "sam_recall_proxy": round(float(sam_positive) / max(float(compared), 1.0), 4),
        "runtime_recall_vs_sam_proxy": round(float(counts["matched"] + counts["disagree"]) / max(float(sam_positive), 1.0), 4),
        "frames": frames,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare SAM3 basketball detections against existing ball states.")
    parser.add_argument("video")
    parser.add_argument("--reference", default=None, help="Runtime JSONL or Layer 1 artifact JSON to compare against")
    parser.add_argument("--reference-kind", choices=["jsonl", "artifact"], default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_SAM3_REPO_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_SAM3_BALL_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--match-distance-px", type=float, default=80.0)
    args = parser.parse_args(argv)
    report = compare_sam3_ball_to_runtime(
        Path(args.video),
        Path(args.output),
        reference_path=Path(args.reference) if args.reference else None,
        reference_kind=args.reference_kind,
        model_name=args.model,
        prompt=args.prompt,
        device=args.device,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        top_k=args.top_k,
        match_distance_px=args.match_distance_px,
    )
    print(args.output)
    print(json.dumps({k: report[k] for k in ("compared_frames", "classification_counts", "sam_positive_frames", "runtime_positive_frames", "runtime_recall_vs_sam_proxy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

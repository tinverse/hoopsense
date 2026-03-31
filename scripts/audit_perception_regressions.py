#!/usr/bin/env python3
"""Summarize reviewed Layer 1 regression cases against current artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "perception_regression_cases.json"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data" / "review_artifacts" / "layer1"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "perception_regression_audit.json"
MOTION_ACTIVE_THRESHOLD_PX = 6.0
OVERLAP_IOU_THRESHOLD = 0.2


CATEGORY_SIGNAL_MAP = {
    "spectator_false_positive": [
        "detection_count",
        "active_candidate_count",
        "low_motion_detection_count",
        "zero_motion_detection_count",
        "track_ids",
    ],
    "bystander_false_positive": [
        "detection_count",
        "active_candidate_count",
        "low_motion_detection_count",
        "zero_motion_detection_count",
        "track_ids",
    ],
    "false_positive_generic": [
        "detection_count",
        "active_candidate_count",
        "track_ids",
    ],
    "cluster_miss": [
        "detection_count",
        "overlap_pair_count",
        "synthesized_detection_count",
        "track_ids",
    ],
    "cluster_merge": [
        "detection_count",
        "overlap_pair_count",
        "synthesized_detection_count",
        "track_ids",
    ],
    "false_negative_generic": [
        "detection_count",
        "synthesized_detection_count",
        "track_ids",
    ],
    "id_switch": [
        "track_ids",
        "motion_speed_max",
        "motion_speed_median",
        "synthesized_detection_count",
    ],
    "uniform_confusion": [
        "uniform_bucket_counts",
        "active_uniform_bucket_counts",
        "track_ids",
    ],
    "live_play_context": [
        "active_candidate_count",
        "motion_speed_max",
        "motion_speed_median",
        "track_ids",
    ],
    "dead_ball_context": [
        "active_candidate_count",
        "low_motion_detection_count",
        "zero_motion_detection_count",
        "track_ids",
    ],
    "scene_context": [
        "detection_count",
        "active_candidate_count",
        "track_ids",
    ],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
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


def summarize_frame(frame: dict) -> dict:
    detections = frame.get("detections", [])
    track_ids = sorted(
        str(detection.get("track_id"))
        for detection in detections
        if detection.get("track_id") is not None
    )
    active = [d for d in detections if d.get("active_player_candidate")]
    motion_speeds = [float(d.get("motion_speed_px") or 0.0) for d in detections]
    uniform_bucket_counts = Counter(
        str(d.get("uniform_bucket") or "unknown") for d in detections
    )
    active_uniform_bucket_counts = Counter(
        str(d.get("uniform_bucket") or "unknown") for d in active
    )
    synthesized_count = sum(1 for d in detections if d.get("synthesized"))
    low_motion_count = sum(
        1 for speed in motion_speeds if speed < MOTION_ACTIVE_THRESHOLD_PX
    )
    zero_motion_count = sum(1 for speed in motion_speeds if speed <= 0.5)

    overlap_pair_count = 0
    for idx, left in enumerate(detections):
        bbox_left = left.get("bbox_xyxy")
        if not bbox_left:
            continue
        for right in detections[idx + 1 :]:
            bbox_right = right.get("bbox_xyxy")
            if not bbox_right:
                continue
            if bbox_iou(bbox_left, bbox_right) >= OVERLAP_IOU_THRESHOLD:
                overlap_pair_count += 1

    return {
        "detection_count": len(detections),
        "track_ids": track_ids,
        "active_candidate_count": len(active),
        "active_candidate_track_ids": sorted(
            str(d.get("track_id"))
            for d in active
            if d.get("track_id") is not None
        ),
        "synthesized_detection_count": synthesized_count,
        "low_motion_detection_count": low_motion_count,
        "zero_motion_detection_count": zero_motion_count,
        "motion_speed_min": round(min(motion_speeds), 3) if motion_speeds else 0.0,
        "motion_speed_median": round(median(motion_speeds), 3) if motion_speeds else 0.0,
        "motion_speed_max": round(max(motion_speeds), 3) if motion_speeds else 0.0,
        "uniform_bucket_counts": dict(sorted(uniform_bucket_counts.items())),
        "active_uniform_bucket_counts": dict(sorted(active_uniform_bucket_counts.items())),
        "overlap_pair_count": overlap_pair_count,
    }


def select_category_signals(metrics: dict, categories: list[str]) -> dict:
    signal_keys = []
    for category in categories:
        signal_keys.extend(CATEGORY_SIGNAL_MAP.get(category, []))
    if not signal_keys:
        signal_keys = ["detection_count", "active_candidate_count", "track_ids"]
    seen = set()
    selected = {}
    for key in signal_keys:
        if key in seen:
            continue
        seen.add(key)
        selected[key] = metrics.get(key)
    return selected


def build_case_report(case: dict, artifact_dir: Path) -> dict:
    artifact_path = artifact_dir / f"{case['clip_id']}.perception.json"
    if not artifact_path.exists():
        return {
            "case_id": case["case_id"],
            "clip_id": case["clip_id"],
            "frame_idx": case["frame_idx"],
            "categories": case.get("categories", []),
            "scene_context": case.get("scene_context"),
            "artifact_found": False,
            "frame_found": False,
            "notes": case.get("notes", []),
        }

    artifact = load_json(artifact_path)
    frame = next(
        (frame for frame in artifact.get("frames", []) if frame.get("frame_idx") == case["frame_idx"]),
        None,
    )
    if frame is None:
        return {
            "case_id": case["case_id"],
            "clip_id": case["clip_id"],
            "frame_idx": case["frame_idx"],
            "categories": case.get("categories", []),
            "scene_context": case.get("scene_context"),
            "artifact_found": True,
            "frame_found": False,
            "artifact_path": display_path(artifact_path),
            "notes": case.get("notes", []),
        }

    metrics = summarize_frame(frame)
    return {
        "case_id": case["case_id"],
        "clip_id": case["clip_id"],
        "frame_idx": case["frame_idx"],
        "categories": case.get("categories", []),
        "scene_context": case.get("scene_context"),
        "artifact_found": True,
        "frame_found": True,
        "artifact_path": display_path(artifact_path),
        "notes": case.get("notes", []),
        "frame_metrics": metrics,
        "category_signals": select_category_signals(metrics, case.get("categories", [])),
    }


def build_report(fixture: dict, artifact_dir: Path, fixture_path: Path) -> dict:
    case_reports = [build_case_report(case, artifact_dir) for case in fixture.get("cases", [])]
    by_category = defaultdict(
        lambda: {
            "case_count": 0,
            "artifact_found_count": 0,
            "frame_found_count": 0,
            "detection_count_total": 0,
            "active_candidate_count_total": 0,
            "synthesized_detection_count_total": 0,
            "motion_speed_max_seen": 0.0,
        }
    )

    for case_report in case_reports:
        categories = case_report.get("categories", []) or ["uncategorized"]
        metrics = case_report.get("frame_metrics", {})
        for category in categories:
            summary = by_category[category]
            summary["case_count"] += 1
            summary["artifact_found_count"] += int(case_report.get("artifact_found", False))
            summary["frame_found_count"] += int(case_report.get("frame_found", False))
            summary["detection_count_total"] += int(metrics.get("detection_count", 0))
            summary["active_candidate_count_total"] += int(metrics.get("active_candidate_count", 0))
            summary["synthesized_detection_count_total"] += int(metrics.get("synthesized_detection_count", 0))
            summary["motion_speed_max_seen"] = max(
                float(summary["motion_speed_max_seen"]),
                float(metrics.get("motion_speed_max", 0.0)),
            )

    category_summary = {}
    for category, values in sorted(by_category.items()):
        case_count = max(int(values["case_count"]), 1)
        category_summary[category] = {
            **dict(sorted(values.items())),
            "avg_detection_count": round(values["detection_count_total"] / case_count, 3),
            "avg_active_candidate_count": round(values["active_candidate_count_total"] / case_count, 3),
            "avg_synthesized_detection_count": round(values["synthesized_detection_count_total"] / case_count, 3),
        }

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture": display_path(fixture_path),
        "artifact_dir": display_path(artifact_dir),
        "case_count": len(case_reports),
        "limitations": [
            "This audit reports current artifact signals for reviewed cases; it does not declare pass/fail quality.",
            "Layer 1 review artifacts are currently person-only; ball detections and live-play truth are not included here.",
            "Use category rollups to steer heuristic changes, then validate on held-out clips to avoid clip-specific overfitting.",
        ],
        "category_summary": category_summary,
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reviewed Layer 1 regression cases against current artifacts.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    artifact_dir = Path(args.artifact_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixture = load_json(fixture_path)
    report = build_report(fixture, artifact_dir, fixture_path)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()

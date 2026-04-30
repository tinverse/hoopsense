"""Evaluate and compare Layer 1 perception artifacts across named runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data" / "review_artifacts" / "layer1"
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "perception_regression_cases.json"
DEFAULT_OUTPUT = REPO_ROOT / "tmp_runs" / "layer1_evaluation" / "comparison_report.json"


FALSE_POSITIVE_CATEGORIES = {
    "spectator_false_positive",
    "bystander_false_positive",
    "false_positive_generic",
}
FALSE_NEGATIVE_CATEGORIES = {
    "cluster_miss",
    "false_negative_generic",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 4)


def longest_state_run(frames: list[dict], state_name: str) -> int:
    longest = 0
    current = 0
    for frame in frames:
        state = (frame.get("ball_state") or {}).get("state")
        if state == state_name:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_artifact(artifact: dict) -> dict:
    frames = artifact.get("frames") or []
    detections_per_frame = [len(frame.get("detections") or []) for frame in frames]
    active_per_frame = [
        sum(1 for detection in frame.get("detections") or [] if detection.get("active_player_candidate"))
        for frame in frames
    ]
    raw_proposals_per_frame = [len(frame.get("raw_player_detections") or []) for frame in frames]
    ball_state_counts = Counter(str((frame.get("ball_state") or {}).get("state") or "missing") for frame in frames)
    ball_keep_alive_counts = Counter(str((frame.get("ball_state") or {}).get("keep_alive_kind") or "none") for frame in frames)
    live_play_counts = Counter(str(frame.get("live_play_label") or "unknown") for frame in frames)
    engagement_counts = Counter()
    temporal_counts = Counter()
    source_counts = Counter()
    track_ids = set()
    identity_ids = set()
    ambiguous_identity_count = 0
    synthesized_count = 0

    for frame in frames:
        for detection in frame.get("detections") or []:
            if detection.get("track_id") is not None:
                track_ids.add(str(detection.get("track_id")))
            if detection.get("identity_track_id", detection.get("track_id")) is not None:
                identity_ids.add(str(detection.get("identity_track_id", detection.get("track_id"))))
            if detection.get("identity_is_ambiguous"):
                ambiguous_identity_count += 1
            if detection.get("synthesized"):
                synthesized_count += 1
            engagement_counts[str(detection.get("engagement_state") or "unknown")] += 1
            temporal_counts[str(detection.get("tracklet_temporal_state") or "unknown")] += 1
            source_counts[str(detection.get("player_detection_source") or "unknown")] += 1

    frame_count = len(frames)
    video = artifact.get("video") or {}
    fps = float(video.get("fps") or 0.0)
    duration_s = round(frame_count / fps, 3) if fps > 0.0 else 0.0
    active_low_frames = sum(1 for value in active_per_frame if value < 6)
    active_high_frames = sum(1 for value in active_per_frame if value > 12)

    return {
        "clip_id": artifact.get("clip_id") or Path(str(artifact.get("video_path") or "")).stem,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "player": {
            "detection_count_total": int(sum(detections_per_frame)),
            "active_candidate_count_total": int(sum(active_per_frame)),
            "raw_proposal_count_total": int(sum(raw_proposals_per_frame)),
            "detections_per_frame_median": round(median(detections_per_frame), 4) if detections_per_frame else 0.0,
            "detections_per_frame_p95": percentile(detections_per_frame, 95),
            "active_per_frame_median": round(median(active_per_frame), 4) if active_per_frame else 0.0,
            "active_per_frame_p95": percentile(active_per_frame, 95),
            "active_low_frame_count": int(active_low_frames),
            "active_high_frame_count": int(active_high_frames),
            "active_low_frame_ratio": round(active_low_frames / frame_count, 4) if frame_count else 0.0,
            "active_high_frame_ratio": round(active_high_frames / frame_count, 4) if frame_count else 0.0,
            "source_counts": dict(sorted(source_counts.items())),
            "engagement_state_counts": dict(sorted(engagement_counts.items())),
            "tracklet_temporal_state_counts": dict(sorted(temporal_counts.items())),
        },
        "ball": {
            "state_counts": dict(sorted(ball_state_counts.items())),
            "keep_alive_counts": dict(sorted(ball_keep_alive_counts.items())),
            "observed_frame_ratio": round(ball_state_counts.get("observed", 0) / frame_count, 4) if frame_count else 0.0,
            "observed_or_predicted_frame_ratio": round(
                (ball_state_counts.get("observed", 0) + ball_state_counts.get("predicted_short_gap", 0)) / frame_count,
                4,
            ) if frame_count else 0.0,
            "longest_missing_run_frames": int(longest_state_run(frames, "missing")),
            "longest_predicted_run_frames": int(longest_state_run(frames, "predicted_short_gap")),
        },
        "identity": {
            "unique_track_count": len(track_ids),
            "unique_identity_count": len(identity_ids),
            "ambiguous_identity_detection_count": int(ambiguous_identity_count),
            "synthesized_detection_count": int(synthesized_count),
        },
        "live_play": {
            "label_counts": dict(sorted(live_play_counts.items())),
        },
    }


def summarize_frame_for_feedback(frame: dict) -> dict:
    detections = frame.get("detections") or []
    active = [detection for detection in detections if detection.get("active_player_candidate")]
    ball_state = frame.get("ball_state") or {}
    return {
        "detection_count": len(detections),
        "active_candidate_count": len(active),
        "track_ids": sorted(str(detection.get("track_id")) for detection in detections if detection.get("track_id") is not None),
        "active_track_ids": sorted(str(detection.get("track_id")) for detection in active if detection.get("track_id") is not None),
        "ambiguous_identity_detection_count": sum(1 for detection in detections if detection.get("identity_is_ambiguous")),
        "synthesized_detection_count": sum(1 for detection in detections if detection.get("synthesized")),
        "live_play_label": frame.get("live_play_label"),
        "ball_state": ball_state.get("state"),
        "ball_confidence": ball_state.get("confidence"),
    }


def category_pressure(categories: list[str], metrics: dict, scene_context: str | None) -> float:
    category_set = set(categories)
    pressure = 0.0
    if category_set & FALSE_POSITIVE_CATEGORIES:
        pressure += float(metrics.get("active_candidate_count") or 0)
    if category_set & FALSE_NEGATIVE_CATEGORIES:
        pressure += max(0.0, 10.0 - float(metrics.get("active_candidate_count") or 0))
    if "cluster_merge" in category_set:
        pressure += float(metrics.get("synthesized_detection_count") or 0)
    if "id_switch" in category_set:
        pressure += float(metrics.get("ambiguous_identity_detection_count") or 0)
    if scene_context == "live_play" and metrics.get("live_play_label") == "dead_ball":
        pressure += 5.0
    if scene_context == "dead_ball" and metrics.get("live_play_label") == "live":
        pressure += 5.0
    return round(pressure, 4)


def evaluate_feedback_cases(artifacts_by_clip: dict[str, dict], fixture: dict | None) -> dict:
    if not fixture:
        return {"case_count": 0, "matched_case_count": 0, "pressure_total": 0.0, "cases": []}

    cases = []
    by_category = defaultdict(lambda: {"case_count": 0, "matched_case_count": 0, "pressure_total": 0.0})
    for case in fixture.get("cases") or []:
        clip_id = case.get("clip_id")
        frame_idx = int(case.get("frame_idx") or 0)
        artifact = artifacts_by_clip.get(str(clip_id))
        categories = list(case.get("categories") or [])
        result = {
            "case_id": case.get("case_id") or f"{clip_id}:{frame_idx}",
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "categories": categories,
            "scene_context": case.get("scene_context"),
            "notes": case.get("notes") or [],
            "artifact_found": artifact is not None,
            "frame_found": False,
        }
        metrics = None
        if artifact is not None:
            frame = next((f for f in artifact.get("frames") or [] if int(f.get("frame_idx") or 0) == frame_idx), None)
            if frame is not None:
                metrics = summarize_frame_for_feedback(frame)
                result["frame_found"] = True
                result["frame_metrics"] = metrics
                result["pressure"] = category_pressure(categories, metrics, case.get("scene_context"))
        if metrics is None:
            result["pressure"] = 0.0
        cases.append(result)
        for category in categories or ["uncategorized"]:
            summary = by_category[category]
            summary["case_count"] += 1
            summary["matched_case_count"] += int(result["frame_found"])
            summary["pressure_total"] += float(result["pressure"])

    category_summary = {}
    for category, values in sorted(by_category.items()):
        matched = max(1, int(values["matched_case_count"]))
        category_summary[category] = {
            "case_count": int(values["case_count"]),
            "matched_case_count": int(values["matched_case_count"]),
            "pressure_total": round(float(values["pressure_total"]), 4),
            "pressure_per_matched_case": round(float(values["pressure_total"]) / matched, 4),
        }
    return {
        "case_count": len(cases),
        "matched_case_count": sum(1 for case in cases if case.get("frame_found")),
        "pressure_total": round(sum(float(case.get("pressure") or 0.0) for case in cases), 4),
        "category_summary": category_summary,
        "cases": cases,
    }


def load_artifacts(artifact_dir: Path) -> dict[str, dict]:
    artifacts = {}
    for path in sorted(artifact_dir.glob("*.perception.json")):
        artifact = load_json(path)
        clip_id = artifact.get("clip_id") or path.name.removesuffix(".perception.json")
        artifacts[str(clip_id)] = artifact
    return artifacts


def parse_run(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        name = Path(path).name or "run"
    if not name:
        raise ValueError(f"invalid run spec {value!r}: missing name")
    return name, Path(path)


def build_run_scorecard(name: str, artifact_dir: Path, fixture: dict | None) -> dict:
    artifacts = load_artifacts(artifact_dir)
    clip_summaries = {
        clip_id: summarize_artifact(artifact)
        for clip_id, artifact in sorted(artifacts.items())
    }
    feedback = evaluate_feedback_cases(artifacts, fixture)
    return {
        "name": name,
        "artifact_dir": str(artifact_dir),
        "clip_count": len(clip_summaries),
        "clips": clip_summaries,
        "feedback": feedback,
    }


def _metric_at(run: dict, path: tuple[str, ...]) -> float:
    value = run
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    return float(value or 0.0)


def build_deltas(runs: list[dict]) -> list[dict]:
    if len(runs) < 2:
        return []
    baseline = runs[0]
    deltas = []
    for run in runs[1:]:
        deltas.append({
            "baseline": baseline["name"],
            "candidate": run["name"],
            "feedback_pressure_delta": round(
                _metric_at(run, ("feedback", "pressure_total")) - _metric_at(baseline, ("feedback", "pressure_total")),
                4,
            ),
            "matched_case_delta": int(_metric_at(run, ("feedback", "matched_case_count")) - _metric_at(baseline, ("feedback", "matched_case_count"))),
            "clip_count_delta": int(run.get("clip_count", 0) - baseline.get("clip_count", 0)),
        })
    return deltas


def build_comparison_report(run_specs: list[tuple[str, Path]], fixture_path: Path | None = DEFAULT_FIXTURE) -> dict:
    fixture = load_json(fixture_path) if fixture_path and fixture_path.exists() else None
    runs = [build_run_scorecard(name, path, fixture) for name, path in run_specs]
    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "layer1_perception_run_comparison_v1",
        "fixture": str(fixture_path) if fixture_path else None,
        "run_count": len(runs),
        "limitations": [
            "Feedback-derived pressure metrics are weak labels, not final pass/fail ground truth.",
            "Use deltas to choose experiments, then verify important cases manually in the labeller.",
            "Variant artifact directories should be immutable snapshots; do not compare against a directory that is being overwritten by a live run.",
        ],
        "runs": runs,
        "deltas_vs_first_run": build_deltas(runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare named Layer 1 perception artifact runs.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Named artifact directory, e.g. baseline=data/review_artifacts/layer1. Can be repeated.",
    )
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Regression fixture JSON path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output comparison JSON path")
    args = parser.parse_args(argv)

    run_specs = [parse_run(value) for value in args.run] or [("current", DEFAULT_ARTIFACT_DIR)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_comparison_report(run_specs, Path(args.fixture) if args.fixture else None)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

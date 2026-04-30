"""First-pass target-court bootstrap from sparse video evidence.

This module estimates a conservative image-space target-court prior before the
normal ball/player stages consume detections. It does not pretend to solve full
camera calibration from weak cues. Instead it samples a bounded set of frames,
aggregates court/hoop proposals plus player foot anchors, and emits an auditable
region that downstream scoring can use to reject background-court objects.
"""

from __future__ import annotations

from dataclasses import dataclass


NCAA_COURT_DIMENSIONS_FT = {"length": 94.0, "width": 50.0}
NBA_COURT_DIMENSIONS_FT = {"length": 94.0, "width": 50.0}
DEFAULT_MAX_POSE_SEGMENT_FRAMES = 120
DEFAULT_HIGH_MOTION_PX = 18.0


@dataclass
class _CandidateRegion:
    bbox_xyxy: list[float]
    source: str
    confidence: float
    frame_idx: int
    source_frame_idx: int


def _clip(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _bbox_area(bbox):
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_contains_point(bbox, point, *, margin_px=0.0):
    if not bbox or len(bbox) != 4 or point is None:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x, y = [float(v) for v in point]
    return (x1 - margin_px) <= x <= (x2 + margin_px) and (y1 - margin_px) <= y <= (y2 + margin_px)


def _bbox_center(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]


def _union_bbox(boxes, *, image_width, image_height, padding_ratio=0.04):
    valid = [box for box in boxes if box and len(box) == 4 and _bbox_area(box) > 0.0]
    if not valid:
        return None
    x1 = min(float(box[0]) for box in valid)
    y1 = min(float(box[1]) for box in valid)
    x2 = max(float(box[2]) for box in valid)
    y2 = max(float(box[3]) for box in valid)
    pad = padding_ratio * max(float(image_width), float(image_height), 1.0)
    return [
        round(_clip(x1 - pad, 0.0, image_width), 3),
        round(_clip(y1 - pad, 0.0, image_height), 3),
        round(_clip(x2 + pad, 0.0, image_width), 3),
        round(_clip(y2 + pad, 0.0, image_height), 3),
    ]


def _sample_frames(frames, *, max_sample_frames):
    if not frames:
        return []
    if len(frames) <= max_sample_frames:
        return list(frames)
    last = len(frames) - 1
    indexes = sorted({round(i * last / max(max_sample_frames - 1, 1)) for i in range(max_sample_frames)})
    return [frames[int(index)] for index in indexes]


def _frame_motion_px(frame):
    frame_quality = frame.get("frame_quality") or {}
    value = frame_quality.get("global_motion_px")
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _camera_pose_segments(frames, *, max_pose_segment_frames, high_motion_px):
    if not frames:
        return []
    segments = []
    current = {
        "pose_segment_id": 0,
        "start_index": 0,
        "end_index": 0,
        "start_frame_idx": int(frames[0].get("frame_idx") or 0),
        "end_frame_idx": int(frames[0].get("frame_idx") or 0),
        "continuity_segment_id": frames[0].get("continuity_segment_id"),
        "split_reasons": ["clip_start"],
        "high_motion_frame_count": 0,
    }
    previous_continuity = frames[0].get("continuity_segment_id")
    previous_was_high_motion = False
    for index, frame in enumerate(frames[1:], start=1):
        frame_idx = int(frame.get("frame_idx") or 0)
        continuity = frame.get("continuity_segment_id")
        motion_px = _frame_motion_px(frame)
        high_motion = motion_px >= float(high_motion_px)
        reasons = []
        if continuity != previous_continuity:
            reasons.append("continuity_segment_changed")
        if index - int(current["start_index"]) >= int(max_pose_segment_frames):
            reasons.append("max_pose_segment_frames")
        if high_motion and not previous_was_high_motion and index > int(current["start_index"]):
            reasons.append("camera_pan_high_motion")
        if reasons:
            current["end_index"] = index - 1
            current["end_frame_idx"] = int(frames[index - 1].get("frame_idx") or 0)
            segments.append(current)
            current = {
                "pose_segment_id": len(segments),
                "start_index": index,
                "end_index": index,
                "start_frame_idx": frame_idx,
                "end_frame_idx": frame_idx,
                "continuity_segment_id": continuity,
                "split_reasons": reasons,
                "high_motion_frame_count": 1 if high_motion else 0,
            }
        else:
            current["high_motion_frame_count"] += 1 if high_motion else 0
        previous_continuity = continuity
        previous_was_high_motion = high_motion
    current["end_index"] = len(frames) - 1
    current["end_frame_idx"] = int(frames[-1].get("frame_idx") or 0)
    segments.append(current)
    return segments


def annotate_camera_pose_segments(
    frames,
    *,
    max_pose_segment_frames=DEFAULT_MAX_POSE_SEGMENT_FRAMES,
    high_motion_px=DEFAULT_HIGH_MOTION_PX,
):
    """Assign camera-pose segment ids used by first-pass grounding."""
    segments = _camera_pose_segments(
        frames,
        max_pose_segment_frames=max(1, int(max_pose_segment_frames)),
        high_motion_px=float(high_motion_px),
    )
    for segment in segments:
        for frame in frames[int(segment["start_index"]): int(segment["end_index"]) + 1]:
            frame["camera_pose_segment_id"] = int(segment["pose_segment_id"])
    return {
        "kind": "camera_pose_segments_v1",
        "segment_count": len(segments),
        "max_pose_segment_frames": int(max_pose_segment_frames),
        "high_motion_px": round(float(high_motion_px), 3),
        "segments": [
            {
                "pose_segment_id": int(segment["pose_segment_id"]),
                "start_frame_idx": int(segment["start_frame_idx"]),
                "end_frame_idx": int(segment["end_frame_idx"]),
                "continuity_segment_id": segment.get("continuity_segment_id"),
                "split_reasons": list(segment.get("split_reasons") or []),
                "high_motion_frame_count": int(segment.get("high_motion_frame_count") or 0),
            }
            for segment in segments
        ],
    }


def _label_for_region(region):
    return str(region.get("text_label") or region.get("kind") or "").lower()


def _source_frame_idx(region, bootstrap, frame_idx):
    for value in (region.get("frame_idx"), bootstrap.get("frame_idx"), frame_idx):
        if value is not None:
            return int(value)
    return int(frame_idx)


def _collect_regions(frame, *, image_width, image_height):
    regions = []
    frame_idx = int(frame.get("frame_idx") or 0)
    sources = []
    bootstrap = frame.get("bootstrap_context") or {}
    grounding = frame.get("grounding_context") or {}
    scene_prior = frame.get("scene_prior") or {}
    sources.extend(bootstrap.get("proposal_regions") or [])
    sources.extend(grounding.get("proposal_regions") or [])
    sources.extend(scene_prior.get("proposal_regions") or [])
    for region in sources:
        bbox = region.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        area_ratio = _bbox_area(bbox) / max(float(image_width * image_height), 1.0)
        if area_ratio <= 0.002 or area_ratio >= 0.96:
            continue
        label = _label_for_region(region)
        if not any(token in label for token in ("court", "hoop", "basket", "play_region")):
            continue
        confidence = float(region.get("confidence") or region.get("area_ratio") or 0.25)
        regions.append(
            _CandidateRegion(
                bbox_xyxy=[float(v) for v in bbox],
                source=label or "region",
                confidence=max(0.0, min(1.0, confidence)),
                frame_idx=frame_idx,
                source_frame_idx=_source_frame_idx(region, bootstrap, frame_idx),
            )
        )
    return regions


def _dedupe_regions(regions):
    deduped = []
    seen = set()
    for region in regions:
        key = (
            region.source,
            int(region.source_frame_idx),
            tuple(round(float(value), 1) for value in region.bbox_xyxy),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    return deduped


def _footpoint(detection):
    bbox = detection.get("bbox_xyxy") or []
    if len(bbox) != 4:
        return None
    keypoints = detection.get("keypoints_xy") or []
    confidences = detection.get("keypoints_conf") or []
    ankle_points = []
    for idx in (15, 16):
        if idx >= len(keypoints):
            continue
        confidence = float(confidences[idx]) if idx < len(confidences) else 1.0
        if confidence >= 0.2:
            ankle_points.append((float(keypoints[idx][0]), float(keypoints[idx][1])))
    if ankle_points:
        return [
            sum(point[0] for point in ankle_points) / len(ankle_points),
            sum(point[1] for point in ankle_points) / len(ankle_points),
        ]
    x1, _y1, x2, y2 = [float(v) for v in bbox]
    return [(x1 + x2) * 0.5, y2]


def _collect_foot_anchors(frame):
    anchors = []
    for detection in frame.get("detections") or []:
        if detection.get("class_id") not in (None, 0):
            continue
        point = _footpoint(detection)
        if point is None:
            continue
        anchors.append(
            {
                "frame_idx": int(frame.get("frame_idx") or 0),
                "xy": [round(float(point[0]), 3), round(float(point[1]), 3)],
                "track_id": detection.get("track_id"),
                "active_player_candidate": bool(detection.get("active_player_candidate")),
            }
        )
    return anchors


def _score_region(region, foot_anchors, hoop_regions, *, image_width, image_height):
    bbox = region.bbox_xyxy
    frame_scale = max(float(image_width), float(image_height), 1.0)
    margin = 0.04 * frame_scale
    foot_support = sum(1 for anchor in foot_anchors if _bbox_contains_point(bbox, anchor["xy"], margin_px=margin))
    hoop_support = 0
    expanded_for_hoop = [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin]
    for hoop in hoop_regions:
        if _bbox_contains_point(expanded_for_hoop, _bbox_center(hoop.bbox_xyxy)):
            hoop_support += 1
    area_ratio = _bbox_area(bbox) / max(float(image_width * image_height), 1.0)
    area_score = 1.0 - min(1.0, abs(area_ratio - 0.42) / 0.42)
    return {
        "score": round(
            min(
                1.0,
                0.18
                + 0.34 * min(1.0, foot_support / 8.0)
                + 0.24 * min(1.0, hoop_support / 2.0)
                + 0.14 * area_score
                + 0.10 * region.confidence,
            ),
            4,
        ),
        "foot_anchor_support": int(foot_support),
        "hoop_support": int(hoop_support),
        "area_ratio": round(float(area_ratio), 4),
    }


def _resolve_image_dimensions(frames, video_meta):
    image_width = int((video_meta or {}).get("width") or 0)
    image_height = int((video_meta or {}).get("height") or 0)
    if image_width > 0 and image_height > 0:
        return image_width, image_height
    for frame in frames:
        for detection in frame.get("detections") or []:
            bbox = detection.get("bbox_xyxy") or []
            if len(bbox) == 4:
                image_width = max(image_width, int(max(float(bbox[0]), float(bbox[2]))))
                image_height = max(image_height, int(max(float(bbox[1]), float(bbox[3]))))
    return image_width, image_height


def _build_prior_for_frame_set(frames, *, video_meta, max_sample_frames, pose_segment=None):
    image_width, image_height = _resolve_image_dimensions(frames, video_meta)
    sampled = _sample_frames(frames, max_sample_frames=max(1, int(max_sample_frames)))
    sampled_frame_idxs = [int(frame.get("frame_idx") or 0) for frame in sampled]

    regions = []
    foot_anchors = []
    for frame in sampled:
        regions.extend(_collect_regions(frame, image_width=image_width, image_height=image_height))
        foot_anchors.extend(_collect_foot_anchors(frame))
    regions = _dedupe_regions(regions)

    court_regions = [region for region in regions if "court" in region.source or "play_region" in region.source]
    hoop_regions = [region for region in regions if "hoop" in region.source or "basket" in region.source]

    scored = []
    for region in court_regions:
        score_parts = _score_region(region, foot_anchors, hoop_regions, image_width=image_width, image_height=image_height)
        scored.append(
            {
                "bbox_xyxy": [round(float(v), 3) for v in region.bbox_xyxy],
                "source": region.source,
                "frame_idx": int(region.frame_idx),
                "confidence": round(float(region.confidence), 4),
                **score_parts,
            }
        )
    scored.sort(key=lambda item: (float(item["score"]), int(item["foot_anchor_support"]), int(item["hoop_support"])), reverse=True)

    if scored:
        selected_bbox = scored[0]["bbox_xyxy"]
        status = "ready"
        source = "grounding_region_with_foot_and_hoop_support"
        confidence = float(scored[0]["score"])
    elif foot_anchors:
        xs = [float(anchor["xy"][0]) for anchor in foot_anchors]
        ys = [float(anchor["xy"][1]) for anchor in foot_anchors]
        point_box = [min(xs), min(ys), max(xs), max(ys)]
        selected_bbox = _union_bbox([point_box], image_width=image_width, image_height=image_height, padding_ratio=0.10)
        status = "weak_foot_anchor_only"
        source = "player_foot_anchor_envelope"
        confidence = min(0.42, 0.12 + 0.03 * len(foot_anchors))
    else:
        selected_bbox = None
        status = "insufficient_evidence"
        source = "none"
        confidence = 0.0

    prior = {
        "kind": "target_court_first_pass_v1",
        "status": status,
        "source": source,
        "confidence": round(float(confidence), 4),
        "sampled_frame_idxs": sampled_frame_idxs,
        "sampled_frame_count": len(sampled_frame_idxs),
        "bbox_xyxy": selected_bbox,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "evidence_counts": {
            "candidate_region_count": len(regions),
            "court_region_count": len(court_regions),
            "hoop_region_count": len(hoop_regions),
            "foot_anchor_count": len(foot_anchors),
        },
        "selected_candidate": scored[0] if scored else None,
        "top_candidates": scored[:5],
        "court_dimensions_prior": {
            "preferred": "ncaa",
            "ncaa_ft": NCAA_COURT_DIMENSIONS_FT,
            "nba_ft": NBA_COURT_DIMENSIONS_FT,
        },
        "homography": {
            "status": "insufficient_correspondences",
            "reason": "first_pass_region_prior_without_reliable_court_line_or_keypoint_correspondences",
            "image_to_court": None,
        },
        "intended_use": "reject_or_downweight_background_court_ball_and_player_candidates_before_tracking",
    }
    if pose_segment is not None:
        prior["pose_segment_id"] = int(pose_segment["pose_segment_id"])
        prior["start_frame_idx"] = int(pose_segment["start_frame_idx"])
        prior["end_frame_idx"] = int(pose_segment["end_frame_idx"])
        prior["continuity_segment_id"] = pose_segment.get("continuity_segment_id")
        prior["split_reasons"] = list(pose_segment.get("split_reasons") or [])
        prior["high_motion_frame_count"] = int(pose_segment.get("high_motion_frame_count") or 0)
    return prior


def build_target_court_first_pass(
    frames,
    video_meta=None,
    *,
    max_sample_frames=12,
    max_pose_segment_frames=DEFAULT_MAX_POSE_SEGMENT_FRAMES,
    high_motion_px=DEFAULT_HIGH_MOTION_PX,
):
    """Attach and return camera-pan-aware target-court priors from sampled frames."""
    video_meta = video_meta or {}
    pose_segments = _camera_pose_segments(
        frames,
        max_pose_segment_frames=max(1, int(max_pose_segment_frames)),
        high_motion_px=float(high_motion_px),
    )
    segment_priors = []
    for segment in pose_segments:
        segment_frames = frames[int(segment["start_index"]): int(segment["end_index"]) + 1]
        segment_priors.append(
            _build_prior_for_frame_set(
                segment_frames,
                video_meta=video_meta,
                max_sample_frames=max_sample_frames,
                pose_segment=segment,
            )
        )

    primary = max(segment_priors, key=lambda prior: float(prior.get("confidence") or 0.0), default=None)
    if primary is None:
        primary = _build_prior_for_frame_set(
            frames,
            video_meta=video_meta,
            max_sample_frames=max_sample_frames,
            pose_segment=None,
        )

    prior_by_segment_id = {int(prior["pose_segment_id"]): prior for prior in segment_priors if prior.get("pose_segment_id") is not None}

    for frame in frames:
        pose_segment_id = None
        frame_idx = int(frame.get("frame_idx") or 0)
        for segment in pose_segments:
            if int(segment["start_frame_idx"]) <= frame_idx <= int(segment["end_frame_idx"]):
                pose_segment_id = int(segment["pose_segment_id"])
                break
        frame_prior = prior_by_segment_id.get(pose_segment_id, primary)
        frame["target_court_prior"] = {
            "kind": frame_prior["kind"],
            "status": frame_prior["status"],
            "confidence": frame_prior["confidence"],
            "bbox_xyxy": frame_prior["bbox_xyxy"],
            "source": frame_prior["source"],
            "pose_segment_id": pose_segment_id,
            "start_frame_idx": frame_prior.get("start_frame_idx"),
            "end_frame_idx": frame_prior.get("end_frame_idx"),
        }
    return {
        **primary,
        "kind": "target_court_first_pass_v2",
        "camera_motion_policy": {
            "mode": "time_varying_pose_segments",
            "max_pose_segment_frames": int(max_pose_segment_frames),
            "high_motion_px": round(float(high_motion_px), 3),
            "split_signals": ["continuity_segment_changed", "max_pose_segment_frames", "camera_pan_high_motion"],
        },
        "segment_count": len(segment_priors),
        "segments": segment_priors,
    }

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackletStitcherConfig:
    max_gap: int
    continuity_partition_field: str
    reset_identity_on_discontinuity: bool
    occlusion_first_within_segment: bool


def empty_identity_hypothesis_summary():
    return {
        "groups": [],
        "selected_links": [],
        "candidate_lookup": {},
        "decision_ledger": [],
        "predecessor_candidate_lookup": {},
    }


def _build_identity_repair_meta(link, canonical_track_id, hypothesis_summary, *, recovery_match=None, evidence_source=None):
    repair_meta = {
        "kind": "short_gap_identity_bridge",
        "predecessor_track_id": link["predecessor_track_id"],
        "successor_track_id": link["successor_track_id"],
        "canonical_track_id": canonical_track_id,
        "gap_frames": link["gap_frames"],
        "link_score": link["score"],
        "reasons": link["soft_evidence"],
        "selected_link": link.get("selected_link"),
        "link_candidates": hypothesis_summary.get("predecessor_candidate_lookup", {}).get(int(link["predecessor_track_id"]), []),
    }
    if recovery_match is not None:
        repair_meta["recovery_match"] = {
            "proposal_score": recovery_match["proposal_score"],
            "iou": recovery_match["iou"],
            "center_distance_px": recovery_match["center_distance_px"],
            "match_score": recovery_match["match_score"],
        }
    if evidence_source is not None:
        repair_meta["evidence_source"] = evidence_source
    return repair_meta


def _annotate_hypothesis_groups(frame_index, hypothesis_summary):
    for group in hypothesis_summary["groups"]:
        for frame_idx in range(group["start_frame_idx"], group["end_frame_idx"] + 1):
            frame = frame_index.get(frame_idx)
            if frame is None:
                continue
            frame.setdefault("identity_hypothesis_group_ids", [])
            frame["identity_hypothesis_group_ids"].append(group["group_id"])


def _apply_selected_identity_links(track_frames, hypothesis_summary):
    for link in hypothesis_summary["selected_links"]:
        predecessor_track_id = link["predecessor_track_id"]
        successor_track_id = link["successor_track_id"]
        canonical_track_id = track_frames[predecessor_track_id][0][1].get("identity_track_id", predecessor_track_id)
        repair_meta = _build_identity_repair_meta(link, canonical_track_id, hypothesis_summary)
        for _frame_idx, detection in track_frames[successor_track_id]:
            detection["identity_track_id"] = canonical_track_id
            detection["identity_track_source"] = "repaired"
            detection["identity_repair"] = repair_meta


def _synthesize_intra_track_gaps(frames, frame_index, track_frames, *, config, interpolate_detection):
    for track_id, observations in track_frames.items():
        observations.sort(key=lambda item: item[0])
        for (start_idx, start_det), (end_idx, end_det) in zip(observations, observations[1:]):
            gap = end_idx - start_idx - 1
            if gap <= 0 or gap > config.max_gap:
                continue
            if (
                config.reset_identity_on_discontinuity
                and start_det.get(config.continuity_partition_field) != end_det.get(config.continuity_partition_field)
            ):
                continue
            if not config.occlusion_first_within_segment:
                continue
            if not start_det.get("active_player_candidate") or not end_det.get("active_player_candidate"):
                continue
            for missing_frame_idx in range(start_idx + 1, end_idx):
                frame = frame_index.get(missing_frame_idx)
                if frame is None:
                    continue
                existing_track_ids = {
                    detection.get("track_id")
                    for detection in frame.get("detections", [])
                }
                if track_id in existing_track_ids:
                    continue
                alpha = (missing_frame_idx - start_idx) / float(end_idx - start_idx)
                synthesized = interpolate_detection(
                    start_det,
                    end_det,
                    missing_frame_idx,
                    frame["t_ms"],
                    alpha,
                )
                frame.setdefault("detections", []).append(synthesized)


def _stitch_selected_identity_links(frames, frame_index, hypothesis_summary, *, interpolate_detection, match_discovery_recovery_detection):
    for link in hypothesis_summary["selected_links"]:
        start_idx = link["start_det"]["_frame_idx"]
        end_idx = link["end_det"]["_frame_idx"]
        start_det = link["start_det"]
        end_det = link["end_det"]
        canonical_track_id = start_det.get("identity_track_id", start_det.get("track_id"))
        for missing_frame_idx in range(start_idx + 1, end_idx):
            frame = frame_index.get(missing_frame_idx)
            if frame is None:
                continue
            existing_identity_ids = {
                detection.get("identity_track_id", detection.get("track_id"))
                for detection in frame.get("detections", [])
            }
            if canonical_track_id in existing_identity_ids:
                continue
            alpha = (missing_frame_idx - start_idx) / float(end_idx - start_idx)
            synthesized = interpolate_detection(
                start_det,
                end_det,
                missing_frame_idx,
                frame["t_ms"],
                alpha,
            )
            recovery_match = match_discovery_recovery_detection(
                frame,
                synthesized.get("bbox_xyxy") or [],
            )
            if recovery_match is not None:
                recovered_detection = recovery_match["detection"]
                recovered_detection["identity_track_id"] = canonical_track_id
                recovered_detection["identity_track_source"] = "repaired_recovery"
                recovered_detection["identity_repair"] = _build_identity_repair_meta(
                    link,
                    canonical_track_id,
                    hypothesis_summary,
                    recovery_match=recovery_match,
                    evidence_source="discovery_recovery_proposal",
                )
                continue
            synthesized["identity_track_id"] = canonical_track_id
            synthesized["identity_track_source"] = "repaired"
            synthesized["identity_repair"] = _build_identity_repair_meta(
                link,
                canonical_track_id,
                hypothesis_summary,
            )
            frame.setdefault("detections", []).append(synthesized)


def _finalize_stitched_frames(frames):
    for frame in frames:
        frame["identity_hypothesis_group_ids"] = sorted(set(frame.get("identity_hypothesis_group_ids", [])))
        frame.setdefault("detections", [])
        frame["detections"].sort(
            key=lambda detection: (
                detection.get("identity_track_id") is None,
                detection.get("identity_track_id") if detection.get("identity_track_id") is not None else 1_000_000,
                detection.get("track_id") is None,
                detection.get("track_id") if detection.get("track_id") is not None else 1_000_000,
                not detection.get("synthesized", False),
            )
        )
        for detection in frame.get("detections", []):
            detection.pop("_frame_idx", None)
            detection.pop("_t_ms", None)


def stitch_tracklets(frames, track_frames, hypothesis_summary, *, config, interpolate_detection, match_discovery_recovery_detection):
    frame_index = {frame["frame_idx"]: frame for frame in frames}
    _annotate_hypothesis_groups(frame_index, hypothesis_summary)
    _apply_selected_identity_links(track_frames, hypothesis_summary)
    _synthesize_intra_track_gaps(
        frames,
        frame_index,
        track_frames,
        config=config,
        interpolate_detection=interpolate_detection,
    )
    _stitch_selected_identity_links(
        frames,
        frame_index,
        hypothesis_summary,
        interpolate_detection=interpolate_detection,
        match_discovery_recovery_detection=match_discovery_recovery_detection,
    )
    _finalize_stitched_frames(frames)
    return frames

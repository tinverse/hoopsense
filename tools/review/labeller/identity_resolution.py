from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IdentityResolutionConfig:
    policy_version: str
    max_gap_frames: int
    max_overlap_probe_frames: int
    max_gap_probe_frames: int
    max_assignment_candidates_per_group: int
    max_assignment_hypotheses_per_group: int
    max_global_hypotheses: int
    min_size_consistency: float
    repair_score_threshold: float
    max_court_distance: float
    max_center_distance_ratio: float
    ambiguity_margin: float
    candidate_min_score: float
    continuity_partition_field: str
    reset_identity_on_discontinuity: bool
    hypothesis_max_candidates_per_group: int
    short_gap_link_weights: dict


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _bbox_center_xy(detection):
    bbox_xywh = detection.get("smoothed_bbox_xywh") or detection.get("bbox_xywh")
    if bbox_xywh is not None and len(bbox_xywh) >= 2:
        return float(bbox_xywh[0]), float(bbox_xywh[1])
    bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_size_xy(detection):
    bbox_xywh = detection.get("smoothed_bbox_xywh") or detection.get("bbox_xywh")
    if bbox_xywh is not None and len(bbox_xywh) >= 4:
        return max(float(bbox_xywh[2]), 1.0), max(float(bbox_xywh[3]), 1.0)
    bbox_xyxy = detection.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return max(x2 - x1, 1.0), max(y2 - y1, 1.0)


def _serialize_hard_filter_details(details):
    serialized_details = {}
    for key, value in details.items():
        if isinstance(value, float):
            serialized_details[key] = round(float(value), 4)
        else:
            serialized_details[key] = value
    return serialized_details


def _identity_link_hard_pass(*passed_checks, **details):
    return {
        "status": "pass",
        "rejection_reason": None,
        "details": _serialize_hard_filter_details(details),
        "passed_checks": list(passed_checks),
        "failed_checks": [],
    }


def _identity_link_hard_reject(reason, *, passed_checks=None, **details):
    return {
        "status": "fail",
        "rejection_reason": reason,
        "details": _serialize_hard_filter_details(details),
        "passed_checks": list(passed_checks or []),
        "failed_checks": [reason],
    }


def evaluate_identity_link_candidate(start_det, end_det, gap_frames, fps, config):
    passed_checks = []
    if not start_det.get("active_player_candidate") or not end_det.get("active_player_candidate"):
        return {
            "hard_filter": _identity_link_hard_reject(
                "inactive_endpoint",
                passed_checks=passed_checks,
                predecessor_active=bool(start_det.get("active_player_candidate")),
                successor_active=bool(end_det.get("active_player_candidate")),
            ),
            "soft_evidence": {},
            "score": None,
        }
    passed_checks.append("active_endpoint")

    start_bucket = start_det.get("uniform_bucket")
    end_bucket = end_det.get("uniform_bucket")
    if (
        start_bucket in {"dark", "light"}
        and end_bucket in {"dark", "light"}
        and start_bucket != end_bucket
    ):
        return {
            "hard_filter": _identity_link_hard_reject(
                "uniform_bucket_conflict",
                passed_checks=passed_checks,
                predecessor_uniform_bucket=start_bucket,
                successor_uniform_bucket=end_bucket,
            ),
            "soft_evidence": {},
            "score": None,
        }
    passed_checks.append("uniform_bucket_compatible")

    dt = max(1.0 / max(float(fps), 1.0), 1e-3)
    elapsed_s = max(gap_frames + 1, 1) * dt
    start_cx, start_cy = _bbox_center_xy(start_det)
    end_cx, end_cy = _bbox_center_xy(end_det)
    vel_x, vel_y = (start_det.get("smoothed_velocity_xy") or [0.0, 0.0])[:2]
    predicted_x = float(start_cx) + float(vel_x) * elapsed_s
    predicted_y = float(start_cy) + float(vel_y) * elapsed_s
    center_distance = float(np.linalg.norm([end_cx - predicted_x, end_cy - predicted_y]))

    _start_w, start_h = _bbox_size_xy(start_det)
    _end_w, end_h = _bbox_size_xy(end_det)
    avg_height = max(1.0, (start_h + end_h) * 0.5)
    max_center_distance = max(40.0, avg_height * config.max_center_distance_ratio)
    if center_distance > max_center_distance:
        return {
            "hard_filter": _identity_link_hard_reject(
                "max_center_distance_exceeded",
                passed_checks=passed_checks,
                predicted_center_distance_px=center_distance,
                max_center_distance_px=max_center_distance,
            ),
            "soft_evidence": {
                "predicted_center_distance_px": round(center_distance, 3),
                "max_center_distance_px": round(max_center_distance, 3),
            },
            "score": None,
        }
    passed_checks.append("center_distance_within_limit")
    position_score = _clip01(1.0 - (center_distance / max_center_distance))

    start_w, start_h = _bbox_size_xy(start_det)
    end_w, end_h = _bbox_size_xy(end_det)
    width_ratio = min(start_w, end_w) / max(start_w, end_w, 1.0)
    height_ratio = min(start_h, end_h) / max(start_h, end_h, 1.0)
    size_score = (width_ratio + height_ratio) * 0.5
    if size_score < config.min_size_consistency:
        return {
            "hard_filter": _identity_link_hard_reject(
                "min_size_consistency_failed",
                passed_checks=passed_checks,
                size_score=size_score,
                min_size_consistency=config.min_size_consistency,
            ),
            "soft_evidence": {
                "size_score": round(size_score, 4),
            },
            "score": None,
        }
    passed_checks.append("size_consistency_within_limit")

    start_vel = np.array((start_det.get("smoothed_velocity_xy") or [0.0, 0.0])[:2], dtype=float)
    end_vel = np.array((end_det.get("smoothed_velocity_xy") or [0.0, 0.0])[:2], dtype=float)
    start_speed = float(np.linalg.norm(start_vel))
    end_speed = float(np.linalg.norm(end_vel))
    if start_speed > 1.0 and end_speed > 1.0:
        velocity_score = _clip01((float(np.dot(start_vel, end_vel)) / (start_speed * end_speed) + 1.0) * 0.5)
    else:
        velocity_score = 0.5

    if start_bucket == end_bucket and start_bucket in {"dark", "light"}:
        uniform_score = 1.0
    elif start_bucket == "unknown" or end_bucket == "unknown":
        uniform_score = 0.55
    else:
        uniform_score = 0.0

    court_score = 0.5
    court_distance = None
    if start_det.get("court_xy") and end_det.get("court_xy"):
        court_distance = float(
            np.linalg.norm(
                [
                    float(end_det["court_xy"][0]) - float(start_det["court_xy"][0]),
                    float(end_det["court_xy"][1]) - float(start_det["court_xy"][1]),
                ]
            )
        )
        if court_distance > config.max_court_distance:
            return {
                "hard_filter": _identity_link_hard_reject(
                    "max_court_distance_exceeded",
                    passed_checks=passed_checks,
                    court_distance=court_distance,
                    max_court_distance=config.max_court_distance,
                ),
                "soft_evidence": {
                    "court_distance": round(court_distance, 3),
                },
                "score": None,
            }
        court_score = _clip01(1.0 - (court_distance / config.max_court_distance))
    passed_checks.append("court_distance_within_limit")

    gap_score = _clip01(1.0 - ((gap_frames - 1) / max(config.max_gap_frames, 1)))
    confidence_score = min(float(start_det.get("confidence") or 0.0), float(end_det.get("confidence") or 0.0))
    weights = config.short_gap_link_weights
    link_score = (
        float(weights["predicted_position"]) * position_score
        + float(weights["velocity_alignment"]) * velocity_score
        + float(weights["size_consistency"]) * size_score
        + float(weights["uniform_bucket"]) * uniform_score
        + float(weights["court_distance"]) * court_score
        + float(weights["temporal_gap"]) * gap_score
        + float(weights["confidence_floor"]) * confidence_score
    )
    return {
        "hard_filter": _identity_link_hard_pass(
            *passed_checks,
            temporal_non_overlap=True,
            max_gap_ok=True,
        ),
        "soft_evidence": {
            "position_score": round(position_score, 4),
            "velocity_score": round(velocity_score, 4),
            "size_score": round(size_score, 4),
            "uniform_score": round(uniform_score, 4),
            "court_score": round(court_score, 4),
            "gap_score": round(gap_score, 4),
            "confidence_score": round(confidence_score, 4),
            "predicted_center_distance_px": round(center_distance, 3),
            "max_center_distance_px": round(max_center_distance, 3),
            "court_distance": round(court_distance, 3) if court_distance is not None else None,
        },
        "score": round(float(link_score), 4),
    }


def score_identity_link(start_det, end_det, gap_frames, fps, config):
    evaluation = evaluate_identity_link_candidate(start_det, end_det, gap_frames, fps, config)
    if evaluation["hard_filter"]["status"] != "pass":
        return None
    return {
        "score": evaluation["score"],
        "gap_frames": int(gap_frames),
        "reasons": evaluation["soft_evidence"],
    }


def _build_identity_link_candidate_record(predecessor_track_id, predecessor_last_frame, predecessor_last_det, successor_track_id, successor_first_frame, successor_first_det, fps, *, config, max_gap, candidate_threshold, selection_threshold):
    gap_frames = successor_first_frame - predecessor_last_frame - 1
    if gap_frames < -int(config.max_overlap_probe_frames):
        return None
    if gap_frames > max_gap + int(config.max_gap_probe_frames):
        return None

    if gap_frames < 0:
        evaluation = {
            "hard_filter": _identity_link_hard_reject(
                "temporal_overlap",
                predecessor_end_frame_idx=int(predecessor_last_frame),
                successor_start_frame_idx=int(successor_first_frame),
                gap_frames=int(gap_frames),
            ),
            "soft_evidence": {},
            "score": None,
        }
    elif gap_frames > max_gap:
        evaluation = {
            "hard_filter": _identity_link_hard_reject(
                "max_gap_exceeded",
                predecessor_end_frame_idx=int(predecessor_last_frame),
                successor_start_frame_idx=int(successor_first_frame),
                gap_frames=int(gap_frames),
                max_gap_frames=int(max_gap),
            ),
            "soft_evidence": {},
            "score": None,
        }
    else:
        segment_mismatch = (
            config.reset_identity_on_discontinuity
            and predecessor_last_det.get(config.continuity_partition_field) != successor_first_det.get(config.continuity_partition_field)
        )
        if segment_mismatch:
            evaluation = {
                "hard_filter": _identity_link_hard_reject(
                    "continuity_partition_mismatch",
                    predecessor_segment_id=predecessor_last_det.get(config.continuity_partition_field),
                    successor_segment_id=successor_first_det.get(config.continuity_partition_field),
                    passed_checks=["temporal_non_overlap", "max_gap_ok"],
                ),
                "soft_evidence": {},
                "score": None,
            }
        else:
            evaluation = evaluate_identity_link_candidate(predecessor_last_det, successor_first_det, gap_frames, fps, config)

    score = evaluation["score"]
    passes_candidate_threshold = bool(score is not None and float(score) >= float(candidate_threshold))
    passes_selection_threshold = bool(score is not None and float(score) >= float(selection_threshold))
    if evaluation["hard_filter"]["status"] != "pass":
        selection_status = "rejected_hard"
    elif not passes_candidate_threshold:
        selection_status = "below_candidate_threshold"
    else:
        selection_status = "candidate"
    return {
        "predecessor_track_id": predecessor_track_id,
        "successor_track_id": successor_track_id,
        "start_frame_idx": int(predecessor_last_frame),
        "end_frame_idx": int(successor_first_frame),
        "gap_frames": int(gap_frames),
        "start_det": predecessor_last_det,
        "end_det": successor_first_det,
        "hard_filter": evaluation["hard_filter"],
        "soft_evidence": evaluation["soft_evidence"],
        "score": score,
        "passes_candidate_threshold": passes_candidate_threshold,
        "passes_selection_threshold": passes_selection_threshold,
        "candidate_threshold": round(float(candidate_threshold), 4),
        "selection_threshold": round(float(selection_threshold), 4),
        "selection_status": selection_status,
        "selected_link": None,
    }


def generate_identity_link_candidates(track_frames, fps, *, config, max_gap, candidate_threshold, selection_threshold):
    track_ids = sorted(track_frames)
    candidates = []
    candidate_id = 0
    for predecessor_track_id in track_ids:
        predecessor_obs = sorted(track_frames.get(predecessor_track_id, []), key=lambda item: item[0])
        if not predecessor_obs:
            continue
        predecessor_last_frame, predecessor_last_det = predecessor_obs[-1]
        for successor_track_id in track_ids:
            if successor_track_id == predecessor_track_id:
                continue
            successor_obs = sorted(track_frames.get(successor_track_id, []), key=lambda item: item[0])
            if not successor_obs:
                continue
            successor_first_frame, successor_first_det = successor_obs[0]
            candidate = _build_identity_link_candidate_record(
                predecessor_track_id,
                predecessor_last_frame,
                predecessor_last_det,
                successor_track_id,
                successor_first_frame,
                successor_first_det,
                fps,
                config=config,
                max_gap=max_gap,
                candidate_threshold=candidate_threshold,
                selection_threshold=selection_threshold,
            )
            if candidate is None:
                continue
            candidate["candidate_id"] = f"h{candidate_id}"
            candidates.append(candidate)
            candidate_id += 1
    return candidates


def _candidate_sort_key(candidate):
    score = float(candidate.get("score") or 0.0)
    return (-score, int(candidate.get("gap_frames") or 0), int(candidate.get("successor_track_id") or 0), int(candidate.get("predecessor_track_id") or 0))


def serialize_identity_link_candidate(candidate, *, best_score=None):
    score = candidate.get("score")
    return {
        "candidate_id": candidate["candidate_id"],
        "predecessor_track_id": int(candidate["predecessor_track_id"]),
        "successor_track_id": int(candidate["successor_track_id"]),
        "start_frame_idx": int(candidate["start_frame_idx"]),
        "end_frame_idx": int(candidate["end_frame_idx"]),
        "gap_frames": int(candidate["gap_frames"]),
        "score": round(float(score), 4) if score is not None else None,
        "score_margin_to_best": round(float(best_score - score), 4) if (best_score is not None and score is not None) else None,
        "passes_candidate_threshold": bool(candidate.get("passes_candidate_threshold")),
        "passes_selection_threshold": bool(candidate.get("passes_selection_threshold")),
        "selection_status": candidate.get("selection_status"),
        "hard_filter": candidate.get("hard_filter"),
        "soft_evidence": candidate.get("soft_evidence") or {},
        "selected_link": candidate.get("selected_link"),
    }


def _build_candidate_components(candidates):
    if not candidates:
        return []
    adjacency = {candidate["candidate_id"]: set() for candidate in candidates}
    for idx, left in enumerate(candidates):
        for right in candidates[idx + 1 :]:
            if left["predecessor_track_id"] == right["predecessor_track_id"] or left["successor_track_id"] == right["successor_track_id"]:
                adjacency[left["candidate_id"]].add(right["candidate_id"])
                adjacency[right["candidate_id"]].add(left["candidate_id"])

    candidate_lookup = {candidate["candidate_id"]: candidate for candidate in candidates}
    components = []
    seen = set()
    for candidate in sorted(candidates, key=_candidate_sort_key):
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            continue
        stack = [candidate_id]
        component_ids = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component_ids.append(current)
            stack.extend(adjacency[current] - seen)
        components.append([candidate_lookup[item] for item in sorted(component_ids)])
    return components


def _enumerate_component_assignment_hypotheses(group_candidates, config):
    ranked_candidates = sorted(group_candidates, key=_candidate_sort_key)[: int(config.max_assignment_candidates_per_group)]
    hypotheses = []

    def dfs(index, used_predecessors, used_successors, selected_candidates, total_score):
        if index >= len(ranked_candidates):
            hypotheses.append({
                "candidate_ids": tuple(candidate["candidate_id"] for candidate in selected_candidates),
                "candidate_records": tuple(selected_candidates),
                "link_count": len(selected_candidates),
                "total_score": round(float(total_score), 4),
            })
            return

        candidate = ranked_candidates[index]
        dfs(index + 1, used_predecessors, used_successors, selected_candidates, total_score)

        predecessor_track_id = candidate["predecessor_track_id"]
        successor_track_id = candidate["successor_track_id"]
        if predecessor_track_id in used_predecessors or successor_track_id in used_successors:
            return

        used_predecessors.add(predecessor_track_id)
        used_successors.add(successor_track_id)
        selected_candidates.append(candidate)
        dfs(index + 1, used_predecessors, used_successors, selected_candidates, total_score + float(candidate.get("score") or 0.0))
        selected_candidates.pop()
        used_predecessors.remove(predecessor_track_id)
        used_successors.remove(successor_track_id)

    dfs(0, set(), set(), [], 0.0)

    unique = {}
    for hypothesis in hypotheses:
        key = tuple(sorted(hypothesis["candidate_ids"]))
        if key not in unique or float(hypothesis["total_score"]) > float(unique[key]["total_score"]):
            unique[key] = hypothesis

    ranked_hypotheses = sorted(
        unique.values(),
        key=lambda item: (
            -float(item["total_score"]),
            -int(item["link_count"]),
            list(item["candidate_ids"]),
        ),
    )
    return ranked_hypotheses[: int(config.max_assignment_hypotheses_per_group)]


def _serialize_assignment_hypothesis(hypothesis, *, rank, best_total_score, selected_hypothesis_id, next_best_score=None):
    total_score = float(hypothesis["total_score"])
    hypothesis_id = hypothesis["hypothesis_id"]
    return {
        "hypothesis_id": hypothesis_id,
        "rank": int(rank),
        "candidate_ids": list(hypothesis["candidate_ids"]),
        "link_count": int(hypothesis["link_count"]),
        "total_score": round(total_score, 4),
        "score_margin_to_best": round(float(best_total_score - total_score), 4),
        "score_delta_to_next": round(float(total_score - next_best_score), 4) if next_best_score is not None else None,
        "selected": hypothesis_id == selected_hypothesis_id,
    }


def _rank_global_hypothesis_states(states):
    return sorted(
        states,
        key=lambda item: (
            -float(item["total_score"]),
            -len(item["selected_candidate_ids"]),
            tuple(item["group_hypothesis_ids"]),
        ),
    )


def _build_global_identity_hypotheses(assignment_components, candidate_lookup, *, config):
    if not assignment_components:
        return []

    base_group_hypothesis_ids = []
    base_selected_candidate_ids = set()
    base_total_score = 0.0
    variable_groups = []

    for component in assignment_components:
        component_hypotheses = list(component.get("assignment_hypotheses") or [])
        if not component_hypotheses:
            continue

        selected_hypothesis = None
        if component.get("selected_hypothesis_id") is not None:
            selected_hypothesis = next(
                (
                    hypothesis
                    for hypothesis in component_hypotheses
                    if hypothesis["hypothesis_id"] == component["selected_hypothesis_id"]
                ),
                None,
            )
        if selected_hypothesis is None:
            selected_hypothesis = component_hypotheses[0]

        if component.get("status") == "deferred_ambiguous":
            variable_groups.append({
                "group_id": component["group_id"],
                "hypotheses": component_hypotheses,
            })
            continue

        base_group_hypothesis_ids.append(selected_hypothesis["hypothesis_id"])
        base_selected_candidate_ids.update(selected_hypothesis.get("candidate_ids") or [])
        base_total_score += float(selected_hypothesis.get("total_score") or 0.0)

    states = [{
        "group_hypothesis_ids": tuple(base_group_hypothesis_ids),
        "selected_candidate_ids": set(base_selected_candidate_ids),
        "total_score": round(float(base_total_score), 4),
        "ambiguous_group_ids": tuple(),
    }]

    for variable_group in variable_groups:
        expanded_states = []
        for state in states:
            for hypothesis in variable_group["hypotheses"]:
                candidate_ids = set(hypothesis.get("candidate_ids") or [])
                if state["selected_candidate_ids"] & candidate_ids:
                    continue
                expanded_states.append({
                    "group_hypothesis_ids": tuple(list(state["group_hypothesis_ids"]) + [hypothesis["hypothesis_id"]]),
                    "selected_candidate_ids": set(state["selected_candidate_ids"]) | candidate_ids,
                    "total_score": round(float(state["total_score"]) + float(hypothesis.get("total_score") or 0.0), 4),
                    "ambiguous_group_ids": tuple(list(state["ambiguous_group_ids"]) + [variable_group["group_id"]]),
                })
        deduped = {}
        for state in expanded_states:
            key = (tuple(state["group_hypothesis_ids"]), tuple(sorted(state["selected_candidate_ids"])))
            previous = deduped.get(key)
            if previous is None or float(state["total_score"]) > float(previous["total_score"]):
                deduped[key] = state
        states = _rank_global_hypothesis_states(list(deduped.values()))[: int(config.max_global_hypotheses)]

    if not states:
        return []

    ranked_states = _rank_global_hypothesis_states(states)[: int(config.max_global_hypotheses)]
    best_total_score = float(ranked_states[0]["total_score"])
    normalizer = sum(math.exp(float(state["total_score"]) - best_total_score) for state in ranked_states)
    serialized = []
    for rank, state in enumerate(ranked_states, start=1):
        total_score = float(state["total_score"])
        serialized.append({
            "global_hypothesis_id": f"gh{rank - 1}",
            "rank": int(rank),
            "group_hypothesis_ids": list(state["group_hypothesis_ids"]),
            "ambiguous_group_ids": list(state["ambiguous_group_ids"]),
            "selected_candidate_ids": sorted(state["selected_candidate_ids"]),
            "selected_link_count": len(state["selected_candidate_ids"]),
            "total_score": round(total_score, 4),
            "score_margin_to_best": round(best_total_score - total_score, 4),
            "score_share": round(math.exp(total_score - best_total_score) / normalizer, 4) if normalizer > 0.0 else 0.0,
            "committed_only": len(state["ambiguous_group_ids"]) == 0,
        })
    return serialized


def _resolve_canonical_track_id(track_id, successor_to_predecessor):
    canonical_track_id = int(track_id)
    seen = set()
    while canonical_track_id in successor_to_predecessor and canonical_track_id not in seen:
        seen.add(canonical_track_id)
        canonical_track_id = int(successor_to_predecessor[canonical_track_id])
    return canonical_track_id


def _materialize_identity_chain(canonical_track_id, predecessor_to_successor):
    chain = [int(canonical_track_id)]
    current_track_id = int(canonical_track_id)
    seen = {current_track_id}
    while current_track_id in predecessor_to_successor:
        current_track_id = int(predecessor_to_successor[current_track_id])
        if current_track_id in seen:
            break
        seen.add(current_track_id)
        chain.append(current_track_id)
    return chain


def _build_track_identity_options(global_hypotheses, assignment_components, candidate_lookup):
    if not global_hypotheses or not assignment_components:
        return []

    relevant_track_ids = sorted({
        int(candidate_lookup[candidate_id]["predecessor_track_id"])
        for component in assignment_components
        for candidate_id in component.get("component_candidate_ids", [])
    } | {
        int(candidate_lookup[candidate_id]["successor_track_id"])
        for component in assignment_components
        for candidate_id in component.get("component_candidate_ids", [])
    })
    if not relevant_track_ids:
        return []

    options_by_track = {track_id: {} for track_id in relevant_track_ids}
    for global_hypothesis in global_hypotheses:
        successor_to_predecessor = {}
        predecessor_to_successor = {}
        for candidate_id in global_hypothesis.get("selected_candidate_ids") or []:
            candidate = candidate_lookup.get(candidate_id)
            if candidate is None:
                continue
            predecessor_track_id = int(candidate["predecessor_track_id"])
            successor_track_id = int(candidate["successor_track_id"])
            predecessor_to_successor[predecessor_track_id] = successor_track_id
            successor_to_predecessor[successor_track_id] = predecessor_track_id

        for track_id in relevant_track_ids:
            canonical_track_id = _resolve_canonical_track_id(track_id, successor_to_predecessor)
            chain_track_ids = _materialize_identity_chain(canonical_track_id, predecessor_to_successor)
            option = options_by_track[track_id].setdefault(
                canonical_track_id,
                {
                    "canonical_track_id": canonical_track_id,
                    "chain_track_ids": chain_track_ids,
                    "hypothesis_ids": [],
                    "group_hypothesis_ids": set(),
                    "support_share": 0.0,
                    "best_total_score": float(global_hypothesis["total_score"]),
                    "best_score_margin_to_best": float(global_hypothesis["score_margin_to_best"]),
                },
            )
            option["hypothesis_ids"].append(global_hypothesis["global_hypothesis_id"])
            option["group_hypothesis_ids"].update(global_hypothesis.get("group_hypothesis_ids") or [])
            option["support_share"] += float(global_hypothesis.get("score_share") or 0.0)
            option["best_total_score"] = max(option["best_total_score"], float(global_hypothesis["total_score"]))
            option["best_score_margin_to_best"] = min(
                option["best_score_margin_to_best"],
                float(global_hypothesis["score_margin_to_best"]),
            )

    serialized = []
    for track_id in relevant_track_ids:
        raw_options = list(options_by_track[track_id].values())
        ranked_options = sorted(
            raw_options,
            key=lambda option: (
                -float(option["support_share"]),
                -float(option["best_total_score"]),
                int(option["canonical_track_id"]),
            ),
        )
        serialized_options = []
        for option in ranked_options:
            serialized_options.append({
                "canonical_track_id": int(option["canonical_track_id"]),
                "chain_track_ids": [int(value) for value in option["chain_track_ids"]],
                "hypothesis_ids": sorted(option["hypothesis_ids"]),
                "group_hypothesis_ids": sorted(option["group_hypothesis_ids"]),
                "support_share": round(float(option["support_share"]), 4),
                "best_total_score": round(float(option["best_total_score"]), 4),
                "best_score_margin_to_best": round(float(option["best_score_margin_to_best"]), 4),
            })
        serialized.append({
            "track_id": int(track_id),
            "is_ambiguous": len(serialized_options) > 1,
            "option_count": len(serialized_options),
            "best_canonical_track_id": serialized_options[0]["canonical_track_id"] if serialized_options else int(track_id),
            "options": serialized_options,
        })
    return serialized


def resolve_identity_links(track_frames, fps, *, config, max_gap, candidate_threshold, selection_threshold):
    candidate_records = generate_identity_link_candidates(
        track_frames,
        fps,
        config=config,
        max_gap=max_gap,
        candidate_threshold=candidate_threshold,
        selection_threshold=selection_threshold,
    )
    candidate_lookup = {candidate["candidate_id"]: candidate for candidate in candidate_records}

    candidate_level_candidates = [
        candidate
        for candidate in candidate_records
        if candidate["hard_filter"]["status"] == "pass" and candidate["passes_candidate_threshold"]
    ]
    assignment_level_candidates = [
        candidate
        for candidate in candidate_level_candidates
        if candidate["passes_selection_threshold"]
    ]

    component_candidates = _build_candidate_components(candidate_level_candidates)
    assignment_components = []
    selected_pairs = set()
    selected_links = []
    assignment_level_candidate_ids = {candidate["candidate_id"] for candidate in assignment_level_candidates}

    for group_index, group_candidates in enumerate(component_candidates):
        assignment_group_candidates = [
            candidate
            for candidate in group_candidates
            if candidate["candidate_id"] in assignment_level_candidate_ids
        ]
        hypotheses = _enumerate_component_assignment_hypotheses(assignment_group_candidates, config)
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            hypothesis["hypothesis_id"] = f"g{group_index}a{hypothesis_index}"

        best_hypothesis = hypotheses[0] if hypotheses else None
        second_best_score = float(hypotheses[1]["total_score"]) if len(hypotheses) > 1 else 0.0
        best_total_score = float(best_hypothesis["total_score"]) if best_hypothesis is not None else 0.0
        hypothesis_score_delta = round(best_total_score - second_best_score, 4) if best_hypothesis is not None else None
        is_ambiguous = bool(best_hypothesis and hypothesis_score_delta <= float(config.ambiguity_margin))
        selected_hypothesis_id = None
        selected_candidate_ids = set()
        if best_hypothesis is not None and best_hypothesis["link_count"] > 0 and not is_ambiguous:
            selected_hypothesis_id = best_hypothesis["hypothesis_id"]
            selected_candidate_ids = set(best_hypothesis["candidate_ids"])
            for candidate in best_hypothesis["candidate_records"]:
                candidate["selection_status"] = "selected"
                candidate["selected_link"] = {
                    "policy_version": config.policy_version,
                    "confidence_delta": hypothesis_score_delta,
                    "best_competitor_score": round(second_best_score, 4),
                    "is_ambiguous": False,
                    "hypothesis_id": selected_hypothesis_id,
                    "assignment_total_score": round(best_total_score, 4),
                }
                selected_pairs.add((candidate["predecessor_track_id"], candidate["successor_track_id"]))
                selected_links.append(candidate)
        elif best_hypothesis is not None and best_hypothesis["link_count"] > 0:
            for candidate in best_hypothesis["candidate_records"]:
                candidate["selection_status"] = "deferred_ambiguous"

        alternative_candidate_ids = set()
        for hypothesis in hypotheses[1:] if selected_hypothesis_id is not None else hypotheses[1:] if best_hypothesis is not None else []:
            alternative_candidate_ids.update(hypothesis["candidate_ids"])
        for candidate in group_candidates:
            if candidate.get("selection_status") == "selected":
                continue
            if candidate.get("selection_status") == "deferred_ambiguous":
                continue
            if candidate["hard_filter"]["status"] != "pass":
                candidate["selection_status"] = "rejected_hard"
            elif not candidate["passes_candidate_threshold"]:
                candidate["selection_status"] = "below_candidate_threshold"
            elif not candidate["passes_selection_threshold"]:
                candidate["selection_status"] = "candidate"
            elif candidate["candidate_id"] in alternative_candidate_ids:
                candidate["selection_status"] = "alternate_hypothesis"
            else:
                candidate["selection_status"] = candidate.get("selection_status") or "candidate"

        serialized_hypotheses = []
        for rank, hypothesis in enumerate(hypotheses, start=1):
            next_best = float(hypotheses[rank]["total_score"]) if rank < len(hypotheses) else None
            serialized_hypotheses.append(
                _serialize_assignment_hypothesis(
                    hypothesis,
                    rank=rank,
                    best_total_score=best_total_score,
                    selected_hypothesis_id=selected_hypothesis_id,
                    next_best_score=next_best,
                )
            )

        component_status = "deferred"
        if selected_hypothesis_id is not None:
            component_status = "selected"
        elif best_hypothesis is not None and best_hypothesis["link_count"] > 0 and is_ambiguous:
            component_status = "deferred_ambiguous"

        component_best_candidate_score = max((float(candidate["score"]) for candidate in group_candidates if candidate.get("score") is not None), default=None)
        assignment_components.append(
            {
                "group_id": f"g{group_index}",
                "status": component_status,
                "component_candidate_ids": [candidate["candidate_id"] for candidate in group_candidates],
                "display_candidates": [serialize_identity_link_candidate(candidate, best_score=component_best_candidate_score) for candidate in sorted(group_candidates, key=_candidate_sort_key)[: int(config.hypothesis_max_candidates_per_group)]],
                "assignment_hypotheses": serialized_hypotheses,
                "selected_hypothesis_id": selected_hypothesis_id,
                "hypothesis_score_delta": hypothesis_score_delta,
                "best_hypothesis_score": round(best_total_score, 4) if best_hypothesis is not None else None,
                "selected_candidate_count": len(selected_candidate_ids),
            }
        )

    predecessor_groups = {}
    for candidate in candidate_records:
        predecessor_groups.setdefault(candidate["predecessor_track_id"], []).append(candidate)
    status_rank = {
        "selected": 0,
        "deferred_ambiguous": 1,
        "alternate_hypothesis": 2,
        "alternate": 3,
        "blocked_by_assignment": 4,
        "below_candidate_threshold": 5,
        "rejected_hard": 6,
        "candidate": 7,
        None: 8,
    }
    decision_ledger = []
    for predecessor_track_id in sorted(predecessor_groups):
        group_candidates = predecessor_groups[predecessor_track_id]
        best_score = max((float(candidate["score"]) for candidate in group_candidates if candidate.get("score") is not None), default=None)
        ranked = sorted(
            group_candidates,
            key=lambda candidate: (
                status_rank.get(candidate.get("selection_status"), 99),
                *_candidate_sort_key(candidate),
            ),
        )
        selected_candidate = next((candidate for candidate in ranked if candidate.get("selection_status") == "selected"), None)
        decision_ledger.append(
            {
                "predecessor_track_id": int(predecessor_track_id),
                "selected_candidate_id": selected_candidate["candidate_id"] if selected_candidate is not None else None,
                "candidate_count": len(group_candidates),
                "candidates": [serialize_identity_link_candidate(candidate, best_score=best_score) for candidate in ranked[:5]],
            }
        )

    predecessor_candidate_lookup = {int(item["predecessor_track_id"]): item["candidates"] for item in decision_ledger}
    global_hypotheses = _build_global_identity_hypotheses(assignment_components, candidate_lookup, config=config)
    track_identity_options = _build_track_identity_options(global_hypotheses, assignment_components, candidate_lookup)
    return {
        "candidate_records": candidate_records,
        "candidate_lookup": candidate_lookup,
        "selected_links": selected_links,
        "selected_pairs": selected_pairs,
        "decision_ledger": decision_ledger,
        "predecessor_candidate_lookup": predecessor_candidate_lookup,
        "assignment_components": assignment_components,
        "global_hypotheses": global_hypotheses,
        "track_identity_options": track_identity_options,
    }


def build_identity_hypothesis_summary(track_frames, fps, *, config, max_gap):
    resolved = resolve_identity_links(
        track_frames,
        fps,
        config=config,
        max_gap=max_gap,
        candidate_threshold=config.candidate_min_score,
        selection_threshold=config.repair_score_threshold,
    )

    if not resolved["assignment_components"]:
        return {
            "groups": [],
            "selected_links": resolved["selected_links"],
            "candidate_lookup": resolved["candidate_lookup"],
            "decision_ledger": resolved["decision_ledger"],
            "predecessor_candidate_lookup": resolved["predecessor_candidate_lookup"],
            "global_hypotheses": resolved.get("global_hypotheses") or [],
            "track_identity_options": resolved.get("track_identity_options") or [],
        }

    groups = []
    for component in resolved["assignment_components"]:
        group_candidates = [resolved["candidate_lookup"][candidate_id] for candidate_id in component["component_candidate_ids"]]
        groups.append(
            {
                "group_id": component["group_id"],
                "status": component["status"],
                "start_frame_idx": int(min(candidate["start_frame_idx"] for candidate in group_candidates)),
                "end_frame_idx": int(max(candidate["end_frame_idx"] for candidate in group_candidates)),
                "predecessor_track_ids": sorted({int(candidate["predecessor_track_id"]) for candidate in group_candidates}),
                "successor_track_ids": sorted({int(candidate["successor_track_id"]) for candidate in group_candidates}),
                "candidate_count": len(group_candidates),
                "selected_candidate_count": int(component["selected_candidate_count"]),
                "selected_hypothesis_id": component["selected_hypothesis_id"],
                "hypothesis_score_delta": component["hypothesis_score_delta"],
                "best_hypothesis_score": component["best_hypothesis_score"],
                "assignment_hypotheses": component["assignment_hypotheses"],
                "candidates": component["display_candidates"],
            }
        )

    return {
        "groups": groups,
        "selected_links": resolved["selected_links"],
        "candidate_lookup": resolved["candidate_lookup"],
        "decision_ledger": resolved["decision_ledger"],
        "predecessor_candidate_lookup": resolved["predecessor_candidate_lookup"],
        "global_hypotheses": resolved.get("global_hypotheses") or [],
        "track_identity_options": resolved.get("track_identity_options") or [],
    }

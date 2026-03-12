"""Lightweight perception helpers that should remain importable in tests."""

from __future__ import annotations

import numpy as np


def match_pose_to_box(box, poses, frame_w, frame_h):
    if not poses or len(poses) == 0:
        return None
    cx, cy, w, h = box
    if cx <= 1.1 and cy <= 1.1 and w <= 1.1 and h <= 1.1:
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
    else:
        x1, y1 = (cx - w / 2) / frame_w, (cy - h / 2) / frame_h
        x2, y2 = (cx + w / 2) / frame_w, (cy + h / 2) / frame_h
    best_idx = -1
    max_in_box = -1
    for i, pose in enumerate(poses):
        pose_np = np.array(pose)
        valid_kpts = pose_np[np.all(pose_np > 0, axis=1)]
        if len(valid_kpts) == 0:
            continue
        in_box = 0
        for kpt in valid_kpts:
            if x1 <= kpt[0] <= x2 and y1 <= kpt[1] <= y2:
                in_box += 1
        ratio = in_box / len(valid_kpts)
        if ratio > max_in_box:
            max_in_box = ratio
            best_idx = i
    if best_idx != -1 and max_in_box > 0.3:
        return poses[best_idx].tolist()
    return None

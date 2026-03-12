import json
from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def project_pixel_to_court(u: float, v: float, h_matrix: np.ndarray) -> np.ndarray:
    """Project a pixel coordinate onto the court plane using a 3x3 homography."""
    p = np.array([u, v, 1.0], dtype=float)
    p_world = np.asarray(h_matrix, dtype=float) @ p
    if abs(p_world[2]) < 1e-6:
        raise ValueError("Singular homography projection (w=0)")
    return p_world[:2] / p_world[2]


def resolve_floor_point(kpts_2d: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    """Resolve the player's floor anchor from the ankle midpoint."""
    if len(kpts_2d) < 17:
        raise ValueError("Expected 17 keypoints for floor resolution")
    l_ank, r_ank = np.asarray(kpts_2d[15], dtype=float), np.asarray(kpts_2d[16], dtype=float)
    return project_pixel_to_court((l_ank[0] + r_ank[0]) / 2.0, (l_ank[1] + r_ank[1]) / 2.0, h_matrix)


def lift_keypoints_to_3d(kpts_2d: np.ndarray, h_matrix: np.ndarray, *, z_scale: float = 0.75) -> np.ndarray:
    """
    Stage-1 single-camera lifter.

    The floor anchor comes from the ankle midpoint. Each keypoint gets its own court-plane
    XY projection, and Z is estimated from the projected Y displacement relative to the floor.
    """
    kpts_2d = np.asarray(kpts_2d, dtype=float)
    if kpts_2d.shape != (17, 2):
        raise ValueError(f"Expected (17, 2) keypoints, got {kpts_2d.shape}")

    floor_xy = resolve_floor_point(kpts_2d, h_matrix)
    lifted = []
    for u, v in kpts_2d:
        world_xy = project_pixel_to_court(float(u), float(v), h_matrix)
        z_est = abs(world_xy[1] - floor_xy[1]) * z_scale
        lifted.append([world_xy[0], world_xy[1], z_est])
    return np.asarray(lifted, dtype=float)


def homography_sanity(h_matrix: np.ndarray) -> dict[str, float | bool]:
    h_matrix = np.asarray(h_matrix, dtype=float)
    if h_matrix.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) homography matrix, got {h_matrix.shape}")
    det = float(np.linalg.det(h_matrix))
    return {
        "finite": bool(np.isfinite(h_matrix).all()),
        "determinant": det,
        "non_singular": abs(det) > 1e-9,
    }


@dataclass
class GeometryReadinessReport:
    homography_finite: bool
    homography_non_singular: bool
    homography_determinant: float
    sequence_count: int
    frame_count: int
    keypoint_count: int
    mean_height_cm: float
    max_height_cm: float
    ankle_grounding_error_cm: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def build_geometry_readiness_report(
    sequences: Iterable[np.ndarray], h_matrix: np.ndarray, *, z_scale: float = 0.75
) -> GeometryReadinessReport:
    sequences = [np.asarray(seq, dtype=float) for seq in sequences]
    sanity = homography_sanity(h_matrix)

    total_frames = 0
    total_keypoints = 0
    heights = []
    ankle_errors = []

    for seq in sequences:
        if seq.ndim != 3 or seq.shape[1:] != (17, 2):
            raise ValueError(f"Expected sequence shape (T, 17, 2), got {seq.shape}")
        total_frames += seq.shape[0]
        total_keypoints += seq.shape[0] * seq.shape[1]
        for frame in seq:
            lifted = lift_keypoints_to_3d(frame, h_matrix, z_scale=z_scale)
            heights.extend(lifted[:, 2].tolist())
            ankle_errors.append(abs(float(lifted[15, 2])))
            ankle_errors.append(abs(float(lifted[16, 2])))

    if not heights:
        raise ValueError("At least one frame is required for geometry readiness")

    return GeometryReadinessReport(
        homography_finite=bool(sanity["finite"]),
        homography_non_singular=bool(sanity["non_singular"]),
        homography_determinant=float(sanity["determinant"]),
        sequence_count=len(sequences),
        frame_count=total_frames,
        keypoint_count=total_keypoints,
        mean_height_cm=float(np.mean(heights)),
        max_height_cm=float(np.max(heights)),
        ankle_grounding_error_cm=float(np.mean(ankle_errors)),
    )

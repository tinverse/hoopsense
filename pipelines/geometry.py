import numpy as np
from typing import Dict, Union


def project_pixel_to_court(u: float, v: float, h_matrix: np.ndarray) -> np.ndarray:
    """Projects pixel (u, v) to world (x, y) using a 3x3 homography matrix."""
    pixel_point = np.array([u, v, 1.0])
    world_point = h_matrix @ pixel_point
    return world_point[:2] / (world_point[2] + 1e-6)


def lift_keypoints_to_3d(kpts_2d: np.ndarray,
                         h_matrix: np.ndarray,
                         *,
                         z_scale: float = 0.75) -> np.ndarray:
    """
    Lift 2D normalized keypoints to 3D.
    Note: Stage 1 heuristic based on world-Y distance from feet.
    """
    kpts_2d = np.asarray(kpts_2d, dtype=float)
    if kpts_2d.shape != (17, 2):
        raise ValueError(f"Expected (17, 2) pose, got {kpts_2d.shape}")

    # Midpoint of ankles (indices 15, 16) as the floor anchor
    floor_anchor = (kpts_2d[15] + kpts_2d[16]) / 2.0
    floor_xy = project_pixel_to_court(floor_anchor[0], floor_anchor[1], h_matrix)

    lifted = []
    for u, v in kpts_2d:
        world_xy = project_pixel_to_court(u, v, h_matrix)
        # Z is estimated by world-Y delta from anchor * perspective scale
        z_est = abs(world_xy[1] - floor_xy[1]) * z_scale
        lifted.append([world_xy[0], world_xy[1], z_est])

    return np.asarray(lifted, dtype=float)


def homography_sanity(h_matrix: np.ndarray) -> Dict[str, Union[float, bool]]:
    h_matrix = np.asarray(h_matrix, dtype=float)
    if h_matrix.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) homography matrix, got {h_matrix.shape}")

    det = np.linalg.det(h_matrix)
    return {
        "determinant": float(det),
        "invertible": bool(abs(det) > 1e-6)
    }


def play_region_prior_for_point(
    play_region_summary: Dict[str, object] | None,
    u: float,
    v: float,
) -> float:
    """Score whether an image-space point falls inside the current play-region prior."""
    if not play_region_summary:
        return 0.0
    mask_grid = play_region_summary.get("mask_grid")
    mask_shape = play_region_summary.get("mask_shape")
    if not mask_grid or not mask_shape or len(mask_shape) != 2:
        return 0.0
    height, width = int(mask_shape[0]), int(mask_shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    image_width = max(float(play_region_summary.get("image_width", width)), 1.0)
    image_height = max(float(play_region_summary.get("image_height", height)), 1.0)
    gx = min(width - 1, max(0, int((float(u) / image_width) * width)))
    gy = min(height - 1, max(0, int((float(v) / image_height) * height)))
    return float(mask_grid[gy][gx])


def geometry_evidence_gate(
    play_region_summary: Dict[str, object] | None,
    u: float,
    v: float,
    *,
    min_prior: float = 0.5,
) -> Dict[str, Union[float, bool]]:
    """Return whether a point is supported enough to count as in-play geometry evidence."""
    prior = play_region_prior_for_point(play_region_summary, u, v)
    return {
        "play_region_prior": round(float(prior), 4),
        "geometry_region_ok": bool(prior >= float(min_prior)),
    }

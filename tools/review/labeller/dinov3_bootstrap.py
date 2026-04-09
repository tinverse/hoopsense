import importlib.util
import math
from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _two_means(features, iterations=12):
    """Split dense features into two coarse clusters without sklearn."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    if len(flat) < 2:
        return np.zeros((features.shape[0], features.shape[1]), dtype=np.uint8)

    center_a = flat[0].copy()
    center_b = flat[-1].copy()
    if np.allclose(center_a, center_b):
        center_b = flat[len(flat) // 2].copy()

    labels = np.zeros(len(flat), dtype=np.int32)
    for _ in range(iterations):
        dist_a = np.linalg.norm(flat - center_a, axis=1)
        dist_b = np.linalg.norm(flat - center_b, axis=1)
        labels = (dist_b < dist_a).astype(np.int32)
        if np.any(labels == 0):
            center_a = flat[labels == 0].mean(axis=0)
        if np.any(labels == 1):
            center_b = flat[labels == 1].mean(axis=0)
    return labels.reshape(features.shape[0], features.shape[1]).astype(np.uint8)


def _choose_foreground_cluster(cluster_map):
    """Choose the less border-dominant cluster as foreground."""
    border_mask = np.zeros_like(cluster_map, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    ratios = {}
    areas = {}
    for label in (0, 1):
        mask = cluster_map == label
        areas[label] = int(mask.sum())
        border_hits = int((mask & border_mask).sum())
        ratios[label] = border_hits / max(areas[label], 1)

    preferred = min((0, 1), key=lambda label: (ratios[label], areas[label]))
    return preferred


def foreground_mask_from_dense_features(feature_grid):
    """Infer a coarse foreground/background mask from dense feature clusters."""
    if feature_grid.ndim != 3:
        raise ValueError("feature_grid must have shape (H, W, C)")
    cluster_map = _two_means(feature_grid)
    foreground_label = _choose_foreground_cluster(cluster_map)
    return (cluster_map == foreground_label).astype(np.uint8)


def summarize_foreground_mask(mask):
    """Serialize a compact bootstrap summary for artifacts and review."""
    height, width = mask.shape[:2]
    foreground_ratio = float(mask.mean()) if height and width else 0.0
    ys, xs = np.where(mask > 0)
    bbox_xyxy = None
    if len(xs) > 0 and len(ys) > 0:
        bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "mask_shape": [int(height), int(width)],
        "foreground_ratio": round(foreground_ratio, 4),
        "foreground_bbox_xyxy": bbox_xyxy,
        "mask_grid": mask.astype(int).tolist(),
    }


def foreground_prior_for_point(summary, x, y):
    """Return a soft foreground prior for a point in image space."""
    if not summary:
        return 0.0
    mask_grid = summary.get("mask_grid")
    mask_shape = summary.get("mask_shape")
    if not mask_grid or not mask_shape or len(mask_shape) != 2:
        return 0.0
    height, width = int(mask_shape[0]), int(mask_shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    gx = min(width - 1, max(0, int((float(x) / max(float(summary.get("image_width", width)), 1.0)) * width)))
    gy = min(height - 1, max(0, int((float(y) / max(float(summary.get("image_height", height)), 1.0)) * height)))
    return float(mask_grid[gy][gx])


def _is_transformers_available():
    return importlib.util.find_spec("transformers") is not None


@dataclass
class Dinov3BootstrapResult:
    enabled: bool
    status: str
    backend: str
    model_name: str | None
    frame_idx: int | None
    summary: dict | None

    def to_payload(self):
        payload = {
            "enabled": bool(self.enabled),
            "status": self.status,
            "backend": self.backend,
            "model_name": self.model_name,
            "frame_idx": self.frame_idx,
        }
        if self.summary is not None:
            payload.update(self.summary)
        return payload


class Dinov3Bootstrapper:
    """Optional coarse foreground/background bootstrap based on DINOv3 features.

    This is intentionally a bootstrap pre-pass only. It runs on one chosen frame
    and emits a compact low-resolution mask summary for later geometry/perception
    stages to inspect.
    """

    def __init__(self, model_name=DEFAULT_DINOV3_MODEL, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None

    def is_available(self):
        return _is_transformers_available()

    def load(self):
        if not self.is_available():
            raise RuntimeError("transformers is not available; DINOv3 bootstrap cannot be loaded")
        if not self._gpu_ready():
            raise RuntimeError("DINOv3 bootstrap requires a CUDA-capable device")
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

    @staticmethod
    def _status_for_exception(exc):
        text = str(exc).lower()
        if "gated repo" in text or "restricted" in text or "401" in text:
            return "gated_repo"
        if "out of memory" in text or "cuda out of memory" in text:
            return "oom"
        return "load_failed"

    def _gpu_ready(self):
        if not str(self.device).startswith("cuda"):
            return False
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            return False
        import torch

        return bool(torch.cuda.is_available())

    def dense_features(self, frame_bgr):
        if self.model is None or self.processor is None:
            raise RuntimeError("Dinov3Bootstrapper.load() must be called before dense_features()")

        import torch

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state[0].detach().cpu().numpy()
        num_special_tokens = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        if tokens.shape[0] <= num_special_tokens:
            raise RuntimeError("DINOv3 output did not produce dense patch tokens")
        patch_tokens = tokens[num_special_tokens:]
        grid_size = int(math.sqrt(patch_tokens.shape[0]))
        if grid_size * grid_size != patch_tokens.shape[0]:
            raise RuntimeError("DINOv3 patch token count is not a perfect square")
        return patch_tokens.reshape(grid_size, grid_size, patch_tokens.shape[-1])

    def run_on_frame(self, frame_bgr, frame_idx=0):
        if not self.is_available():
            return Dinov3BootstrapResult(
                enabled=False,
                status="transformers_unavailable",
                backend="dinov3_transformers_v1",
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=None,
            )
        if not self._gpu_ready():
            return Dinov3BootstrapResult(
                enabled=False,
                status="cuda_unavailable",
                backend="dinov3_transformers_v1",
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=None,
            )
        try:
            self.load()
            features = self.dense_features(frame_bgr)
            mask = foreground_mask_from_dense_features(features)
            summary = summarize_foreground_mask(mask)
            summary["image_height"] = int(frame_bgr.shape[0])
            summary["image_width"] = int(frame_bgr.shape[1])
            return Dinov3BootstrapResult(
                enabled=True,
                status="ready",
                backend="dinov3_transformers_v1",
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=summary,
            )
        except Exception as exc:
            return Dinov3BootstrapResult(
                enabled=False,
                status=self._status_for_exception(exc),
                backend="dinov3_transformers_v1",
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=None,
            )

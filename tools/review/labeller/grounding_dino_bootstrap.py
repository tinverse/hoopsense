import importlib.util
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_GROUNDING_DINO_PROMPT = "basketball court. basketball player. basketball hoop. basketball referee."
DEFAULT_GROUNDING_DINO_BOX_THRESHOLD = 0.25
DEFAULT_GROUNDING_DINO_TEXT_THRESHOLD = 0.25
BOOTSTRAP_MASK_LONG_SIDE = 64
BOOTSTRAP_MAX_FOREGROUND_RATIO = 0.75
BOOTSTRAP_MAX_BBOX_WIDTH_RATIO = 0.95
BOOTSTRAP_MAX_BBOX_HEIGHT_RATIO = 0.95

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_HF_CACHE_DIR = REPO_ROOT / ".local_models" / "huggingface"


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


def scale_summary_bbox_to_image(summary, *, image_width, image_height):
    """Project the low-resolution mask bbox into full-image coordinates."""
    if not summary:
        return summary
    bbox = summary.get("foreground_bbox_xyxy")
    mask_shape = summary.get("mask_shape") or []
    if not bbox or len(mask_shape) != 2:
        return summary
    mask_h = max(1, int(mask_shape[0]))
    mask_w = max(1, int(mask_shape[1]))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    scale_x = float(image_width) / float(mask_w)
    scale_y = float(image_height) / float(mask_h)
    summary["foreground_bbox_xyxy"] = [
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round((x2 + 1.0) * scale_x)),
        int(round((y2 + 1.0) * scale_y)),
    ]
    return summary


def bootstrap_summary_is_informative(summary):
    """Reject broad foreground masks that are too coarse to help perception."""
    if not summary:
        return False
    foreground_ratio = float(summary.get("foreground_ratio") or 0.0)
    if foreground_ratio > BOOTSTRAP_MAX_FOREGROUND_RATIO:
        return False
    bbox = summary.get("foreground_bbox_xyxy")
    image_width = int(summary.get("image_width") or 0)
    image_height = int(summary.get("image_height") or 0)
    if not bbox or image_width <= 0 or image_height <= 0:
        return True
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bbox_width_ratio = max(0.0, x2 - x1) / max(float(image_width), 1.0)
    bbox_height_ratio = max(0.0, y2 - y1) / max(float(image_height), 1.0)
    if (
        bbox_width_ratio >= BOOTSTRAP_MAX_BBOX_WIDTH_RATIO
        and bbox_height_ratio >= BOOTSTRAP_MAX_BBOX_HEIGHT_RATIO
    ):
        return False
    return True


def bootstrap_mask_to_image(summary):
    """Upsample the stored low-resolution mask grid to full image resolution."""
    if not summary:
        return None
    mask_grid = summary.get("mask_grid")
    image_width = int(summary.get("image_width") or 0)
    image_height = int(summary.get("image_height") or 0)
    if not mask_grid or image_width <= 0 or image_height <= 0:
        return None
    mask = np.array(mask_grid, dtype=np.uint8)
    if mask.ndim != 2:
        return None
    return cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)


def component_boxes_from_mask(mask):
    """Extract connected component boxes from a binary mask without sklearn/scipy."""
    mask_u8 = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_u8.ndim != 2 or mask_u8.size == 0:
        return []
    height, width = mask_u8.shape
    visited = np.zeros_like(mask_u8, dtype=bool)
    boxes = []
    frame_area = float(max(height * width, 1))

    for y0 in range(height):
        for x0 in range(width):
            if mask_u8[y0, x0] == 0 or visited[y0, x0]:
                continue
            stack = [(x0, y0)]
            visited[y0, x0] = True
            min_x = max_x = x0
            min_y = max_y = y0
            area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or mask_u8[ny, nx] == 0:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
            boxes.append({
                "bbox_xyxy": [int(min_x), int(min_y), int(max_x + 1), int(max_y + 1)],
                "area_px": int(area),
                "area_ratio": round(float(area) / frame_area, 4),
            })
    return boxes


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


def _is_hf_available():
    return importlib.util.find_spec("huggingface_hub") is not None


def _split_prompt_labels(prompt):
    if isinstance(prompt, (list, tuple)):
        labels = [str(item).strip() for item in prompt if str(item).strip()]
        return labels or ["basketball player"]
    text = str(prompt or "").strip()
    if not text:
        return ["basketball player"]
    if "." in text:
        labels = [part.strip() for part in text.split(".") if part.strip()]
        if labels:
            return labels
    if "," in text:
        labels = [part.strip() for part in text.split(",") if part.strip()]
        if labels:
            return labels
    return [text]


def _normalize_text_label(text_label):
    return str(text_label or "").strip().strip(".").lower()


def foreground_mask_from_regions(regions, *, image_width, image_height, long_side=BOOTSTRAP_MASK_LONG_SIDE):
    """Rasterize Grounding DINO proposal boxes into a compact binary mask."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    full_mask = np.zeros((int(image_height), int(image_width)), dtype=np.uint8)
    for region in regions:
        bbox = region.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        ix1 = max(0, min(int(image_width) - 1, int(np.floor(x1))))
        iy1 = max(0, min(int(image_height) - 1, int(np.floor(y1))))
        ix2 = max(ix1 + 1, min(int(image_width), int(np.ceil(x2))))
        iy2 = max(iy1 + 1, min(int(image_height), int(np.ceil(y2))))
        full_mask[iy1:iy2, ix1:ix2] = 1
    if int(full_mask.sum()) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if image_width >= image_height:
        mask_w = int(long_side)
        mask_h = max(1, int(round((float(image_height) / float(image_width)) * float(long_side))))
    else:
        mask_h = int(long_side)
        mask_w = max(1, int(round((float(image_width) / float(image_height)) * float(long_side))))
    resized = cv2.resize(full_mask, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
    return (np.array(resized, dtype=np.uint8) > 0).astype(np.uint8)


@dataclass
class GroundingDinoBootstrapResult:
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


class GroundingDinoBootstrapper:
    """Promptable Grounding DINO bootstrap for coarse play-region priors."""

    def __init__(
        self,
        model_name=DEFAULT_GROUNDING_DINO_MODEL,
        text_prompt=DEFAULT_GROUNDING_DINO_PROMPT,
        box_threshold=DEFAULT_GROUNDING_DINO_BOX_THRESHOLD,
        text_threshold=DEFAULT_GROUNDING_DINO_TEXT_THRESHOLD,
        device="cpu",
    ):
        self.model_name = model_name
        self.text_prompt = text_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device = device
        self.processor = None
        self.model = None

    def is_available(self):
        return _is_transformers_available() and _is_hf_available()

    def _gpu_ready(self):
        if not str(self.device).startswith("cuda"):
            return False
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            return False
        import torch

        return bool(torch.cuda.is_available())

    @staticmethod
    def _status_for_exception(exc):
        text = str(exc).lower()
        if "gated repo" in text or "restricted" in text or "401" in text:
            return "gated_repo"
        if "couldn't connect" in text or "connection" in text or "offline" in text or "timed out" in text:
            return "weights_unavailable"
        if "not found" in text:
            return "weights_unavailable"
        if "out of memory" in text or "cuda out of memory" in text:
            return "oom"
        return "load_failed"

    def _resolve_model_source(self):
        from huggingface_hub import snapshot_download

        LOCAL_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            return snapshot_download(
                repo_id=self.model_name,
                cache_dir=str(LOCAL_HF_CACHE_DIR),
                local_files_only=True,
            )
        except Exception:
            return snapshot_download(
                repo_id=self.model_name,
                cache_dir=str(LOCAL_HF_CACHE_DIR),
            )

    def load(self):
        if self.model is not None and self.processor is not None:
            return self
        if not _is_transformers_available():
            raise RuntimeError("transformers is not available; Grounding DINO bootstrap cannot be loaded")
        if not _is_hf_available():
            raise RuntimeError("huggingface_hub is not available; Grounding DINO bootstrap cannot be loaded")
        if not self._gpu_ready():
            raise RuntimeError("Grounding DINO bootstrap requires a CUDA-capable device")

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_source = self._resolve_model_source()
        self.processor = AutoProcessor.from_pretrained(model_source, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_source, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        return self

    def predict_regions(self, frame_bgr):
        if self.model is None or self.processor is None:
            raise RuntimeError("GroundingDinoBootstrapper.load() must be called before predict_regions()")

        import torch

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = frame_bgr.shape[:2]
        candidate_labels = _split_prompt_labels(self.text_prompt)
        inputs = self.processor(images=Image.fromarray(frame_rgb), text=[candidate_labels], return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image_height, image_width)],
        )
        if not results:
            return []
        regions = []
        result = results[0]
        for box, score, text_label in zip(result.get("boxes", []), result.get("scores", []), result.get("text_labels", [])):
            box_values = box.detach().float().cpu().tolist() if hasattr(box, "detach") else [float(v) for v in box]
            score_value = float(score.detach().float().cpu().item()) if hasattr(score, "detach") else float(score)
            regions.append(
                {
                    "bbox_xyxy": [float(v) for v in box_values],
                    "confidence": round(score_value, 4),
                    "text_label": _normalize_text_label(text_label),
                }
            )
        return regions

    def run_on_frame(self, frame_bgr, frame_idx=0):
        backend_name = "grounding_dino_transformers_v1"
        if not self.is_available():
            return GroundingDinoBootstrapResult(
                enabled=False,
                status="transformers_unavailable",
                backend=backend_name,
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=None,
            )
        if not self._gpu_ready():
            return GroundingDinoBootstrapResult(
                enabled=False,
                status="cuda_unavailable",
                backend=backend_name,
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=None,
            )
        try:
            self.load()
            regions = self.predict_regions(frame_bgr)
            if not regions:
                return GroundingDinoBootstrapResult(
                    enabled=False,
                    status="no_detections",
                    backend=backend_name,
                    model_name=self.model_name,
                    frame_idx=frame_idx,
                    summary={
                        "image_height": int(frame_bgr.shape[0]),
                        "image_width": int(frame_bgr.shape[1]),
                        "text_prompt": self.text_prompt,
                        "box_threshold": self.box_threshold,
                        "text_threshold": self.text_threshold,
                        "proposal_regions": [],
                    },
                )
            mask = foreground_mask_from_regions(
                regions,
                image_width=int(frame_bgr.shape[1]),
                image_height=int(frame_bgr.shape[0]),
            )
            summary = summarize_foreground_mask(mask)
            summary["image_height"] = int(frame_bgr.shape[0])
            summary["image_width"] = int(frame_bgr.shape[1])
            summary["text_prompt"] = self.text_prompt
            summary["prompt_labels"] = _split_prompt_labels(self.text_prompt)
            summary["box_threshold"] = self.box_threshold
            summary["text_threshold"] = self.text_threshold
            summary["proposal_regions"] = regions
            summary = scale_summary_bbox_to_image(
                summary,
                image_width=int(frame_bgr.shape[1]),
                image_height=int(frame_bgr.shape[0]),
            )
            if not bootstrap_summary_is_informative(summary):
                return GroundingDinoBootstrapResult(
                    enabled=False,
                    status="foreground_too_broad",
                    backend=backend_name,
                    model_name=self.model_name,
                    frame_idx=frame_idx,
                    summary=summary,
                )
            return GroundingDinoBootstrapResult(
                enabled=True,
                status="ready",
                backend=backend_name,
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=summary,
            )
        except Exception as exc:
            return GroundingDinoBootstrapResult(
                enabled=False,
                status=self._status_for_exception(exc),
                backend=backend_name,
                model_name=self.model_name,
                frame_idx=frame_idx,
                summary=None,
            )

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_SAM3_REPO_MODEL = "facebook/sam3"
DEFAULT_SAM3_CHECKPOINT = "sam3.pt"
DEFAULT_SAM3_MULTIPLEX_REPO_MODEL = "facebook/sam3.1"
DEFAULT_SAM3_MULTIPLEX_CHECKPOINT = "sam3.1_multiplex.pt"
DEFAULT_SAM3_TEXT_PROMPT = "basketball player"
DEFAULT_SAM3_BALL_PROMPT = "basketball"
SAM_MIN_COMPONENT_AREA_RATIO = 0.0025
SAM_MAX_COMPONENT_AREA_RATIO = 0.22
SAM_MIN_MASK_AREA_PX = 600
SAM_PROPOSAL_IOU_MAX = 0.30
SAM_BALL_MIN_MASK_AREA_PX = 8
SAM_BALL_MAX_MASK_AREA_RATIO = 0.01
SAM_BALL_MAX_ASPECT_ERROR = 0.8

# The upstream SAM3 package pairs `build_sam3_image_model(...)` with the base
# image checkpoint (`facebook/sam3`, `sam3.pt`). The multiplex SAM 3.1
# checkpoint is intended for the multiplex/video predictor path and partially
# loads into the image builder, which emits missing-key warnings.


def _is_hf_available():
    return importlib.util.find_spec("huggingface_hub") is not None


def _is_sam3_available():
    return importlib.util.find_spec("sam3") is not None


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _connected_component_boxes(mask, *, image_width, image_height):
    mask_u8 = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_u8.ndim != 2 or mask_u8.size == 0:
        return []
    if mask_u8.shape[1] != image_width or mask_u8.shape[0] != image_height:
        mask_u8 = cv2.resize(mask_u8, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    frame_area = float(image_width * image_height)
    boxes = []
    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / max(frame_area, 1.0)
        if area_ratio < SAM_MIN_COMPONENT_AREA_RATIO or area_ratio > SAM_MAX_COMPONENT_AREA_RATIO:
            continue
        boxes.append(
            {
                "bbox_xyxy": [x, y, x + w, y + h],
                "area_px": area,
                "area_ratio": round(area_ratio, 4),
            }
        )
    return boxes


def _normalize_binary_mask(mask):
    if mask is None:
        return None
    mask_np = np.array(mask)
    if mask_np.size == 0:
        return None
    while mask_np.ndim > 2 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    while mask_np.ndim > 2 and mask_np.shape[-1] == 1:
        mask_np = mask_np[..., 0]
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return None
    return np.array(mask_np > 0, dtype=np.uint8)


def _mask_bbox(mask):
    mask_u8 = _normalize_binary_mask(mask)
    if mask_u8 is None:
        return None
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], int(mask_u8.sum())


def _bootstrap_mask_to_image(bootstrap_context):
    if not bootstrap_context or not bootstrap_context.get("enabled"):
        return None
    mask_grid = bootstrap_context.get("mask_grid")
    image_width = int(bootstrap_context.get("image_width") or 0)
    image_height = int(bootstrap_context.get("image_height") or 0)
    if not mask_grid or image_width <= 0 or image_height <= 0:
        return None
    mask = np.array(mask_grid, dtype=np.uint8)
    if mask.ndim != 2:
        return None
    return cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)


def build_sam_recovery_rois(frame, *, ambiguous_iou_threshold=SAM_PROPOSAL_IOU_MAX):
    detections = frame.get("detections") or []
    bootstrap_mask = _bootstrap_mask_to_image(frame.get("bootstrap_context"))
    grounding_context = frame.get("grounding_context") or {}
    rois = []
    person_boxes = [d.get("bbox_xyxy") for d in detections if d.get("bbox_xyxy")]

    if bootstrap_mask is not None:
        image_height, image_width = bootstrap_mask.shape[:2]
        for component in _connected_component_boxes(
            bootstrap_mask,
            image_width=image_width,
            image_height=image_height,
        ):
            bbox = component["bbox_xyxy"]
            max_iou = max((_bbox_iou(bbox, existing) for existing in person_boxes), default=0.0)
            if max_iou < ambiguous_iou_threshold:
                rois.append(
                    {
                        "kind": "unexplained_dino_blob",
                        "bbox_xyxy": bbox,
                        "source_iou": round(float(max_iou), 4),
                        "area_ratio": component["area_ratio"],
                    }
                )

    for proposal_region in grounding_context.get("proposal_regions") or []:
        bbox = proposal_region.get("bbox_xyxy")
        if not bbox:
            continue
        max_iou = max((_bbox_iou(bbox, existing) for existing in person_boxes), default=0.0)
        if max_iou < ambiguous_iou_threshold:
            rois.append(
                {
                    "kind": "grounding_anchor_region",
                    "bbox_xyxy": [float(v) for v in bbox],
                    "source_iou": round(float(max_iou), 4),
                    "source_trigger_reason": grounding_context.get("trigger_reason"),
                }
            )

    for detection in detections:
        bbox = detection.get("bbox_xyxy")
        if not bbox:
            continue
        reasons = detection.get("active_player_reasons") or {}
        merge_risk = float(reasons.get("merge_risk") or 0.0)
        if merge_risk >= 0.45 or bool(detection.get("on_court_candidate")) is False:
            rois.append(
                {
                    "kind": "ambiguous_yolo_detection",
                    "bbox_xyxy": [float(v) for v in bbox],
                    "track_id": detection.get("track_id"),
                    "merge_risk": round(merge_risk, 4),
                }
            )
    return rois


@dataclass
class SamRefineResult:
    enabled: bool
    status: str
    backend: str
    model_name: str | None
    proposals: list


class Sam3RoiRefiner:
    def __init__(
        self,
        model_name=DEFAULT_SAM3_REPO_MODEL,
        checkpoint_filename=DEFAULT_SAM3_CHECKPOINT,
        text_prompt=DEFAULT_SAM3_TEXT_PROMPT,
        device="cpu",
    ):
        self.model_name = model_name
        self.checkpoint_filename = checkpoint_filename
        self.text_prompt = text_prompt
        self.device = device
        self.model = None
        self.processor = None

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
        if "not found" in text and ".pt" in text:
            return "weights_unavailable"
        if "out of memory" in text or "cuda out of memory" in text:
            return "oom"
        return "load_failed"

    def load(self):
        if not _is_sam3_available():
            raise RuntimeError("sam3 repo package is not installed")
        if not _is_hf_available():
            raise RuntimeError("huggingface_hub is not available")
        if not self._gpu_ready():
            raise RuntimeError("SAM 3 refiner requires a CUDA-capable device")

        from huggingface_hub import hf_hub_download
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint_path = hf_hub_download(
            repo_id=self.model_name,
            filename=self.checkpoint_filename,
        )
        self.model = build_sam3_image_model(
            checkpoint_path=str(Path(checkpoint_path)),
            load_from_HF=False,
            device="cuda" if str(self.device).startswith("cuda") else self.device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(self.model, device="cuda" if str(self.device).startswith("cuda") else self.device)
        return self

    def refine(self, frame_bgr, rois):
        if not rois:
            return SamRefineResult(
                enabled=False,
                status="no_rois",
                backend="sam3_repo_v1",
                model_name=self.model_name,
                proposals=[],
            )
        if not _is_sam3_available():
            return SamRefineResult(False, "sam3_unavailable", "sam3_repo_v1", self.model_name, [])
        if not _is_hf_available():
            return SamRefineResult(False, "huggingface_hub_unavailable", "sam3_repo_v1", self.model_name, [])
        if not self._gpu_ready():
            return SamRefineResult(False, "cuda_unavailable", "sam3_repo_v1", self.model_name, [])
        try:
            self.load()
        except Exception as exc:
            return SamRefineResult(False, self._status_for_exception(exc), "sam3_repo_v1", self.model_name, [])

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = frame_bgr.shape[:2]
        proposals = []
        torch = None
        autocast_context = None
        if str(self.device).startswith("cuda"):
            import torch as _torch

            torch = _torch
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        for roi in rois:
            x1, y1, x2, y2 = [float(v) for v in roi["bbox_xyxy"]]
            ix1 = max(0, min(image_width - 1, int(np.floor(x1))))
            iy1 = max(0, min(image_height - 1, int(np.floor(y1))))
            ix2 = max(ix1 + 1, min(image_width, int(np.ceil(x2))))
            iy2 = max(iy1 + 1, min(image_height, int(np.ceil(y2))))
            crop_rgb = frame_rgb[iy1:iy2, ix1:ix2]
            if crop_rgb.size == 0:
                continue
            if autocast_context is not None:
                with autocast_context:
                    state = self.processor.set_image(Image.fromarray(crop_rgb))
                    output = self.processor.set_text_prompt(state=state, prompt=self.text_prompt)
            else:
                state = self.processor.set_image(Image.fromarray(crop_rgb))
                output = self.processor.set_text_prompt(state=state, prompt=self.text_prompt)
            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")
            if masks is None or boxes is None or scores is None or len(scores) == 0:
                continue
            score_values = scores.detach().float()
            best_idx = int(score_values.argmax().item())
            mask_payload = _mask_bbox(masks[best_idx].detach().float().cpu().numpy())
            if mask_payload is None:
                continue
            (mx1_local, my1_local, mx2_local, my2_local), area_px = mask_payload
            mx1, mx2 = mx1_local + ix1, mx2_local + ix1
            my1, my2 = my1_local + iy1, my2_local + iy1
            if area_px < SAM_MIN_MASK_AREA_PX:
                continue
            proposals.append(
                {
                    "kind": roi["kind"],
                    "bbox_xyxy": [mx1, my1, mx2, my2],
                    "bbox_xywh": [
                        float((mx1 + mx2) * 0.5),
                        float((my1 + my2) * 0.5),
                        float(max(0, mx2 - mx1)),
                        float(max(0, my2 - my1)),
                    ],
                    "mask_area_px": area_px,
                    "mask_area_ratio": round(area_px / max(float(image_width * image_height), 1.0), 4),
                    "sam_score": round(float(score_values[best_idx].item()), 4),
                    "source_roi_bbox_xyxy": [float(v) for v in roi["bbox_xyxy"]],
                    "source_kind": roi["kind"],
                    "source_track_id": roi.get("track_id"),
                    "source_merge_risk": roi.get("merge_risk"),
                    "source_iou": roi.get("source_iou"),
                    "source_trigger_reason": roi.get("source_trigger_reason"),
                }
            )
            if torch is not None:
                torch.cuda.empty_cache()
        return SamRefineResult(True, "ready", "sam3_repo_v1", self.model_name, proposals)


@dataclass
class SamBallDetection:
    bbox_xyxy: list
    bbox_xywh: list
    center_xy: list
    mask_area_px: int
    mask_area_ratio: float
    score: float
    prompt: str


class Sam3BallDetector:
    def __init__(
        self,
        model_name=DEFAULT_SAM3_REPO_MODEL,
        checkpoint_filename=DEFAULT_SAM3_CHECKPOINT,
        text_prompt=DEFAULT_SAM3_BALL_PROMPT,
        device="cpu",
    ):
        self.model_name = model_name
        self.checkpoint_filename = checkpoint_filename
        self.text_prompt = text_prompt
        self.device = device
        self.model = None
        self.processor = None

    def _gpu_ready(self):
        if not str(self.device).startswith("cuda"):
            return False
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            return False
        import torch

        return bool(torch.cuda.is_available())

    def load(self):
        if not _is_sam3_available():
            raise RuntimeError("sam3 repo package is not installed")
        if not _is_hf_available():
            raise RuntimeError("huggingface_hub is not available")
        if not self._gpu_ready():
            raise RuntimeError("SAM 3 ball detector requires a CUDA-capable device")

        from huggingface_hub import hf_hub_download
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint_path = hf_hub_download(
            repo_id=self.model_name,
            filename=self.checkpoint_filename,
        )
        runtime_device = "cuda" if str(self.device).startswith("cuda") else self.device
        self.model = build_sam3_image_model(
            checkpoint_path=str(Path(checkpoint_path)),
            load_from_HF=False,
            device=runtime_device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(self.model, device=runtime_device)
        return self

    def _candidate_from_mask(self, mask, score, image_width, image_height):
        payload = _mask_bbox(mask)
        if payload is None:
            return None
        (x1, y1, x2, y2), area_px = payload
        if area_px < SAM_BALL_MIN_MASK_AREA_PX:
            return None
        area_ratio = area_px / max(float(image_width * image_height), 1.0)
        if area_ratio > SAM_BALL_MAX_MASK_AREA_RATIO:
            return None
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_error = abs((width / float(height)) - 1.0)
        if aspect_error > SAM_BALL_MAX_ASPECT_ERROR:
            return None
        center_x = float((x1 + x2) * 0.5)
        center_y = float((y1 + y2) * 0.5)
        size_penalty = min(area_ratio / SAM_BALL_MAX_MASK_AREA_RATIO, 1.0)
        shape_score = max(0.0, 1.0 - aspect_error / SAM_BALL_MAX_ASPECT_ERROR)
        ranking_score = 0.75 * float(score) + 0.20 * shape_score + 0.05 * (1.0 - size_penalty)
        return {
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_xywh": [center_x, center_y, float(width), float(height)],
            "center_xy": [center_x, center_y],
            "mask_area_px": int(area_px),
            "mask_area_ratio": round(float(area_ratio), 6),
            "score": round(float(score), 4),
            "_ranking_score": float(ranking_score),
        }

    def detect(self, frame_bgr):
        if self.model is None or self.processor is None:
            self.load()
        image_height, image_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        torch = None
        autocast_context = None
        if str(self.device).startswith("cuda"):
            import torch as _torch

            torch = _torch
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if autocast_context is not None:
            with autocast_context:
                state = self.processor.set_image(Image.fromarray(frame_rgb))
                output = self.processor.set_text_prompt(state=state, prompt=self.text_prompt)
        else:
            state = self.processor.set_image(Image.fromarray(frame_rgb))
            output = self.processor.set_text_prompt(state=state, prompt=self.text_prompt)
        masks = output.get("masks")
        scores = output.get("scores")
        if masks is None or scores is None or len(scores) == 0:
            return []
        score_values = scores.detach().float()
        candidates = []
        for idx in range(len(score_values)):
            candidate = self._candidate_from_mask(
                masks[idx].detach().float().cpu().numpy(),
                float(score_values[idx].item()),
                image_width,
                image_height,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: item["_ranking_score"], reverse=True)
        if torch is not None:
            torch.cuda.empty_cache()
        return [
            SamBallDetection(
                bbox_xyxy=item["bbox_xyxy"],
                bbox_xywh=item["bbox_xywh"],
                center_xy=item["center_xy"],
                mask_area_px=item["mask_area_px"],
                mask_area_ratio=item["mask_area_ratio"],
                score=item["score"],
                prompt=self.text_prompt,
            )
            for item in candidates
        ]

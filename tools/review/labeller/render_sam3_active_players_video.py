import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from tools.review.labeller.sam_refiner import (
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SAM3_REPO_MODEL,
    DEFAULT_SAM3_TEXT_PROMPT,
    _is_hf_available,
    _is_sam3_available,
    _normalize_binary_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class Sam3TextSegmenter:
    def __init__(
        self,
        *,
        model_name=DEFAULT_SAM3_REPO_MODEL,
        checkpoint_filename=DEFAULT_SAM3_CHECKPOINT,
        text_prompt=DEFAULT_SAM3_TEXT_PROMPT,
        device="cuda:0",
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
        import torch

        return bool(torch.cuda.is_available())

    def load(self):
        if not _is_sam3_available():
            raise RuntimeError("sam3 repo package is not installed")
        if not _is_hf_available():
            raise RuntimeError("huggingface_hub is not available")
        if not self._gpu_ready():
            raise RuntimeError("SAM 3 player segmentation requires a CUDA-capable device")

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

    def segment_roi(self, frame_bgr, roi_bbox):
        if self.model is None or self.processor is None:
            self.load()
        x1, y1, x2, y2 = [int(round(v)) for v in roi_bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], max(x1 + 1, x2))
        y2 = min(frame_bgr.shape[0], max(y1 + 1, y2))
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return None

        frame_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        torch = None
        autocast_context = None
        if str(self.device).startswith("cuda"):
            import torch as _torch

            torch = _torch
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        from PIL import Image

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
            return None

        score_values = scores.detach().float()
        best_idx = int(score_values.argmax().item())
        mask = _normalize_binary_mask(masks[best_idx].detach().float().cpu().numpy())
        if mask is None or mask.sum() <= 0:
            return None
        if mask.shape[:2] != crop_bgr.shape[:2]:
            mask = cv2.resize(mask, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.array(mask > 0, dtype=np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        bbox_xyxy = [int(xs.min()) + x1, int(ys.min()) + y1, int(xs.max()) + x1, int(ys.max()) + y1]
        full_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask.astype(np.uint8)
        if torch is not None:
            torch.cuda.empty_cache()
        return {
            "bbox_xyxy": bbox_xyxy,
            "mask": full_mask,
            "score": float(score_values[best_idx].item()),
        }


def _load_perception_frames(perception_json_path):
    with open(perception_json_path, "r") as f:
        artifact = json.load(f)
    frame_map = {}
    for frame in artifact.get("frames", []):
        frame_map[int(frame["frame_idx"])] = frame
    return artifact, frame_map


def _active_rois(perception_frame):
    detections = perception_frame.get("detections") or []
    return [
        detection for detection in detections
        if detection.get("active_player_candidate") and detection.get("bbox_xyxy")
    ]


def _mask_color_for_index(index):
    palette = [
        (57, 214, 98),
        (55, 155, 255),
        (235, 185, 52),
        (217, 83, 79),
        (169, 104, 255),
        (38, 191, 191),
    ]
    return palette[index % len(palette)]


def _overlay_mask(frame, mask, color, alpha=0.28):
    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return frame
    frame_f = frame.astype(np.float32)
    frame_f[mask_bool] = frame_f[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return np.clip(frame_f, 0, 255).astype(np.uint8)


def _draw_segmented_player(frame, detection, segmentation, color):
    frame = _overlay_mask(frame, segmentation["mask"], color)
    x1, y1, x2, y2 = [int(v) for v in segmentation["bbox_xyxy"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"SAM3 ACTIVE P{detection.get('track_id') if detection.get('track_id') is not None else '?'} {int(round(segmentation['score'] * 100))}%"
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (min(frame.shape[1] - 1, x1 + 210), y1), (24, 24, 24), -1)
    cv2.putText(frame, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def render_sam3_active_players_video(
    clip_path,
    perception_json_path,
    output_path,
    *,
    model_name=DEFAULT_SAM3_REPO_MODEL,
    prompt=DEFAULT_SAM3_TEXT_PROMPT,
    device="cuda:0",
    max_players=10,
):
    clip_path = Path(clip_path)
    if not clip_path.is_absolute():
        clip_path = (REPO_ROOT / clip_path).resolve()
    perception_json_path = Path(perception_json_path)
    if not perception_json_path.is_absolute():
        perception_json_path = (REPO_ROOT / perception_json_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, frame_map = _load_perception_frames(perception_json_path)
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    segmenter = Sam3TextSegmenter(model_name=model_name, text_prompt=prompt, device=device)
    segmenter.load()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        perception_frame = frame_map.get(frame_idx, {})
        active_players = _active_rois(perception_frame)[:max_players]
        overlay = frame.copy()
        for idx, detection in enumerate(active_players):
            segmentation = segmenter.segment_roi(overlay, detection["bbox_xyxy"])
            if segmentation is None:
                continue
            overlay = _draw_segmented_player(
                overlay,
                detection,
                segmentation,
                _mask_color_for_index(idx),
            )
        cv2.rectangle(overlay, (12, 12), (420, 48), (24, 24, 24), -1)
        status = f"SAM3 active-player masks frame {frame_idx + 1}/{frame_count} active_rois {len(active_players)}"
        cv2.putText(overlay, status, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        writer.write(overlay)
        frame_idx += 1

    writer.release()
    cap.release()
    return output_path


def transcode_websafe(input_path, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render a SAM3 active-player segmentation overlay video.")
    parser.add_argument("clip_path", help="Absolute or repo-relative path to the clip")
    parser.add_argument("--perception-json", required=True, help="Layer 1 perception artifact with active-player candidates")
    parser.add_argument("--output", required=True, help="Rendered MP4 output path")
    parser.add_argument("--websafe-output", default=None, help="Optional H.264 web-safe MP4 output path")
    parser.add_argument("--model", default=DEFAULT_SAM3_REPO_MODEL, help="SAM3 repo id")
    parser.add_argument("--prompt", default=DEFAULT_SAM3_TEXT_PROMPT, help="SAM3 text prompt")
    parser.add_argument("--device", default="cuda:0", help="Runtime device")
    parser.add_argument("--max-players", type=int, default=10, help="Maximum active-player ROIs to segment per frame")
    args = parser.parse_args()

    raw_output = render_sam3_active_players_video(
        args.clip_path,
        args.perception_json,
        args.output,
        model_name=args.model,
        prompt=args.prompt,
        device=args.device,
        max_players=max(1, int(args.max_players)),
    )
    print(raw_output)
    if args.websafe_output:
        websafe_path = transcode_websafe(raw_output, args.websafe_output)
        print(websafe_path)


if __name__ == "__main__":
    main()

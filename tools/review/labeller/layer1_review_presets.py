#!/usr/bin/env python3
"""Run reproducible Layer 1 review artifact generation from checked-in presets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PRESET_FILE = REPO_ROOT / "specs" / "layer1_review_presets.yaml"
RUN_DIR = REPO_ROOT / "tmp_runs" / "layer1_review_runs"
GENERATOR_PATH = REPO_ROOT / "tools" / "review" / "labeller" / "generate_layer1_annotations.py"
NATIVE_RUNNER = REPO_ROOT / "scripts" / "run_orin_layer1_python.sh"
DOCKER_RUNNER = REPO_ROOT / "scripts" / "run_orin_layer1_docker.sh"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "review_artifacts" / "layer1"

GENERATOR_ARG_FLAGS = {
    "player_model": "--player-model",
    "player_tracker_backend": "--player-tracker-backend",
    "pose_model": "--model",
    "ball_model": "--ball-model",
    "player_conf": "--player-conf",
    "pose_conf": "--pose-conf",
    "bootstrap_foreground_backend": "--bootstrap-foreground-backend",
    "bootstrap_foreground_model": "--bootstrap-foreground-model",
    "bootstrap_foreground_prompt": "--bootstrap-foreground-prompt",
    "player_recovery_backend": "--player-recovery-backend",
    "player_recovery_model": "--player-recovery-model",
    "player_recovery_prompt": "--player-recovery-prompt",
    "ball_bootstrap_backend": "--ball-bootstrap-backend",
    "ball_bootstrap_model": "--ball-bootstrap-model",
    "ball_bootstrap_prompt": "--ball-bootstrap-prompt",
}

MODEL_CACHE_ENV_KEYS = (
    "HOOPSENSE_HF_CACHE_DIR",
    "HOOPSENSE_TORCH_CACHE_DIR",
    "HOOPSENSE_YOLO_CACHE_DIR",
    "HOOPSENSE_EASYOCR_CACHE_DIR",
    "HF_HOME",
    "TORCH_HOME",
    "YOLO_CONFIG_DIR",
)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def load_preset_file(path: Path = PRESET_FILE) -> dict[str, Any]:
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    if payload.get("kind") != "layer1_review_presets":
        raise ValueError(f"{path} is not a layer1_review_presets spec")
    presets = payload.get("presets") or {}
    if not isinstance(presets, dict) or not presets:
        raise ValueError(f"{path} does not define any presets")
    return payload


def get_preset(name: str, path: Path = PRESET_FILE) -> dict[str, Any]:
    payload = load_preset_file(path)
    presets = payload["presets"]
    if name not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"unknown Layer 1 review preset {name!r}; available: {available}")
    preset = dict(presets[name] or {})
    preset["name"] = name
    return preset


def expected_output_path(clip_path: Path, output_path: Path | None = None) -> Path:
    if output_path is not None:
        return output_path if output_path.is_absolute() else (REPO_ROOT / output_path).resolve()
    return DEFAULT_OUTPUT_DIR / f"{clip_path.stem}.perception.json"


def build_generator_args(preset: dict[str, Any], *, output_path: Path | None = None, extra_args: list[str] | None = None) -> list[str]:
    args: list[str] = []
    generator_args = preset.get("generator_args") or {}
    for key, flag in GENERATOR_ARG_FLAGS.items():
        if key not in generator_args:
            continue
        value = generator_args[key]
        if value is None:
            continue
        args.extend([flag, str(value)])
    if output_path is not None:
        args.extend(["--output", str(output_path)])
    args.extend(extra_args or [])
    return args


def build_generation_command(
    clip_path: Path,
    preset: dict[str, Any],
    *,
    runner: str | None = None,
    output_path: Path | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    selected_runner = runner or str(preset.get("runner") or "native")
    generator_args = build_generator_args(preset, output_path=output_path, extra_args=extra_args)
    if selected_runner == "native":
        return [str(NATIVE_RUNNER), str(GENERATOR_PATH), str(clip_path), *generator_args, "--device", "cuda:0"]
    if selected_runner == "docker":
        return [str(DOCKER_RUNNER), "python3", "/app/tools/review/labeller/generate_layer1_annotations.py", str(_repo_relative(clip_path)), *generator_args, "--device", "cuda:0"]
    raise ValueError(f"unsupported runner {selected_runner!r}; expected native or docker")


def _git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _model_cache_env() -> dict[str, str]:
    return {key: value for key in MODEL_CACHE_ENV_KEYS if (value := os.environ.get(key))}


def build_run_manifest(
    *,
    clip_path: Path,
    output_path: Path,
    preset: dict[str, Any],
    runner: str,
    command: list[str],
    run_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "kind": "layer1_review_generation_run_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "preset_name": preset["name"],
        "preset_description": preset.get("description"),
        "runner": runner,
        "dry_run": bool(dry_run),
        "clip_path": _repo_relative(clip_path),
        "output_path": _repo_relative(output_path),
        "preset_file": _repo_relative(PRESET_FILE),
        "generator_path": _repo_relative(GENERATOR_PATH),
        "git": {
            "head": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["branch", "--show-current"]),
            "dirty": bool(_git_value(["status", "--porcelain"])),
        },
        "model_cache_env": _model_cache_env(),
        "generator_args": preset.get("generator_args") or {},
        "command": command,
    }


def make_run_dir(clip_path: Path, preset_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_clip = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in clip_path.stem)
    safe_preset = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in preset_name)
    return RUN_DIR / f"{timestamp}_{safe_clip}_{safe_preset}"


def run_preset(args: argparse.Namespace) -> int:
    clip_path = Path(args.clip_path)
    if not clip_path.is_absolute():
        clip_path = (REPO_ROOT / clip_path).resolve()
    preset = get_preset(args.preset)
    runner = args.runner or str(preset.get("runner") or "native")
    output_path = expected_output_path(clip_path, Path(args.output).resolve() if args.output else None)
    command = build_generation_command(
        clip_path,
        preset,
        runner=runner,
        output_path=output_path if args.output else None,
        extra_args=args.extra_generator_args,
    )
    run_dir = Path(args.run_dir).resolve() if args.run_dir else make_run_dir(clip_path, preset["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        clip_path=clip_path,
        output_path=output_path,
        preset=preset,
        runner=runner,
        command=command,
        run_dir=run_dir,
        dry_run=args.dry_run,
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[INFO] Wrote run manifest: {_repo_relative(manifest_path)}")
    print("[INFO] Command:")
    print(" ".join(command))
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Layer 1 review artifact generation from a checked-in preset.")
    parser.add_argument("clip_path", help="Absolute or repo-relative clip path")
    parser.add_argument("--preset", default="grounded_sam3_review", help="Preset name from specs/layer1_review_presets.yaml")
    parser.add_argument("--runner", choices=["native", "docker"], default=None, help="Override the preset runner")
    parser.add_argument("--output", default=None, help="Optional output perception JSON path")
    parser.add_argument("--run-dir", default=None, help="Optional run manifest directory")
    parser.add_argument("--dry-run", action="store_true", help="Write the run manifest and command without executing generation")
    args, extra_generator_args = parser.parse_known_args()
    if extra_generator_args and extra_generator_args[0] == "--":
        extra_generator_args = extra_generator_args[1:]
    args.extra_generator_args = extra_generator_args
    return run_preset(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a repeatable Orin validation artifact for CUDA and core probe checks."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.infra.orin_validation import run_rigorous_probe, validate_environment


OUTPUT_DIR = REPO_ROOT / "artifacts" / "orin_validation"
JSON_PATH = OUTPUT_DIR / "latest.json"
MD_PATH = REPO_ROOT / "ORIN_VALIDATION_REPORT.md"
CHECK_DESCRIPTIONS = {
    "torch": "PyTorch imports successfully in the intended runtime environment.",
    "numpy": "NumPy C-extensions import cleanly inside the validation environment.",
    "cuda_available": "PyTorch sees a CUDA-capable device from the current runtime path.",
    "cuda_tensor": "A test tensor can be allocated onto the CUDA device, not just detected.",
    "action_brain": "The repo's ActionBrain model can load, move to the target device, and run a forward pass.",
    "pose_association": "The lightweight player-pose matching helper behaves correctly on a simple multi-player fixture.",
    "dsl_rule": "The behavior-engine rule path still recognizes a basic jump-shot style sequence.",
    "rust_bridge": "The Rust spatial processor binary can still be invoked from the validation environment.",
}
REQUIRED_CHECKS = {"torch", "cuda_available", "cuda_tensor", "action_brain"}


def build_report(env_result: dict, probe_result: dict) -> dict:
    checks = [
        ("torch", env_result["torch_ok"]),
        ("numpy", env_result["numpy_ok"]),
        ("cuda_available", env_result["cuda_available"]),
        ("cuda_tensor", env_result["cuda_tensor_ok"]),
        ("action_brain", probe_result["action_brain_ok"]),
        ("pose_association", probe_result["pose_association_ok"]),
        ("dsl_rule", probe_result["dsl_rule_ok"]),
        ("rust_bridge", probe_result["rust_bridge_ok"]),
    ]
    status = "pass" if all(passed for name, passed in checks if name in REQUIRED_CHECKS) else "fail"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "environment": env_result,
        "probe": probe_result,
        "checks": [
            {"name": name, "passed": passed, "required": name in REQUIRED_CHECKS}
            for name, passed in checks
        ],
    }


def write_markdown(report: dict) -> None:
    lines = [
        "# Orin Validation Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Overall Status: `{report['status'].upper()}`",
        "",
        "## Environment",
        f"- Python: `{report['environment']['python_executable']}`",
        f"- Platform: `{report['environment']['platform']}`",
        f"- Machine: `{report['environment']['machine']}`",
        f"- PyTorch: `{report['environment']['torch_version']}`",
        f"- PyTorch Module: `{report['environment']['torch_file']}`",
        f"- CUDA Available: `{report['environment']['cuda_available']}`",
        f"- CUDA Device: `{report['environment']['cuda_device_name']}`",
        f"- Probe Device Type: `{report['probe']['device_type']}`",
        f"- ActionBrain Device: `{report['probe']['action_brain_device']}`",
        f"- ActionBrain Latency (ms): `{report['probe']['action_brain_latency_ms']}`",
        "",
        "## Checks",
    ]
    for check in report["checks"]:
        description = CHECK_DESCRIPTIONS.get(check["name"], "")
        state = "SKIP" if check["passed"] is None else ("PASS" if check["passed"] else "FAIL")
        scope = "required" if check["required"] else "supplemental"
        line = f"- `{check['name']}`: `{state}` ({scope})"
        if description:
            line += f"  {description}"
        lines.append(line)
    if report["environment"]["errors"] or report["probe"].get("rust_bridge_stderr"):
        lines.extend(["", "## Notes"])
        for error in report["environment"]["errors"]:
            lines.append(f"- Environment error: `{error}`")
        if report["probe"].get("action_brain_error"):
            lines.append(f"- ActionBrain error: `{report['probe']['action_brain_error']}`")
        if report["probe"].get("rust_bridge_stderr"):
            lines.append(f"- Rust bridge stderr: `{report['probe']['rust_bridge_stderr']}`")
    MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env_result = validate_environment()
    probe_result = run_rigorous_probe()
    report = build_report(env_result, probe_result)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"[INFO] Wrote {JSON_PATH}")
    print(f"[INFO] Wrote {MD_PATH}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

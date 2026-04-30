"""Runtime resource policy helpers for HoopSense perception jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass


CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class ResourcePolicy:
    requested_device: str
    resolved_device: str
    cuda_available: bool
    cuda_device_name: str | None
    cpu_count: int
    cpu_thread_count: int
    cv2_thread_count: int | None
    env_thread_limits: dict[str, str | None]

    def to_payload(self) -> dict:
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "cpu_count": self.cpu_count,
            "cpu_thread_count": self.cpu_thread_count,
            "cv2_thread_count": self.cv2_thread_count,
            "env_thread_limits": dict(self.env_thread_limits),
        }


def default_cpu_thread_count(cpu_count: int | None = None) -> int:
    """Choose a conservative CPU thread count for mixed GPU/CPU workloads."""
    count = int(cpu_count or (os.cpu_count() or 1))
    return max(1, min(4, count))


def apply_cpu_thread_policy(thread_count: int | None = None) -> dict[str, str | None]:
    """Cap common native CPU thread pools unless the caller already set them."""
    chosen = default_cpu_thread_count(thread_count)
    for name in CPU_THREAD_ENV_VARS:
        os.environ.setdefault(name, str(chosen))
    try:
        import cv2

        cv2.setNumThreads(chosen)
    except Exception:
        pass
    return {name: os.environ.get(name) for name in CPU_THREAD_ENV_VARS}


def resolve_torch_device(requested_device: str | None = "auto"):
    """Resolve `auto`/CUDA/CPU into a torch.device plus a serializable policy."""
    import torch

    requested = str(requested_device or "auto")
    cuda_available = bool(torch.cuda.is_available())
    if requested in {"auto", "cuda", "cuda:0"}:
        resolved = "cuda:0" if cuda_available else "cpu"
    else:
        resolved = requested
    if resolved.startswith("cuda") and not cuda_available:
        resolved = "cpu"

    cuda_device_name = None
    if cuda_available:
        try:
            cuda_device_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_device_name = None

    env_limits = apply_cpu_thread_policy()
    cv2_thread_count = None
    try:
        import cv2

        cv2_thread_count = int(cv2.getNumThreads())
    except Exception:
        pass

    policy = ResourcePolicy(
        requested_device=requested,
        resolved_device=resolved,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        cpu_count=int(os.cpu_count() or 1),
        cpu_thread_count=default_cpu_thread_count(),
        cv2_thread_count=cv2_thread_count,
        env_thread_limits=env_limits,
    )
    return torch.device(resolved), policy

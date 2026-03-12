#!/usr/bin/env python3
"""Generate a geometry readiness report from keypoint sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipelines.geometry import build_geometry_readiness_report


def load_sequences(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("sequences", [])
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list of sequences or an object with 'sequences'")
    return [np.asarray(seq, dtype=float) for seq in payload]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a HoopSense geometry readiness report.")
    parser.add_argument("sequence_json", help="JSON file containing one or more (T,17,2) sequences")
    parser.add_argument(
        "--homography-json",
        help="JSON file containing {'h_matrix': [[...],[...],[...]]}. Defaults to identity.",
    )
    parser.add_argument("--output", help="Write the readiness report to this path")
    args = parser.parse_args(argv)

    sequences = load_sequences(Path(args.sequence_json))
    h_matrix = np.eye(3)
    if args.homography_json:
        h_payload = json.loads(Path(args.homography_json).read_text())
        h_matrix = np.asarray(h_payload["h_matrix"], dtype=float)

    report = build_geometry_readiness_report(sequences, h_matrix)
    output = report.to_json()
    if args.output:
        Path(args.output).write_text(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

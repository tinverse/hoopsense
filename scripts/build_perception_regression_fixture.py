#!/usr/bin/env python3
"""Normalize perception feedback JSONL into a categorized regression fixture."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "training" / "perception_feedback.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "perception_regression_cases.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def normalize_note(note: str | None) -> str:
    return (note or "").strip()


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def derive_categories(entry: dict) -> list[str]:
    issue_type = entry.get("issue_type", "")
    note = normalize_note(entry.get("note")).lower()
    categories = set()

    if issue_type == "live_play":
        categories.add("live_play_context")
    if issue_type == "dead_ball":
        categories.add("dead_ball_context")
    if issue_type == "uncertain_play_state":
        categories.add("scene_context")

    if issue_type == "false_positive":
        if _contains_any(note, [r"\bspectator\b", r"\bseated\b", r"\bbench\b"]):
            categories.add("spectator_false_positive")
        elif _contains_any(note, [r"\bbystander", r"\bbystanders\b", r"\bref\b", r"\breferee\b", r"\bsideline\b"]):
            categories.add("bystander_false_positive")
        else:
            categories.add("false_positive_generic")

    if issue_type == "false_negative":
        if _contains_any(note, [r"\bcluster", r"\bclustered\b", r"\bmerged\b", r"\bmissed \d+ players?\b"]):
            categories.add("cluster_miss")
        else:
            categories.add("false_negative_generic")

    if _contains_any(note, [r"\bmissed\b"]) and _contains_any(note, [r"\bcluster", r"\bclustered\b"]):
        categories.add("cluster_miss")

    if _contains_any(note, [r"\bbystander", r"\bbystanders\b"]):
        categories.add("bystander_false_positive")

    if _contains_any(note, [r"\bspectator\b", r"\bseated\b", r"\bbench\b"]):
        categories.add("spectator_false_positive")

    if _contains_any(note, [r"\bref\b", r"\breferee\b", r"\bsideline\b"]):
        categories.add("bystander_false_positive")

    if issue_type == "merge_error" or _contains_any(note, [r"\bmerged\b", r"\bmerge\b"]):
        categories.add("cluster_merge")

    if issue_type == "track_error" or _contains_any(
        note,
        [r"\bsame\b", r"\bbecomes\b", r"\bkeep changing\b", r"\bid switch\b", r"\bnumbers keep changing\b"],
    ):
        categories.add("id_switch")

    if _contains_any(note, [r"\bdark\b", r"\blight\b", r"\buniform\b"]):
        categories.add("uniform_confusion")

    if _contains_any(note, [r"\bdribbling\b", r"\bdribbler\b", r"\bbringing the ball forward\b", r"\blive play\b"]):
        categories.add("live_play_context")

    if _contains_any(
        note,
        [
            r"\bsubbed out\b",
            r"\bsubstitution\b",
            r"\bout of bounds\b",
            r"\bstopped the play\b",
            r"\bdead ball\b",
            r"\binbound\b",
            r"\bno actual play\b",
        ],
    ):
        categories.add("dead_ball_context")

    if issue_type == "general_note" and not categories:
        categories.add("scene_context")

    return sorted(categories)


def build_fixture(entries: list[dict], source_path: Path) -> dict:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for entry in entries:
        clip_id = entry.get("clip_id")
        frame_idx = int(entry.get("frame_idx", 0))
        grouped[(clip_id, frame_idx)].append(entry)

    cases = []
    for (clip_id, frame_idx), group in sorted(grouped.items()):
        t_ms_values = [int(entry.get("t_ms", 0)) for entry in group]
        track_ids = sorted(
            {
                str(entry.get("track_id"))
                for entry in group
                if entry.get("track_id") not in (None, "")
            }
        )
        issue_types = sorted({entry.get("issue_type") for entry in group if entry.get("issue_type")})
        notes = [normalize_note(entry.get("note")) for entry in group if normalize_note(entry.get("note"))]
        categories = sorted({cat for entry in group for cat in derive_categories(entry)})

        scene_context = None
        if "dead_ball_context" in categories:
            scene_context = "dead_ball"
        elif "live_play_context" in categories:
            scene_context = "live_play"

        cases.append(
            {
                "case_id": f"{clip_id}:{frame_idx}",
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "t_ms_min": min(t_ms_values) if t_ms_values else 0,
                "t_ms_max": max(t_ms_values) if t_ms_values else 0,
                "issue_types": issue_types,
                "categories": categories,
                "scene_context": scene_context,
                "track_ids": track_ids,
                "notes": notes,
                "source_count": len(group),
            }
        )

    try:
        source_label = str(source_path.relative_to(REPO_ROOT))
    except ValueError:
        source_label = str(source_path)

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a categorized perception regression fixture.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_jsonl(input_path)
    fixture = build_fixture(entries, input_path)
    output_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()

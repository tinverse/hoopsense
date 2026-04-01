from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MVP_EVENT_RULES_FILE = REPO_ROOT / "specs" / "mvp_event_rules.yaml"
REQUIRED_EVENTS = {
    "pass",
    "catch",
    "shot_attempt",
    "made_shot",
    "missed_shot",
    "rebound_off",
    "rebound_def",
    "turnover",
    "steal",
    "assist_candidate",
}
REQUIRED_EVENT_SECTIONS = {"prerequisites", "actors", "temporal", "counting"}


def load_mvp_event_rules(spec_path=DEFAULT_MVP_EVENT_RULES_FILE):
    """Load the checked-in MVP event rules and validate the contract shape."""
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    if spec.get("kind") != "mvp_event_rules":
        raise ValueError("MVP event rules spec must declare kind: mvp_event_rules")

    assumptions = spec.get("assumptions") or {}
    events = spec.get("events") or {}
    if not assumptions:
        raise ValueError("MVP event rules spec is missing assumptions")
    if not events:
        raise ValueError("MVP event rules spec is missing events")

    missing_events = sorted(REQUIRED_EVENTS - set(events))
    if missing_events:
        raise ValueError(
            f"MVP event rules spec is missing required events: {', '.join(missing_events)}"
        )

    for event_name in REQUIRED_EVENTS:
        event_spec = events.get(event_name) or {}
        missing_sections = sorted(REQUIRED_EVENT_SECTIONS - set(event_spec))
        if missing_sections:
            raise ValueError(
                f"MVP event rule '{event_name}' is missing sections: {', '.join(missing_sections)}"
            )
        actors = event_spec.get("actors") or {}
        if "primary" not in actors:
            raise ValueError(f"MVP event rule '{event_name}' must define a primary actor role")
        counting = event_spec.get("counting") or {}
        if "stat_deltas" not in counting:
            raise ValueError(f"MVP event rule '{event_name}' must define counting.stat_deltas")

    return spec


MVP_EVENT_RULES = load_mvp_event_rules()

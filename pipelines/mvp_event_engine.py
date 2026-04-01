from copy import deepcopy

from pipelines.mvp_rules import MVP_EVENT_RULES


class MvpEventRuleEngine:
    """Validate attributed MVP events and derive stat deltas from rule specs."""

    def __init__(self, spec=None):
        self.spec = spec or MVP_EVENT_RULES
        self.events = self.spec["events"]

    def rule_for(self, event_type):
        if event_type not in self.events:
            raise KeyError(f"Unknown MVP event type: {event_type}")
        return self.events[event_type]

    def validate_event(self, event):
        """Return unmet rule constraints for an already-attributed event payload."""
        rule = self.rule_for(event["event_type"])
        unmet = []
        for prerequisite in rule.get("prerequisites", []):
            if not self._satisfies_prerequisite(prerequisite, event):
                unmet.append(prerequisite)

        actors = rule.get("actors") or {}
        if actors.get("primary") and event.get("actor_id") is None:
            unmet.append("primary_actor_missing")
        secondary_role = actors.get("secondary")
        if secondary_role and not str(secondary_role).endswith("_optional") and event.get("secondary_actor_id") is None:
            unmet.append("secondary_actor_missing")

        temporal = rule.get("temporal") or {}
        must_follow = temporal.get("must_follow")
        if must_follow and event.get("preceding_event_type") != must_follow:
            unmet.append(f"must_follow:{must_follow}")
        expected_follow = temporal.get("expected_follow_up") or {}
        expected_event = expected_follow.get("event")
        if expected_event and event.get("next_event_type") not in {None, expected_event}:
            unmet.append(f"expected_follow_up:{expected_event}")
        allowed_follow = set(temporal.get("allowed_follow_up") or [])
        if allowed_follow and event.get("next_event_type") not in allowed_follow:
            unmet.append("allowed_follow_up")
        resolution_required = temporal.get("resolution_required") or {}
        allowed_resolution = set(resolution_required.get("allowed_events") or [])
        if allowed_resolution and event.get("resolution_event_type") not in allowed_resolution:
            unmet.append("resolution_required")
        required_terminal = temporal.get("required_terminal_event")
        if required_terminal and event.get("terminal_event_type") != required_terminal:
            unmet.append(f"required_terminal_event:{required_terminal}")
        return unmet

    def stat_deltas_for_event(self, event):
        unmet = self.validate_event(event)
        if unmet:
            raise ValueError(f"Event does not satisfy MVP rule constraints: {', '.join(unmet)}")
        rule = self.rule_for(event["event_type"])
        stat_deltas = deepcopy(rule.get("counting", {}).get("stat_deltas", {}))
        if not stat_deltas:
            return {}
        if all(isinstance(value, int) for value in stat_deltas.values()):
            return stat_deltas
        shot_value = event.get("shot_value")
        if shot_value not in stat_deltas:
            raise ValueError(
                f"Event '{event['event_type']}' requires shot_value in {sorted(stat_deltas)}"
            )
        return deepcopy(stat_deltas[shot_value])

    def _satisfies_prerequisite(self, prerequisite, event):
        evidence = event.get("evidence") or {}
        if prerequisite == "live_play":
            return bool(event.get("live_play"))
        if prerequisite == "actor_identity_known":
            return event.get("actor_id") is not None
        if prerequisite == "receiver_identity_known":
            return event.get("secondary_actor_id") is not None
        if prerequisite == "team_in_possession_known":
            return event.get("team_id") is not None
        if prerequisite == "ball_release_evidence":
            return bool(evidence.get("ball_release"))
        if prerequisite == "shot_value_known":
            return event.get("shot_value") in {"two_point", "three_point", "free_throw"}
        if prerequisite == "preceding_shot_attempt":
            return event.get("preceding_event_type") == "shot_attempt"
        if prerequisite == "make_result_evidence":
            return evidence.get("shot_result") == "made"
        if prerequisite == "miss_result_evidence":
            return evidence.get("shot_result") == "missed"
        if prerequisite == "preceding_missed_shot":
            return event.get("preceding_event_type") == "missed_shot"
        if prerequisite == "possession_change_evidence":
            return bool(evidence.get("possession_change"))
        if prerequisite == "loss_of_team_control_evidence":
            return bool(evidence.get("loss_of_team_control"))
        if prerequisite == "possession_gain_evidence":
            return bool(evidence.get("possession_gain"))
        if prerequisite == "preceding_turnover_optional":
            return True
        if prerequisite == "pass_then_made_shot_sequence":
            return bool(evidence.get("pass_then_make"))
        return True

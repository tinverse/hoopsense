from copy import deepcopy


class MvpStatAccumulator:
    """Apply deterministic stat deltas from attributed MVP events.

    The accumulator is intentionally append-only and event-driven. It does not
    try to render a final scorebook yet; it simply maintains running per-player
    totals and emits one `stat_update` row per counted attributed event.
    """

    def __init__(self):
        self.player_totals = {}

    def apply_attributed_event(self, event):
        if event.get("kind") != "attributed_event":
            return None
        player_id = event.get("actor_id")
        stat_deltas = event.get("stat_deltas") or {}
        if player_id is None or not stat_deltas:
            return None

        totals = self.player_totals.setdefault(int(player_id), {})
        for stat_name, delta in stat_deltas.items():
            totals[stat_name] = int(totals.get(stat_name, 0)) + int(delta)

        return {
            "kind": "stat_update",
            "player_id": int(player_id),
            "team_id": int(event["team_id"]) if event.get("team_id") is not None else None,
            "t_ms": int(event["t_ms"]) if event.get("t_ms") is not None else None,
            "source_event_type": event.get("event_type"),
            "source_actor_id": int(event["actor_id"]) if event.get("actor_id") is not None else None,
            "applied_deltas": deepcopy(stat_deltas),
            "running_totals": deepcopy(totals),
        }

    def snapshot_for_player(self, player_id, *, team_id=None, t_ms=None):
        totals = self.player_totals.get(int(player_id))
        if totals is None:
            return None
        return {
            "kind": "stat_snapshot",
            "player_id": int(player_id),
            "team_id": int(team_id) if team_id is not None else None,
            "t_ms": int(t_ms) if t_ms is not None else None,
            "totals": deepcopy(totals),
        }

    def terminal_game_snapshot(self, *, game_id=None, t_ms=None):
        players = []
        for player_id in sorted(self.player_totals):
            players.append(
                {
                    "player_id": int(player_id),
                    "totals": deepcopy(self.player_totals[player_id]),
                }
            )
        return {
            "kind": "game_stat_sheet",
            "game_id": game_id,
            "t_ms": int(t_ms) if t_ms is not None else None,
            "players": players,
        }

import json


def summarize_game(dna_path):
    """
    Computes game-level statistics from the Game DNA.
    """
    events = []
    with open(dna_path, 'r') as f:
        for line in f:
            events.append(json.loads(line))

    # Placeholder for aggregation logic
    stats = {
        "total_events": len(events),
        "home_score": 0,
        "away_score": 0
    }

    # Derive scores from validated 'MadeBasket' events in Layer 4
    for ev in events:
        if ev.get("kind") == "MadeBasket" and ev.get("is_official"):
            points = ev.get("points", 0)
            if ev.get("team_id") == 1:
                stats["home_score"] += points
            else:
                stats["away_score"] += points

    return stats


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/validated_game_dna.jsonl"
    report = summarize_game(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

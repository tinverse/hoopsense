import json
import sys

def summarize_dna(jsonl_path):
    print(f"[INFO] Summarizing Game DNA from {jsonl_path}...")
    events = []
    current_score = [0, 0]
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            kind = row.get("kind")
            t_sec = row.get("t_ms", 0) / 1000.0
            
            # Identify Key Narrative Events
            if kind == "player":
                action = row.get("action")
                if action == "jump_shot":
                    player = row.get("actor_jersey_number", "unknown")
                    events.append(f"[{t_sec:.1f}s] Player #{player} attempted a jump shot.")
            
            elif kind == "referee":
                signal = row.get("signal")
                if signal == "ref_3pt_success":
                    events.append(f"[{t_sec:.1f}s] Referee signaled a SUCCESSFUL 3-pointer.")
            
            # Boundary checks from the Rust Bridge
            if row.get("is_out_of_bounds"):
                events.append(f"[{t_sec:.1f}s] Ball or player went out of bounds.")

    # Synthesize Narrative
    print("\n--- HOOPSENSE GAME SUMMARY ---")
    if not events:
        print("No major events detected in this snippet.")
    else:
        # Deduplicate and print
        last_event = ""
        for e in events:
            if e != last_event:
                print(e)
                last_event = e
    print("------------------------------\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/validated_game_dna.jsonl"
    summarize_dna(path)

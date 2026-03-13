import json
import numpy as np
import argparse
from pathlib import Path


def load_jsonl(path):
    data = []
    if not Path(path).exists():
        return data
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_pck(gt_pose, pred_pose, threshold=0.05):
    """
    Computes Percentage of Correct Keypoints (PCK).
    threshold: normalized distance relative to bbox diagonal.
    """
    gt = np.array(gt_pose)
    pred = np.array(pred_pose)
    
    # Simple L2 distance on normalized coordinates
    distances = np.linalg.norm(gt - pred, axis=1)
    correct = distances < threshold
    return np.mean(correct)


def run_benchmark(gt_path, pred_path):
    print(f"[BENCHMARK] Comparing {gt_path} vs {pred_path}...")
    
    gt_data = load_jsonl(gt_path)
    pred_data = load_jsonl(pred_path)
    
    if not gt_data or not pred_data:
        print("[ERROR] Missing input data for benchmarking.")
        return

    # Filter pred_data to only include player events
    preds = [p for p in pred_data if p.get("kind") == "player"]
    
    stats = {
        "total_gt_samples": len(gt_data),
        "matches_found": 0,
        "id_accuracy": 0.0,
        "avg_pck": 0.0
    }

    pck_scores = []
    id_matches = 0

    for gt in gt_data:
        t_ms = gt["t_ms"]
        gt_tid = gt["track_id"]
        
        # Find corresponding prediction
        # Search for same t_ms and closest bbox/id
        match = None
        for p in preds:
            if abs(p["t_ms"] - t_ms) < 33: # Within 1 frame at 30fps
                # For now, we trust track_id matching for the audit
                if p["track_id"] == gt_tid:
                    match = p
                    break
        
        if match:
            stats["matches_found"] += 1
            if gt.get("type") == "id_verification":
                if gt["label"] == "correct":
                    id_matches += 1
            
            # If both have poses, compute PCK
            # (Requires pose to be in Shush-P JSONL, which we added recently)
            if "pose_2d" in match and "pose_2d" in gt:
                pck = compute_pck(gt["pose_2d"], match["pose_2d"])
                pck_scores.append(pck)

    if stats["matches_found"] > 0:
        stats["id_accuracy"] = id_matches / stats["matches_found"]
        if pck_scores:
            stats["avg_pck"] = np.mean(pck_scores)

    print("\n--- PERCEPTION BENCHMARK REPORT ---")
    print(f"GT Samples: {stats['total_gt_samples']}")
    print(f"Matches:    {stats['matches_found']}")
    print(f"ID Acc:     {stats['id_accuracy']:.2%}")
    print(f"Avg PCK:    {stats['avg_pck']:.2%}")
    print("-----------------------------------\n")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="data/training/manual_gt.jsonl")
    parser.add_argument("--pred", default="data/intelligent_game_dna.jsonl")
    args = parser.parse_args()
    
    run_benchmark(args.gt, args.pred)

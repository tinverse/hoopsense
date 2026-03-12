import json
import os

def validate_hoopsense_contract(file_path):
    required_keys = {
        "kind": str,
        "track_id": int,
        "t_ms": int,
        "x": float,
        "y": float,
        "w": float,
        "h": float,
        "confidence_bps": int
    }
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
                for key, expected_type in required_keys.items():
                    if key not in row:
                        raise ValueError(f"Line {i+1}: Missing key '{key}'")
                    if not isinstance(row[key], expected_type):
                        # Handle cases where ints might be serialized as floats in JSON
                        if expected_type == float and isinstance(row[key], int):
                            continue
                        raise TypeError(f"Line {i+1}: Key '{key}' expected {expected_type}, got {type(row[key])}")
            except json.JSONDecodeError:
                raise ValueError(f"Line {i+1}: Invalid JSON")
                
    print(f"Validation successful: {file_path}")

if __name__ == "__main__":
    test_path = "data/intelligent_game_dna.jsonl"
    if os.path.exists(test_path):
        validate_hoopsense_contract(test_path)
    else:
        print(f"No test file found at {test_path}")

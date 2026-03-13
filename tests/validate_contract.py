import json
import os
import sys
import tempfile


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
    if len(sys.argv) > 1:
        validate_hoopsense_contract(sys.argv[1])
    else:
        sample_row = {
            "kind": "player",
            "track_id": 1,
            "t_ms": 1000,
            "x": 10.0,
            "y": 20.0,
            "w": 30.0,
            "h": 40.0,
            "confidence_bps": 9000,
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as handle:
            handle.write(json.dumps(sample_row) + "\n")
            temp_path = handle.name
        try:
            validate_hoopsense_contract(temp_path)
        finally:
            os.unlink(temp_path)

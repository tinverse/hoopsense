import json
import numpy as np
import hashlib
from collections import Counter
from pathlib import Path


class DatasetManifest:
    """MLOps: Dataset Manifest and Validation for Oracle Outputs."""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.manifest_path = self.dataset_path.with_suffix('.manifest.json')
        self.stats = {
            "dataset_id": f"ds_{self.dataset_path.stem}",
            "schema_version": "2.0.0",
            "sha256_hash": None,
            "sample_count": 0,
            "class_distribution": {},
            "validation_status": "pending",
            "feature_shape": [30, 72],
            "errors": []
        }

    def _compute_hash(self):
        sha256 = hashlib.sha256()
        with open(self.dataset_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_and_generate(self):
        if not self.dataset_path.exists():
            return {"error": "Dataset file not found"}

        self.stats["sha256_hash"] = self._compute_hash()

        labels = []
        with open(self.dataset_path, 'r') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    feat = np.array(data["features_v2"])
                    if feat.shape != tuple(self.stats["feature_shape"]):
                        err = f"Sample {line_idx} has invalid shape {feat.shape}"
                        self.stats["errors"].append(err)
                    labels.append(data["label"])
                except Exception as e:
                    err = f"Line {line_idx} failed parse: {str(e)}"
                    self.stats["errors"].append(err)

        self.stats["sample_count"] = len(labels)
        self.stats["class_distribution"] = dict(Counter(labels))
        self.stats["validation_status"] = "pass" \
            if not self.stats["errors"] else "fail"

        with open(self.manifest_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        return self.stats


if __name__ == "__main__":
    import sys
    default_p = "data/training/synthetic_dataset_v2.jsonl"
    path = sys.argv[1] if len(sys.argv) > 1 else default_p
    manifest = DatasetManifest(path)
    report = manifest.validate_and_generate()
    print(json.dumps(report, indent=2))

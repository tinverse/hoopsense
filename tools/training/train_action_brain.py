import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import random
import subprocess
from collections import Counter
from core.vision.action_brain import ActionBrain
from pipelines.inference import get_label_map
from tools.data.dataset_manifest import DatasetManifest


def get_git_revision_hash():
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        return subprocess.check_output(cmd).decode('ascii').strip()
    except Exception:
        return "unknown"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LABEL_MAP_INV = get_label_map()
LABEL_MAP = {v: k for k, v in LABEL_MAP_INV.items()}


class SyntheticDataset(Dataset):
    def __init__(self, file_path):
        self.X, self.y = [], []
        self.label_counts = Counter()
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.X.append(np.array(data["features_v2"]))
                self.y.append(LABEL_MAP[data["label"]])
                self.label_counts[data["label"]] += 1
        self.active_labels = sorted(self.label_counts.keys())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]]).squeeze()
        return x, y


def train(epochs=150, force_cpu=False):
    set_seed(42)
    v3_path = "data/training/oracle_dataset_v3.jsonl"
    v2_path = "data/training/synthetic_dataset_v2.jsonl"
    data_path = v3_path if os.path.exists(v3_path) else v2_path
    if not os.path.exists(data_path):
        print(f"[ERROR] No data found at {data_path}")
        return

    # 1. MLOps: Dataset Manifest with Hash
    manifest = DatasetManifest(data_path)
    manifest_data = manifest.validate_and_generate()

    full_dataset = SyntheticDataset(data_path)
    train_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Training on {device}")
    model = ActionBrain(num_classes=len(LABEL_MAP)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 2. Record Grounded Lineage
    lineage = {
        "model_architecture": "ActionBrain-TemporalTransformer-V1",
        "git_commit": get_git_revision_hash(),
        "dataset_manifest": {
            "path": str(manifest.manifest_path),
            "sha256": manifest_data.get("sha256_hash")
        },
        "training_summary": {
            "epochs": epochs,
            "device": str(device),
            "best_acc": float(best_val_acc)
        },
        "timestamp": str(np.datetime64('now'))
    }
    with open(os.path.join(model_dir, "training_lineage.json"), 'w') as f:
        json.dump(lineage, f, indent=2)

    torch.save(model.state_dict(), os.path.join(model_dir, "action_brain.pt"))
    print("[SUCCESS] Training complete. Lineage saved.")


if __name__ == "__main__":
    import sys
    if "--smoke-test" in sys.argv:
        train(epochs=2, force_cpu=True)
    else:
        train()

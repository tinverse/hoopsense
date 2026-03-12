import json
import os
import random
import subprocess
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from core.vision.action_brain import ActionBrain
from pipelines.inference import get_label_map
from tools.data.dataset_manifest import DatasetManifest


def get_git_revision_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
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
        print(f"[INFO] Loading synthetic training data (Schema V2) from {file_path}")
        with open(file_path, "r") as handle:
            for line_idx, line in enumerate(handle):
                data = json.loads(line)
                features = np.array(data["features_v2"])
                label_name = data["label"]
                if label_name not in LABEL_MAP:
                    raise ValueError(f"CRITICAL: Unknown label '{label_name}' at line {line_idx}.")
                self.X.append(features)
                self.y.append(LABEL_MAP[label_name])
                self.label_counts[label_name] += 1

        self.active_labels = sorted(self.label_counts.keys())
        self.passive_labels = sorted([label for label in LABEL_MAP.keys() if label not in self.label_counts])
        print("\n--- Taxonomy Audit ---")
        print(f"Total Taxonomy Size: {len(LABEL_MAP)}")
        print(f"Active Classes (Present in Data): {len(self.active_labels)}")
        print(f"Passive Classes (Absent from Data): {len(self.passive_labels)}")
        print(f"Active Labels: {self.active_labels}")
        print("----------------------\n")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]]).squeeze()


def get_stratified_split(dataset, train_ratio=0.8):
    indices_per_class = defaultdict(list)
    for idx, label in enumerate(dataset.y):
        indices_per_class[label].append(idx)

    train_indices, val_indices = [], []
    for indices in indices_per_class.values():
        random.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue
        split_point = max(1, int(len(indices) * train_ratio))
        split_point = min(split_point, len(indices) - 1)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    if not val_indices:
        raise ValueError("Validation split is empty; need at least one held-out sample.")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def print_taxonomy_aware_summary(y_true, y_pred, dataset):
    active_names = dataset.active_labels
    active_ids = [LABEL_MAP[name] for name in active_names]
    cm = np.zeros((len(active_ids), len(active_ids)), dtype=int)
    id_to_idx = {id_: i for i, id_ in enumerate(active_ids)}

    for target, pred in zip(y_true, y_pred):
        if target in id_to_idx and pred in id_to_idx:
            cm[id_to_idx[target], id_to_idx[pred]] += 1

    print("\n--- TAXONOMY-AWARE VALIDATION REPORT (Active Classes Only) ---")
    print(f"{'Class':20} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Support'}")
    print("-" * 55)
    for i, name in enumerate(active_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{name[:20]:20} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f} | {support}")

    print("\n--- CONFUSION MATRIX ---")
    header = " " * 15 + " ".join([f"{n[:3]:>3}" for n in active_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{active_names[i][:12]:>12} | " + " ".join([f"{val:>3}" for val in row])
        print(row_str)
    print("------------------------------------------------------\n")


def train(epochs=150, force_cpu=False):
    set_seed(42)
    v3_path = "data/training/oracle_dataset_v3.jsonl"
    v2_path = "data/training/synthetic_dataset_v2.jsonl"
    data_path = v3_path if os.path.exists(v3_path) else v2_path
    if not os.path.exists(data_path):
        print(f"[ERROR] No training data found at {v3_path} or {v2_path}.")
        return

    manifest = DatasetManifest(data_path)
    manifest_data = manifest.validate_and_generate()
    if manifest_data.get("validation_status") != "pass":
        raise ValueError(f"Dataset manifest validation failed: {manifest_data.get('errors')}")

    full_dataset = SyntheticDataset(data_path)
    train_dataset, val_dataset = get_stratified_split(full_dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Target Device: {device}")

    model = ActionBrain(num_classes=len(LABEL_MAP)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best_val_acc = 0.0
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(target.cpu().numpy())

        val_acc = float(np.mean(np.array(all_preds) == np.array(all_trues)))
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "action_brain.pt"))
            if epoch > 0 and (epoch + 1) % 20 == 0:
                print(f"[CHECKPOINT] Epoch {epoch + 1}: Best Val Acc {val_acc:.4f}")

    model.load_state_dict(torch.load(os.path.join(model_dir, "action_brain.pt"), map_location=device))
    model.eval()
    final_preds, final_trues = [], []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            final_preds.extend(preds.cpu().numpy())
            final_trues.extend(target.cpu().numpy())

    print_taxonomy_aware_summary(final_trues, final_preds, full_dataset)

    lineage = {
        "model_architecture": "ActionBrain-TemporalTransformer-V1",
        "git_commit": get_git_revision_hash(),
        "dataset_manifest": {
            "path": str(manifest.manifest_path),
            "sha256": manifest_data.get("sha256_hash"),
        },
        "training_summary": {
            "epochs": epochs,
            "device": str(device),
            "best_val_accuracy": best_val_acc,
            "dataset_path": data_path,
            "sample_count": len(full_dataset),
        },
        "timestamp": str(np.datetime64("now")),
    }
    with open(os.path.join(model_dir, "training_lineage.json"), "w") as handle:
        json.dump(lineage, handle, indent=2)

    print(f"[SUCCESS] Training complete. Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    import sys

    if "--smoke-test" in sys.argv:
        print("[SMOKE TEST] Running 2 epochs on CPU...")
        train(epochs=2, force_cpu=True)
    else:
        train()

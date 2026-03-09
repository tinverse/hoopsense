import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import json
import numpy as np
import os
import random
from collections import Counter, defaultdict
from core.vision.action_brain import ActionBrain
from pipelines.inference import get_label_map

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Synchronized Label Map
LABEL_MAP_INV = get_label_map()
LABEL_MAP = {v: k for k, v in LABEL_MAP_INV.items()}

class SyntheticDataset(Dataset):
    def __init__(self, file_path):
        self.X, self.y = [], []
        self.label_names = []
        self.label_counts = Counter()
        print(f"[INFO] Loading synthetic training data (Schema V2) from {file_path}")
        with open(file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line)
                features = np.array(data["features_v2"])
                label_name = data["label"]
                
                if label_name not in LABEL_MAP:
                    raise ValueError(f"CRITICAL: Unknown label '{label_name}' at line {line_idx}. "
                                     f"Label map is synchronized with {len(LABEL_MAP)} classes.")
                
                self.X.append(features)
                self.y.append(LABEL_MAP[label_name])
                self.label_names.append(label_name)
                self.label_counts[label_name] += 1
        
        print(f"[INFO] Dataset Loaded. Total samples: {len(self.X)}")
        print(f"[INFO] Active Taxonomy ({len(self.label_counts)}/{len(LABEL_MAP)} classes): {dict(self.label_counts)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]]).squeeze()

def get_stratified_split(dataset, train_ratio=0.8):
    """Manually implements stratified split for small datasets."""
    indices_per_class = defaultdict(list)
    for idx, label in enumerate(dataset.y):
        indices_per_class[label].append(idx)
    
    train_indices, val_indices = [], []
    for label, indices in indices_per_class.items():
        random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def train():
    set_seed(42)
    data_path = "data/training/synthetic_dataset_v2.jsonl"
    if not os.path.exists(data_path):
        print("[ERROR] No training data found. Run generator first.")
        return

    # 1. Initialize Dataset & Stratified Split
    full_dataset = SyntheticDataset(data_path)
    train_dataset, val_dataset = get_stratified_split(full_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Target Device: {device}")
    
    model = ActionBrain(num_classes=len(LABEL_MAP)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_acc = 0.0
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(150):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation Phase with Per-Class Reporting
        model.eval()
        val_correct = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                
                for pred, t in zip(preds, target):
                    label_name = LABEL_MAP_INV[t.item()]
                    class_total[label_name] += 1
                    if pred == t:
                        val_correct += 1
                        class_correct[label_name] += 1
        
        val_acc = val_correct / len(val_dataset)
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "action_brain.pt"))
            if (epoch + 1) > 20:
                print(f"\n[CHECKPOINT] Epoch {epoch+1}: New Best Val Acc: {val_acc:.4f}")
                print("--- Per-Class Validation Report ---")
                for label in sorted(full_dataset.label_counts.keys()):
                    acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
                    print(f"  {label:20}: {acc:.2f} ({class_correct[label]}/{class_total[label]})")

        if (epoch + 1) % 50 == 0:
            avg_train_loss = train_loss / len(train_loader)
            print(f"\nEpoch [{epoch+1}/150] | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\n[SUCCESS] Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()

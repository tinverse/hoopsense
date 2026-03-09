import torch
import torch.optim as optim
import json
import numpy as np
import os
from core.vision.action_brain import ActionBrain

# Map labels to indices
LABEL_MAP = {"jump_shot": 0, "crossover": 1, "rebound": 2, "block": 3, "steal": 4}

def load_synthetic_data(file_path):
    print(f"[INFO] Loading synthetic training data (Schema V2) from {file_path}")
    X, y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Use the 72-dim feature tensor (30 frames x 72 features)
            features = np.array(data["features_v2"])
            X.append(features)
            y.append(LABEL_MAP[data["label"]])
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y))

def train():
    data_path = "data/training/synthetic_dataset_v2.jsonl"
    if not os.path.exists(data_path):
        print("[ERROR] No training data found. Run generator first.")
        return

    # 1. Load Data
    X_train, y_train = load_synthetic_data(data_path)
    
    # 2. Initialize Model
    model = ActionBrain(num_classes=len(LABEL_MAP))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("[INFO] Starting Action Brain Training...")
    model.train()
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    # 3. Save the Trained Brain
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/action_brain.pt")
    print("[SUCCESS] Action Brain training complete. Model saved.")

if __name__ == "__main__":
    train()

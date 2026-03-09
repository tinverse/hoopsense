import torch
import torch.nn as nn
import os

class ActionBrain(nn.Module):
    """
    A Temporal Transformer for Basketball Action Recognition.
    Capacity is tuned for current synthetic dataset scale.
    """
    def __init__(self, input_dim=72, model_dim=128, num_heads=4, num_layers=2, num_classes=64):
        super(ActionBrain, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 2,
            batch_first=True,
            dropout=0.3 # Increased dropout for regularization
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(0.3),
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def save_model(model, path="data/models/action_brain.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Action Brain saved to {path}")

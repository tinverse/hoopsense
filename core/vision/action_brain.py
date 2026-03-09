import torch
import torch.nn as nn
import os

class ActionBrain(nn.Module):
    """
    A Custom Temporal Transformer for Basketball Action Recognition.
    Input: (Batch, Seq_Len, 72) - 30 frames of multimodal features.
    """
    def __init__(self, input_dim=72, model_dim=256, num_heads=8, num_layers=3, num_classes=5):
        super(ActionBrain, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes)
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

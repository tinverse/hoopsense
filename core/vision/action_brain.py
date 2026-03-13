import torch
import torch.nn as nn


class ActionBrain(nn.Module):
    """
    Temporal Transformer for local action recognition.
    Input: (Batch, Frames=30, Dims=72)
    Output: Action probabilities.
    """

    def __init__(self, num_classes=5, input_dim=72, embed_dim=128, nhead=4):
        super().__init__()
        # Input Projection: Map Schema V2 (72 dims) to Transformer embedding space
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Positional Encoding (Learnable for MVP simplicity)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 30, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 30, 72)
        x = self.input_projection(x) + self.pos_embedding
        x = self.transformer(x)

        # Aggregate over time (Mean pooling)
        x = x.mean(dim=1)
        return self.classifier(x)


def main():
    # Smoke test initialization
    model = ActionBrain()
    dummy_input = torch.randn(1, 30, 72)
    output = model(dummy_input)
    print(f"[INFO] ActionBrain Initialized. Output shape: {output.shape}")


if __name__ == "__main__":
    main()

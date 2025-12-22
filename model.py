2import torch
import torch.nn as nn

class BiLSTMSkeletonEmotion(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 2, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # x: (B,T,D)
        x = self.proj(x)
        out, _ = self.lstm(x)      # (B,T,2H)
        feat = out[:, -1, :]       # last timestep
        return self.head(feat)

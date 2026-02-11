import torch
import torch.nn as nn

class SharedTransformer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, x):
        return self.encoder(x)

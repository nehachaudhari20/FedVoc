import torch
import torch.nn as nn


class LowRankAdapter(nn.Module):
    def __init__(self, d_model=768, rank=64):
        super().__init__()

        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)

    def forward(self, x):
        return self.B(self.A(x))

import torch.nn as nn


class LowRankAdapter(nn.Module):
    """
    Low-rank adapter for embedding alignment.

    Changes from HEAVY version:
    - rank kept at 32 (good balance — 64 was too big, 16 too small for random init)
    - B still zero-initialized (free improvement, no compute cost)
    - LayerNorm kept (free improvement, negligible compute)
    - NO pretrained DistilBERT dependency
    """

    def __init__(self, d_model=768, rank=32, alpha=32):
        super().__init__()

        self.scaling = alpha / rank

        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.norm(x + self.B(self.A(x)) * self.scaling)

import torch.nn as nn


class LowRankAdapter(nn.Module):
    """
    Low-rank adapter for embedding alignment.

    Changes from original:
    - rank reduced from 64 → 16 (less overfitting, faster training, smaller comm cost)
    - B initialized to zero so adapter starts as identity (standard LoRA practice)
    - LayerNorm on the residual output for training stability
    - scaling factor (alpha/rank) to control adapter contribution magnitude
    """

    def __init__(self, d_model=768, rank=16, alpha=32):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # controls how much adapter contributes

        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # Standard LoRA init: A ~ N(0,1), B = 0
        # This means adapter(x) = 0 at round 0 → starts from pretrained encoder state
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        adapter_out = self.B(self.A(x)) * self.scaling
        return self.norm(x + adapter_out)

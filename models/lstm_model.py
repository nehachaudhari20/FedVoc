"""
models/lstm_model.py — LSTM-based language model for FedVoc/FedAvg comparison.

Architecture mirrors FedVocModel but replaces DistilBERT encoder with a
multi-layer LSTM. This lets us compare:
  - Transformer (DistilBERT + adapter) vs LSTM
  - FedVoc vs FedAvg
  across the same domains and training setup.

Design choices:
  - Same LOCAL/SHARED split as FedVocModel:
      LOCAL:  embedding, lm_head  (vocab-specific, never aggregated)
      SHARED: lstm, adapter       (aggregated across clients)
  - LSTMAdapter mirrors LowRankAdapter but operates on LSTM hidden states
  - Input dropout + LSTM dropout for regularisation
  - No pretrained weights — LSTM trains from scratch (expected to converge slower)
"""

import torch
import torch.nn as nn


class LSTMAdapter(nn.Module):
    """
    Low-rank adapter for LSTM hidden states.
    Identical structure to LowRankAdapter — rank-16 bottleneck + LayerNorm residual.
    Shared across clients during federated aggregation.
    """

    def __init__(self, d_model=512, rank=16, alpha=32):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)   # identity at init (standard LoRA)

    def forward(self, x):
        return self.norm(x + self.B(self.A(x)) * self.scaling)


class FedVocLSTMModel(nn.Module):
    """
    LSTM-based FedVoc model.

    LOCAL  (never aggregated): embedding, lm_head
    SHARED (aggregated):       lstm, adapter

    Args:
        vocab_size: client-specific vocabulary size
        d_model:    embedding + LSTM hidden dimension (default 512)
        num_layers: number of LSTM layers (default 2)
        rank:       adapter bottleneck rank (default 16)
        dropout:    dropout probability (default 0.3)
    """

    def __init__(self, vocab_size, d_model=512, num_layers=2, rank=16, dropout=0.3):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # ── LOCAL ──────────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size)
        self.input_drop = nn.Dropout(dropout)

        # ── SHARED ─────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.adapter = LSTMAdapter(d_model, rank)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None, hidden=None):
        """
        Args:
            input_ids:      (B, T) token ids
            attention_mask: (B, T) — used for length-masked packing if provided
            hidden:         optional LSTM hidden state tuple for stateful inference

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.input_drop(self.embedding(input_ids))   # (B, T, d_model)

        # Pack padded sequences for efficiency when mask is available
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, hidden = self.lstm(packed, hidden)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
        else:
            lstm_out, hidden = self.lstm(x, hidden)      # (B, T, d_model)

        # Adapter on LSTM output — shared, aggregated each round
        out = self.adapter(lstm_out)
        out = self.output_norm(out)

        logits = self.lm_head(out)                        # (B, T, vocab_size)
        return logits

import torch
import torch.nn as nn


class LSTMAdapter(nn.Module):
    """
    Low-rank adapter for LSTM — same concept as transformer adapter.
    Aligns local embedding space before feeding into shared LSTM encoder.

    Same design as transformer adapter:
    - Zero-init B so adapter starts as identity
    - LayerNorm for stability
    - Scaling factor alpha/rank
    """

    def __init__(self, d_model=128, rank=16, alpha=16):
        super().__init__()

        self.scaling = alpha / rank
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.norm(x + self.B(self.A(x)) * self.scaling)


class FedVocLSTMModel(nn.Module):
    """
    FedVoc LSTM model — lightweight version for ~20 min GPU training.

    Architecture:
        LOCAL  (never shared): embedding, lm_head
        SHARED (aggregated):   adapter, lstm_encoder

    Why these settings:
        d_model=128, num_layers=1 — small enough to train fast,
        big enough to learn domain patterns in 15 rounds.
        No pretrained weights needed — LSTM trains from scratch cleanly.

    Compared to transformer version:
        ~800K total params vs ~66M for DistilBERT
        Much faster per round, similar domain accuracy
    """

    def __init__(self, vocab_size, d_model=128, num_layers=1, dropout=0.3, rank=16):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # LOCAL — vocab-specific, never aggregated
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # SHARED — aggregated across clients each round
        self.adapter = LSTMAdapter(d_model, rank)
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Tie embedding and lm_head weights — reduces params, improves LM quality
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, hidden=None):
        """
        Args:
            input_ids: (batch, seq_len)
            hidden:    optional (h_0, c_0) — pass None to reset each batch

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (h_n, c_n)
        """
        x = self.embedding(input_ids)
        x = self.adapter(x)
        x = self.dropout(x)

        output, hidden = self.lstm_encoder(x, hidden)
        output = self.dropout(output)

        logits = self.lm_head(output)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.d_model, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.d_model, device=device)
        return h, c

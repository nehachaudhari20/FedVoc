import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel
from models.adapter import LowRankAdapter


class FedVocModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, rank=64):
        
        super().__init__()

        # LOCAL embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # ADAPTER (shared)
        self.adapter = LowRankAdapter(d_model, rank)

        # SHARED encoder (vocab independent)
        config = DistilBertConfig(
            vocab_size=30522,   # dummy, unused
            dim=d_model,
            hidden_dim=4 * d_model,
            n_layers=6,
            n_heads=12,
        )

        self.encoder = DistilBertModel(config)

        # LOCAL lm head
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):

        x = self.embedding(input_ids)

        # ALIGNMENT STEP
        x = x + self.adapter(x)
        x = self.dropout(x)

        outputs = self.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        logits = self.lm_head(hidden)

        return logits

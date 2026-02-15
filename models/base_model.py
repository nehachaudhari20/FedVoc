import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel


class DistilBertLM(nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()

        config = DistilBertConfig(
            vocab_size=vocab_size,
            dim=d_model,
            hidden_dim=4 * d_model,
            n_layers=6,          # FULL DistilBERT
            n_heads=12,
            dropout=0.1,
        )

        self.bert = DistilBertModel(config)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

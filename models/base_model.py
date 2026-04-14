import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel
from models.adapter import LowRankAdapter


class FedVocModel(nn.Module):
    """
    FedVoc model — lightweight version for ~30 min GPU training.

    LOCAL  (never shared): embedding, lm_head
    SHARED (aggregated):   adapter, encoder

    Changes from HEAVY version:
    - load_pretrained_encoder() REMOVED — was the single biggest time killer
      (downloads 268MB + runs 66M params from a frozen checkpoint)
    - Random init kept — with 3000-sample cap it converges fine in 15 rounds
    - Everything else identical to improved version
    """

    def __init__(self, vocab_size, d_model=768, rank=32):
        super().__init__()

        # LOCAL — vocab-specific, never aggregated
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

        # SHARED — aggregated across clients each round
        self.adapter = LowRankAdapter(d_model, rank)

        config = DistilBertConfig(
            vocab_size=30522,
            dim=d_model,
            hidden_dim=4 * d_model,
            n_layers=6,
            n_heads=12,
        )
        self.encoder = DistilBertModel(config)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.adapter(x)
        x = self.dropout(x)

        outputs = self.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask
        )

        logits = self.lm_head(outputs.last_hidden_state)
        return logits

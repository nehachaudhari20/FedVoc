import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel
from models.adapter import LowRankAdapter


class FedVocModel(nn.Module):
    """
    FedVoc model with:
    - LOCAL:  embedding, lm_head  (never shared, vocab-specific)
    - SHARED: adapter, encoder    (aggregated across clients)

    Changes from original:
    - Supports loading pretrained DistilBERT encoder weights via load_pretrained_encoder()
    - Adapter rank reduced from 64 → 16 (see adapter.py)
    - Adapter initialized to identity so pretrained encoder is undisturbed at round 0
    """

    def __init__(self, vocab_size, d_model=768, rank=16):
        super().__init__()

        # LOCAL — vocab-specific, never aggregated
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

        # SHARED — aggregated across clients
        self.adapter = LowRankAdapter(d_model, rank)

        config = DistilBertConfig(
            vocab_size=30522,   # dummy, not used (we pass inputs_embeds)
            dim=d_model,
            hidden_dim=4 * d_model,
            n_layers=6,
            n_heads=12,
        )
        self.encoder = DistilBertModel(config)

    def load_pretrained_encoder(self):
        """
        Load pretrained DistilBERT weights into the encoder.
        This is the single highest-leverage improvement — the encoder already
        knows language structure so the adapter only needs to learn alignment.
        Call once before federated training begins.
        """
        pretrained = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.encoder.load_state_dict(pretrained.state_dict())
        print("Loaded pretrained DistilBERT encoder weights.")

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)

        # Adapter aligns local embedding space → shared encoder space
        x = self.adapter(x)
        x = self.dropout(x)

        outputs = self.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask
        )

        logits = self.lm_head(outputs.last_hidden_state)
        return logits

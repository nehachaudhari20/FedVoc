import torch
import torch.nn as nn
import copy

class FedAvgClient:
    def __init__(self, tokenizer, texts, d_model=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.device = torch.device("cpu")

        self.d_model = d_model
        self.embedding = None
        self.lm_head = None

        self.criterion = nn.CrossEntropyLoss()

    def initialize_local_model(self, global_model):
        # Create fresh copy of global model
        self.model = copy.deepcopy(global_model)

        vocab_size = self.tokenizer.get_vocab_size()

        # Local embedding + local LM head
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

    def encode_batch(self, text, seq_len=32):
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:seq_len]

        if len(ids) < seq_len:
            ids += [0] * (seq_len - len(ids))

        return torch.tensor(ids)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for text in self.texts[:100]:
            tokens = self.encode_batch(text).unsqueeze(0)

            emb = self.embedding(tokens)
            out = self.model(emb)
            logits = self.lm_head(out)

            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tokens.view(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss

    def get_model_weights(self):
        return copy.deepcopy(self.model.state_dict())

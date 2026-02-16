import torch
import torch.nn as nn
import torch.optim as optim
from models.base_model import FedVocModel
from torch.nn.utils.rnn import pad_sequence


class FedVocClient:
    def __init__(self, tokenizer, texts, device="cpu"):
        self.tokenizer = tokenizer
        self.texts = texts
        self.device = device

        self.vocab_size = tokenizer.get_vocab_size()

        self.model = FedVocModel(self.vocab_size).to(device)

        # 🔥 Freeze encoder for stability
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def initialize_local_adapter(self, global_adapter_state):
        self.model.adapter.load_state_dict(global_adapter_state)

    def _prepare_batch(self, texts, max_len=32):
        input_ids_list = []

        for text in texts:
            ids = self.tokenizer.encode(text).ids[:max_len]
            if len(ids) < 2:
                continue
            input_ids_list.append(torch.tensor(ids))

        if len(input_ids_list) == 0:
            return None, None, None

        padded = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=0
        )

        attention_mask = (padded != 0).long()

        inputs = padded[:, :-1]
        targets = padded[:, 1:]

        return inputs, targets, attention_mask[:, :-1]

    def train_one_epoch(self, batch_size=16):
        self.model.train()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=2e-4
        )

        criterion = nn.CrossEntropyLoss(ignore_index=0)

        total_loss = 0
        steps = 0

        for i in range(0, min(len(self.texts), 800), batch_size):

            batch_texts = self.texts[i:i + batch_size]

            inputs, targets, mask = self._prepare_batch(batch_texts)

            if inputs is None:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            optimizer.zero_grad()

            logits = self.model(inputs, mask)

            loss = criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def get_adapter_weights(self):
        return self.model.adapter.state_dict()

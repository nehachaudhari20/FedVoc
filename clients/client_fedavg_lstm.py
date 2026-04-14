import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from models.lstm_model import FedVocLSTMModel


class FedAvgLSTMClient:
    """
    FedAvg baseline client using LSTM model — lightweight version.

    Mirrors client_fedavg.py behaviour exactly:
        - Full model shared (embedding + adapter + lstm_encoder + lm_head)
        - Global tokenizer — same vocab for all clients
        - 3000-sample cap
        - Cosine LR scheduler
        - Weighted aggregation handled in run_fedavg_lstm.py
    """

    def __init__(self, tokenizer, texts, device=None,
                 d_model=128, num_layers=1, dropout=0.3, rank=16):
        self.tokenizer = tokenizer
        self.texts = texts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = tokenizer.token_to_id("[PAD]")

        self.model = FedVocLSTMModel(
            vocab_size=self.vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            rank=rank,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-5
        )

    def initialize_local_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

    def _prepare_batch(self, texts, max_len=80):
        input_ids_list = []
        for text in texts:
            ids = self.tokenizer.encode(text).ids[:max_len]
            if len(ids) < 2:
                continue
            input_ids_list.append(torch.tensor(ids))

        if not input_ids_list:
            return None, None

        padded = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_id
        )
        inputs = padded[:, :-1]
        targets = padded[:, 1:]
        return inputs, targets

    def train_one_epoch(self, batch_size=32):
        self.model.train()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, label_smoothing=0.1
        )

        total_loss = 0
        steps = 0

        for i in range(0, min(len(self.texts), 3000), batch_size):
            batch_texts = self.texts[i:i + batch_size]
            inputs, targets = self._prepare_batch(batch_texts)
            if inputs is None:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(inputs, hidden=None)

            loss = criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def step_scheduler(self):
        self.scheduler.step()

    def get_model_weights(self):
        return self.model.state_dict()

    def evaluate(self, test_texts, batch_size=32):
        self.model.eval()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, label_smoothing=0.1
        )

        total_loss = 0
        steps = 0

        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                inputs, targets = self._prepare_batch(batch_texts)
                if inputs is None:
                    continue

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits, _ = self.model(inputs, hidden=None)
                loss = criterion(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                )
                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity

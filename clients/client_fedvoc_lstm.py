import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from models.lstm_model import FedVocLSTMModel


class FedVocLSTMClient:
    """
    FedVoc client using LSTM encoder — lightweight version.

    Same bug fixes as transformer FedVoc client:
        - Optimizer NOT rebuilt every round — momentum preserved
        - Optimizer only rebuilt when encoder freeze state actually changes
        - get_shared_weights() returns adapter ONLY
        - initialize_shared_weights() does not touch the optimizer
        - Proper global init handled in run_fedvoc_lstm.py
        - Weighted aggregation handled in run_fedvoc_lstm.py

    Kept lightweight (same as transformer version):
        - 3000-sample cap per epoch
        - No FedProx
        - No pretrained weights
        - 15 rounds

    Key difference from transformer client:
        - No attention_mask needed — LSTM handles sequences natively
        - Hidden state reset each batch (no BPTT across batches)
        - d_model=128, layers=1 — much smaller than DistilBERT
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

        # Freeze LSTM encoder initially — same warm-up as transformer
        for param in self.model.lstm_encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True

        self.optimizer = self._build_optimizer(lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-5
        )

    def _build_optimizer(self, lr=1e-3):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

    def set_encoder_frozen(self, frozen: bool):
        """
        BUG FIX: only rebuilds optimizer when freeze state actually changes.
        """
        if frozen == self._encoder_frozen:
            return

        for param in self.model.lstm_encoder.parameters():
            param.requires_grad = not frozen
        self._encoder_frozen = frozen

        self.optimizer = self._build_optimizer(lr=self.scheduler.get_last_lr()[0])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-5
        )
        state = "frozen" if frozen else "unfrozen"
        print(f"  LSTM encoder {state} — optimizer rebuilt.")

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
        """
        batch_size=32 for LSTM (larger than transformer's 16)
        LSTM is lighter so more samples fit in GPU memory per batch.
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, label_smoothing=0.1
        )

        total_loss = 0
        steps = 0

        # 3000-sample cap — same as transformer version
        for i in range(0, min(len(self.texts), 3000), batch_size):
            batch_texts = self.texts[i:i + batch_size]
            inputs, targets = self._prepare_batch(batch_texts)
            if inputs is None:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Fresh hidden state each batch — no BPTT across batches
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

    def get_shared_weights(self):
        """
        BUG FIX: adapter ONLY — same strategy as transformer FedVoc.
        LSTM adapter is even smaller: ~8K params with d_model=128, rank=16.
        """
        return {
            "adapter." + k: v
            for k, v in self.model.adapter.state_dict().items()
        }

    def initialize_shared_weights(self, global_shared_state):
        """
        BUG FIX: loads adapter weights WITHOUT rebuilding the optimizer.
        """
        adapter_state = {
            k.replace("adapter.", ""): v
            for k, v in global_shared_state.items()
            if k.startswith("adapter.")
        }
        self.model.adapter.load_state_dict(adapter_state)

    def step_scheduler(self):
        self.scheduler.step()

    def get_current_lr(self):
        return self.scheduler.get_last_lr()[0]

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

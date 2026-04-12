"""
clients/client_fedvoc_lstm.py — FedVoc client using the LSTM model.

Mirrors client_fedvoc.py exactly, swapping FedVocModel → FedVocLSTMModel.
All FedVoc mechanics are preserved:
  - Encoder freeze toggling (here: LSTM layer freeze)
  - Adapter-only sharing (~24K params vs full model)
  - FedProx proximal term to prevent client drift
  - Cosine LR scheduler
  - Optimizer momentum preserved across rounds
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from models.lstm_model import FedVocLSTMModel


class FedVocLSTMClient:
    """
    FedVoc client with an LSTM language model.

    LOCAL  (never shared): embedding, lm_head
    SHARED (aggregated):   lstm, adapter

    Drop-in replacement for FedVocClient — same public API:
        set_encoder_frozen(), initialize_shared_weights(),
        train_one_epoch(), get_shared_weights(),
        step_scheduler(), get_current_lr(), evaluate()
    """

    def __init__(self, tokenizer, texts, device=None,
                 d_model=512, num_layers=2, dropout=0.3):
        self.tokenizer  = tokenizer
        self.texts      = texts
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id     = tokenizer.token_to_id("[PAD]")

        self.model = FedVocLSTMModel(
            self.vocab_size, d_model=d_model,
            num_layers=num_layers, dropout=dropout
        ).to(self.device)

        # Freeze LSTM initially (train adapter only for warm-up rounds)
        for param in self.model.lstm.parameters():
            param.requires_grad = False
        self._lstm_frozen = True

        self.optimizer = self._build_optimizer(lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-5
        )

        self._global_params = None  # FedProx snapshot

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def _build_optimizer(self, lr=3e-4):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

    def set_encoder_frozen(self, frozen: bool):
        """
        Toggle LSTM freeze.  Mirrors FedVocClient.set_encoder_frozen() — named
        identically so run_fedvoc_lstm.py can call it without branching.
        Only rebuilds the optimizer when the freeze state actually changes.
        """
        if frozen == self._lstm_frozen:
            return

        for param in self.model.lstm.parameters():
            param.requires_grad = not frozen
        self._lstm_frozen = frozen

        self.optimizer = self._build_optimizer(lr=self.scheduler.get_last_lr()[0])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-5
        )
        state = "frozen" if frozen else "unfrozen"
        print(f"  LSTM {state} — optimizer rebuilt.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare_batch(self, texts, max_len=80):
        input_ids_list = []
        for text in texts:
            ids = self.tokenizer.encode(text).ids[:max_len]
            if len(ids) < 2:
                continue
            input_ids_list.append(torch.tensor(ids))

        if not input_ids_list:
            return None, None, None

        padded = pad_sequence(input_ids_list, batch_first=True,
                              padding_value=self.pad_id)
        attention_mask = (padded != self.pad_id).long()
        inputs  = padded[:, :-1]
        targets = padded[:, 1:]
        return inputs, targets, attention_mask[:, :-1]

    # ── Shared weights (adapter only) ─────────────────────────────────────────

    def get_shared_weights(self):
        """
        Return adapter + lstm weights for aggregation.

        Unlike the transformer version (encoder is frozen + pretrained so
        we skip it), here we share BOTH lstm and adapter because:
          - The LSTM has no pretrained weights to preserve
          - Sharing lstm weights is what gives FedVoc its cross-client benefit
          - Total: lstm (~8.4M) + adapter (~24K) — still << transformer encoder (66.5M)
        """
        shared = {}
        for k, v in self.model.lstm.state_dict().items():
            shared["lstm." + k] = v
        for k, v in self.model.adapter.state_dict().items():
            shared["adapter." + k] = v
        return shared

    def initialize_shared_weights(self, global_shared_state):
        """
        Load global lstm + adapter weights into local model.
        Saves a snapshot for FedProx. Does NOT rebuild the optimizer.
        """
        lstm_state = {
            k.replace("lstm.", ""): v
            for k, v in global_shared_state.items()
            if k.startswith("lstm.")
        }
        adapter_state = {
            k.replace("adapter.", ""): v
            for k, v in global_shared_state.items()
            if k.startswith("adapter.")
        }
        if lstm_state:
            self.model.lstm.load_state_dict(lstm_state)
        if adapter_state:
            self.model.adapter.load_state_dict(adapter_state)

        # Snapshot for FedProx
        self._global_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train_one_epoch(self, batch_size=16, mu=0.01):
        """
        Train for one epoch with optional FedProx proximal term.

        Args:
            batch_size: mini-batch size
            mu:         FedProx coefficient (0 = disabled)
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, label_smoothing=0.1
        )

        # Snapshot for FedProx (taken once per epoch, not per batch)
        global_snapshot = (
            {n: p.clone().detach()
             for n, p in self.model.named_parameters() if p.requires_grad}
            if mu > 0 and self._global_params is not None else None
        )

        total_loss, steps = 0, 0

        for i in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[i:i + batch_size]
            inputs, targets, mask = self._prepare_batch(batch_texts)
            if inputs is None:
                continue

            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)
            mask    = mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs, mask)
            loss   = criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

            # FedProx proximal term
            if global_snapshot is not None:
                prox = sum(
                    (p - global_snapshot[n]).norm(2) ** 2
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and n in global_snapshot
                )
                loss = loss + (mu / 2) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps      += 1

        return total_loss / max(steps, 1)

    def step_scheduler(self):
        self.scheduler.step()

    def get_current_lr(self):
        return self.scheduler.get_last_lr()[0]

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test_texts, batch_size=16):
        self.model.eval()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, label_smoothing=0.1
        )

        total_loss, steps = 0, 0

        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                inputs, targets, mask = self._prepare_batch(batch_texts)
                if inputs is None:
                    continue

                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)
                mask    = mask.to(self.device)

                logits = self.model(inputs, mask)
                loss   = criterion(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1)
                )
                total_loss += loss.item()
                steps      += 1

        avg_loss   = total_loss / max(steps, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity

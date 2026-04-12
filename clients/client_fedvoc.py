import torch
import torch.nn as nn
import torch.optim as optim
from models.base_model import FedVocModel
from torch.nn.utils.rnn import pad_sequence


class FedVocClient:
    """
    FedVoc client.

    Changes from original:
    - Optimizer is NOT recreated on every round — momentum is preserved
    - Optimizer is only rebuilt when requires_grad actually changes (freeze toggle)
    - FedProx proximal term prevents client drift during local training
    - Full dataset used per epoch (was capped at 3000 samples)
    - Cosine LR scheduler support (stepped externally from run_fedvoc.py)
    - get_shared_weights() returns adapter ONLY (not the full encoder)
      → communication cost drops from 66.5M to ~24K params per round
    - initialize_shared_weights() no longer rebuilds the optimizer
    """

    def __init__(self, tokenizer, texts, device=None):
        self.tokenizer = tokenizer
        self.texts = texts
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = tokenizer.get_vocab_size()
        self.model = FedVocModel(self.vocab_size).to(self.device)
        self.pad_id = tokenizer.token_to_id("[PAD]")

        # Freeze encoder initially
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True

        # Build optimizer once — never rebuild unless freeze state changes
        self.optimizer = self._build_optimizer(lr=3e-4)

        # Cosine scheduler — stepped each round from run_fedvoc.py
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-5
        )

        # Global model snapshot for FedProx proximal term
        self._global_params = None

    def _build_optimizer(self, lr=3e-4):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def set_encoder_frozen(self, frozen: bool):
        """
        Toggle encoder freeze. Only rebuilds the optimizer if the state actually changes,
        preserving Adam momentum across rounds when nothing changes.
        """
        if frozen == self._encoder_frozen:
            return  # no change — keep optimizer as-is

        for param in self.model.encoder.parameters():
            param.requires_grad = not frozen
        self._encoder_frozen = frozen

        # Rebuild optimizer only now (momentum is legitimately stale after param set change)
        self.optimizer = self._build_optimizer(lr=self.scheduler.get_last_lr()[0])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-5
        )
        state = "frozen" if frozen else "unfrozen"
        print(f"  Encoder {state} — optimizer rebuilt.")

    def _prepare_batch(self, texts, max_len=80):
        input_ids_list = []
        for text in texts:
            ids = self.tokenizer.encode(text).ids[:max_len]
            if len(ids) < 2:
                continue
            input_ids_list.append(torch.tensor(ids))

        if not input_ids_list:
            return None, None, None

        padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_id)
        attention_mask = (padded != self.pad_id).long()
        inputs = padded[:, :-1]
        targets = padded[:, 1:]
        return inputs, targets, attention_mask[:, :-1]

    def train_one_epoch(self, batch_size=16, mu=0.01):
        """
        Train for one epoch.

        Args:
            batch_size: mini-batch size
            mu: FedProx proximal coefficient. Set to 0 to disable.
               Prevents local model from drifting too far from global weights.
        """
        self.model.train()

        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=0.1
        )

        # Snapshot global params for FedProx at the start of each epoch
        if mu > 0 and self._global_params is not None:
            global_snapshot = {
                n: p.clone().detach()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }
        else:
            global_snapshot = None

        total_loss = 0
        steps = 0

        # FIX: use full dataset, not min(len, 3000)
        for i in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[i:i + batch_size]
            inputs, targets, mask = self._prepare_batch(batch_texts)

            if inputs is None:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(inputs, mask)
            loss = criterion(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

            # FedProx proximal term: penalise drift from global weights
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
            steps += 1

        return total_loss / max(steps, 1)

    def get_shared_weights(self):
        """
        Return ONLY adapter weights for aggregation.

        Original returned encoder + adapter (~66.5M params).
        Now returns adapter only (~24K params) — a 2700× communication reduction.
        The encoder is warm-started from pretrained weights and then frozen after
        the warm-up rounds, so it doesn't need to be aggregated.
        """
        return {
            "adapter." + k: v
            for k, v in self.model.adapter.state_dict().items()
        }

    def initialize_shared_weights(self, global_shared_state):
        """
        Load global adapter weights into local model.
        Does NOT rebuild the optimizer — momentum is preserved across rounds.
        Saves a snapshot of global params for FedProx.
        """
        adapter_state = {
            k.replace("adapter.", ""): v
            for k, v in global_shared_state.items()
            if k.startswith("adapter.")
        }
        self.model.adapter.load_state_dict(adapter_state)

        # Save snapshot for FedProx proximal term in next local epoch
        self._global_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def step_scheduler(self):
        """Step the LR scheduler — call once per round after training."""
        self.scheduler.step()

    def get_current_lr(self):
        return self.scheduler.get_last_lr()[0]

    def evaluate(self, test_texts, batch_size=16):
        self.model.eval()

        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=0.1
        )

        total_loss = 0
        steps = 0

        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                inputs, targets, mask = self._prepare_batch(batch_texts)

                if inputs is None:
                    continue

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                logits = self.model(inputs, mask)
                loss = criterion(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1)
                )

                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity

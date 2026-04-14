import torch
import torch.nn as nn
import torch.optim as optim
from models.base_model import FedVocModel
from torch.nn.utils.rnn import pad_sequence


class FedVocClient:
    """
    FedVoc client — lightweight version for ~30 min GPU training.

    Bug fixes kept (zero compute cost):
        - Optimizer NOT rebuilt every round — momentum preserved across rounds
        - Optimizer only rebuilt when encoder freeze state actually changes
        - get_shared_weights() returns adapter ONLY (not full encoder)
        - initialize_shared_weights() does not touch the optimizer

    Removed to save time:
        - FedProx proximal term (was cloning all params every batch — ~10% overhead)
        - Full dataset — 3000-sample cap RESTORED (4-5x faster per epoch)
        - Pretrained encoder — REMOVED (was the main 2-hour culprit)

    Kept from improved version:
        - Cosine LR scheduler (free)
        - set_encoder_frozen() only rebuilds optimizer on actual state change
        - Proper global init (handled in run_fedvoc.py)
        - Weighted aggregation (handled in run_fedvoc.py)
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

        # Build optimizer once — never rebuild unless freeze state actually changes
        self.optimizer = self._build_optimizer(lr=3e-4)

        # Cosine LR scheduler — free improvement, no compute cost
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-5
        )

    def _build_optimizer(self, lr=3e-4):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def set_encoder_frozen(self, frozen: bool):
        """
        BUG FIX: only rebuilds optimizer when freeze state actually changes.
        Original code rebuilt optimizer every single round — killed Adam momentum.
        """
        if frozen == self._encoder_frozen:
            return  # nothing changed — keep optimizer and momentum intact

        for param in self.model.encoder.parameters():
            param.requires_grad = not frozen
        self._encoder_frozen = frozen

        # Rebuild optimizer — momentum is stale after param set changes
        self.optimizer = self._build_optimizer(lr=self.scheduler.get_last_lr()[0])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-5
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

        padded = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_id
        )
        attention_mask = (padded != self.pad_id).long()
        inputs = padded[:, :-1]
        targets = padded[:, 1:]
        return inputs, targets, attention_mask[:, :-1]

    def train_one_epoch(self, batch_size=16):
        self.model.train()

        criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=0.1
        )

        total_loss = 0
        steps = 0

        # 3000-sample cap RESTORED — was 4-5x slower without it
        for i in range(0, min(len(self.texts), 3000), batch_size):
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def get_shared_weights(self):
        """
        BUG FIX: return adapter ONLY — not the full 66M-param encoder.
        Original returned encoder + adapter which made comm cost nearly
        identical to FedAvg. Now only ~49K params shared per round.
        """
        return {
            "adapter." + k: v
            for k, v in self.model.adapter.state_dict().items()
        }

    def initialize_shared_weights(self, global_shared_state):
        """
        BUG FIX: loads global adapter weights WITHOUT rebuilding the optimizer.
        Original always rebuilt Adam here — threw away all momentum every round.
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

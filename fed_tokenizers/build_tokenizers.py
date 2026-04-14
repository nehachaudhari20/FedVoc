"""
build_tokenizers.py — builds one domain-specific tokenizer per client.

Run with:
    python -m build_tokenizers

Fixes vs original:
    - vocab_size argument was passed to the function but the trainer
      always used the hardcoded 5000 — now correctly uses the argument
    - min_frequency=2 added — stops singleton noise tokens getting into vocab
    - show_progress=True so you can see it's not stuck
    - Prints actual saved vocab size for each client (useful for debugging)

Domain vocab sizes will differ:
    Shakespeare  — archaic words, thee/thou/hath etc.
    News         — modern proper nouns, political terms
    Medical      — clinical terminology, drug names, anatomy
    This divergence is exactly what FedVoc is designed to exploit.
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from utils.data_loader import load_domain_clients

VOCAB_SIZE = 5000


def build_tokenizer(texts, save_path, vocab_size=VOCAB_SIZE):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,                      # FIX: was hardcoded, ignored the arg
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
        min_frequency=2,                            # NEW: filter singleton tokens
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(save_path)
    return tokenizer.get_vocab_size()


if __name__ == "__main__":
    clients = load_domain_clients()
    os.makedirs("fed_tokenizers", exist_ok=True)

    for i, (client_id, data) in enumerate(clients.items()):
        texts = data["train"]
        path = f"fed_tokenizers/tokenizer_client_{i}.json"

        print(f"\nTraining tokenizer for {client_id} ({len(texts):,} texts)...")
        actual_vocab_size = build_tokenizer(texts, path)
        print(f"Saved to {path} — vocab size: {actual_vocab_size:,}")

    print("\nAll client tokenizers saved.")

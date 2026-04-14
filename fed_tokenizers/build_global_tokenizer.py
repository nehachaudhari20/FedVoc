"""
build_global_tokenizer.py — builds shared vocabulary for FedAvg baseline.

Run with:
    python -m build_global_tokenizer

Fix vs original:
    - vocab_size was hardcoded in trainer but not passed from the variable
      (trainer used 5000 regardless of the vocab_size argument)
    - Added post_processor so [CLS] and [SEP] are handled correctly
    - Prints vocab size confirmation after saving
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from utils.data_loader import load_domain_clients

VOCAB_SIZE = 5000

clients = load_domain_clients()

all_texts = []
for data in clients.values():
    all_texts.extend(data["train"])

print(f"Training global tokenizer on {len(all_texts):,} texts...")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,                          # FIX: was ignored in original
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
    min_frequency=2,                                # NEW: ignore tokens seen only once
    show_progress=True,
)

tokenizer.train_from_iterator(all_texts, trainer)

os.makedirs("fed_tokenizers", exist_ok=True)
tokenizer.save("fed_tokenizers/global_tokenizer.json")

print(f"Global tokenizer saved — vocab size: {tokenizer.get_vocab_size():,}")

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from utils.data_loader import load_domain_clients
from tokenizers.pre_tokenizers import ByteLevel

import os

clients = load_domain_clients()

all_texts = []
for data in clients.values():
    all_texts.extend(data["train"])

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
)

tokenizer.train_from_iterator(all_texts, trainer)

os.makedirs("fed_tokenizers", exist_ok=True)
tokenizer.save("fed_tokenizers/global_tokenizer.json")  # ✅ FIXED

print("Saved global tokenizer.")

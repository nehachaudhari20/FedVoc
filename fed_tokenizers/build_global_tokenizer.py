from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from utils.data_loader import load_shakespeare_clients
import os

clients = load_shakespeare_clients()

all_texts = []
for texts in clients.values():
    all_texts.extend(texts)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=3000, special_tokens=["[UNK]"])
tokenizer.train_from_iterator(all_texts, trainer)

os.makedirs("fed_tokenizers", exist_ok=True)
tokenizer.save("fed_tokenizers/global_tokenizer.json")

print("Saved global tokenizer.")

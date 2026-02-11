from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


def build_tokenizer(texts, save_path, vocab_size=3000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(texts, trainer)

    tokenizer.save(save_path)


if __name__ == "__main__":
    from utils.data_loader import load_shakespeare_clients

    clients = load_shakespeare_clients()

    os.makedirs("fed_tokenizers", exist_ok=True)

    for i, (client_id, texts) in enumerate(clients.items()):
        path = f"fed_tokenizers/tokenizer_client_{i}.json"
        build_tokenizer(texts, path)
        print(f"Saved tokenizer for client {client_id}")

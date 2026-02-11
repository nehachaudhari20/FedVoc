from datasets import load_dataset
from collections import defaultdict


def load_shakespeare_clients(num_clients=5, min_samples=100):
    ds = load_dataset("flwrlabs/shakespeare", split="train")

    client_dict = defaultdict(list)

    for example in ds:
        client_dict[example["character_id"]].append(example["x"])

    clients = {}
    count = 0

    for client_id, texts in client_dict.items():
        if len(texts) >= min_samples:
            clients[client_id] = texts
            count += 1

        if count >= num_clients:
            break

    return clients

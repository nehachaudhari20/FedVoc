from datasets import load_dataset
from collections import defaultdict
import random


def load_shakespeare_clients(num_clients=3, min_samples=100):

    ds = load_dataset("flwrlabs/shakespeare", split="train")

    client_dict = defaultdict(list)

    for example in ds:
        client_dict[example["character_id"]].append(example["x"])

    clients = {}
    count = 0

    for client_id, texts in client_dict.items():

        if len(texts) >= min_samples:

            random.shuffle(texts)

            split_idx = int(0.8 * len(texts))

            train_texts = texts[:split_idx]
            test_texts = texts[split_idx:]

            clients[client_id] = {
                "train": train_texts,
                "test": test_texts
            }

            count += 1

        if count >= num_clients:
            break

    return clients

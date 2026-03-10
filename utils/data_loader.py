from datasets import load_dataset
import random


def load_domain_clients():

    clients = {}

    # -------- Client 0 — Shakespeare --------
    shakespeare = load_dataset("flwrlabs/shakespeare", split="train")

    texts = [ex["x"] for ex in shakespeare if len(ex["x"]) > 20]

    random.shuffle(texts)

    split = int(0.8 * len(texts))

    clients["client_shakespeare"] = {
        "train": texts[:split],
        "test": texts[split:]
    }


    # -------- Client 1 — Reddit --------
    reddit = load_dataset("reddit", "plain_text", split="train[:5000]")

    texts = []

    for ex in reddit:

        body = ex["body"]

        if body is None:
            continue

        if body in ["[removed]", "[deleted]"]:
            continue

        if len(body) < 20:
            continue

        texts.append(body)

    random.shuffle(texts)

    split = int(0.8 * len(texts))

    clients["client_reddit"] = {
        "train": texts[:split],
        "test": texts[split:]
    }


    # -------- Client 2 — PubMed --------
    pubmed = load_dataset("pubmed", split="train[:5000]")

    texts = []

    for ex in pubmed:

        abstract = ex["abstract"]

        if abstract is None:
            continue

        if len(abstract) < 40:
            continue

        texts.append(abstract)

    random.shuffle(texts)

    split = int(0.8 * len(texts))

    clients["client_medical"] = {
        "train": texts[:split],
        "test": texts[split:]
    }

    return clients
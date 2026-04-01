from datasets import load_dataset
import random
import re

random.seed(42)


def clean_text(text):

    if text is None:
        return None

    text = text.strip().lower()

    # remove links only
    text = re.sub(r"http\S+", "", text)

    # KEEP numbers + hyphen words
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:\-']", " ", text)

    text = re.sub(r"\s+", " ", text)

    if len(text) < 20:
        return None

    return text


def process_dataset(dataset, field):

    texts = []

    for ex in dataset:

        t = clean_text(ex[field])

        if t:

            words = t.split()
            if len(words) < 5:
                continue
            if len(words) > 120:
                t = " ".join(words[:120])

            texts.append(t)

    random.shuffle(texts)

    split = int(0.8 * len(texts))

    return {
        "train": texts[:split],
        "test": texts[split:]
    }


def load_domain_clients():

    clients = {}

    # Shakespeare
    shakespeare = load_dataset(
        "flwrlabs/shakespeare",
        split="train[:12000]"
    )

    clients["client_shakespeare"] = process_dataset(
        shakespeare,
        "x"
    )

    # News
    news = load_dataset(
        "ag_news",
        split="train[:15000]"
    )

    clients["client_news"] = process_dataset(
        news,
        "text"
    )

    # Medical
    pubmed = load_dataset(
        "ccdv/pubmed-summarization",
        split="train[:15000]"
    )

    clients["client_medical"] = process_dataset(
        pubmed,
        "article"
    )

    return clients

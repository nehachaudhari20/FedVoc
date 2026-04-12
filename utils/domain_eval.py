"""
domain_eval.py — domain accuracy metric for FedVoc evaluation.

Why this matters:
    Perplexity across different tokenizers is NOT a fair comparison —
    FedVoc and FedAvg use different vocab sizes and token distributions.
    Domain accuracy measures what FedVoc actually claims: that it makes
    better in-domain predictions than a generic global model.

Domain accuracy: for each test sentence, what fraction of the model's
top-k predicted next tokens are domain-relevant words?
"""

import torch
import torch.nn.functional as F


def build_domain_vocab(tokenizer, domain_texts, min_freq=5):
    """
    Build a set of domain-specific tokens from training texts.
    Only includes tokens that appear at least min_freq times —
    filters out rare/noisy tokens.
    """
    from collections import Counter
    counter = Counter()

    for text in domain_texts:
        tokens = tokenizer.encode(text).tokens
        counter.update(tokens)

    vocab = {
        tok for tok, count in counter.items()
        if count >= min_freq
        and tok not in ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
        and len(tok.replace("Ġ", "").replace("##", "")) >= 3
    }
    return vocab


def domain_accuracy(client, test_texts, top_k=3, max_texts=200, device=None):
    """
    Compute domain accuracy: fraction of top-k predictions that are in-domain.

    Args:
        client:     a FedVocClient instance (has .model, .tokenizer, .texts)
        test_texts: list of test strings to evaluate on
        top_k:      how many top predictions to consider
        max_texts:  cap evaluation at this many texts (for speed)

    Returns:
        float in [0, 1] — higher is better (more domain-relevant predictions)
    """
    device = device or client.device
    client.model.eval()

    pad_id = client.tokenizer.token_to_id("[PAD]")
    id_to_token = {v: k for k, v in client.tokenizer.get_vocab().items()}

    # Build domain vocab from training texts
    domain_vocab = build_domain_vocab(client.tokenizer, client.texts)

    hits = 0
    total = 0

    with torch.no_grad():
        for text in test_texts[:max_texts]:
            ids = client.tokenizer.encode(text).ids[-79:]  # leave room for prediction
            if len(ids) < 2:
                continue

            input_tensor = torch.tensor([ids]).to(device)
            mask = (input_tensor != pad_id).long()

            logits = client.model(input_tensor, mask)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            topk_ids = torch.topk(probs, top_k).indices.tolist()

            for idx in topk_ids:
                token = id_to_token.get(idx, "[UNK]")
                # Normalize: strip BPE prefix, lowercase
                clean = token.replace("Ġ", "").replace("##", "").lower()
                if clean in {t.replace("Ġ", "").replace("##", "").lower() for t in domain_vocab}:
                    hits += 1

            total += top_k

    return hits / max(total, 1)


def compare_domain_accuracy(fedvoc_clients, fedavg_client, test_texts_per_domain,
                             top_k=3, max_texts=200):
    """
    Compare FedVoc vs FedAvg domain accuracy across all domains.
    Prints a formatted comparison table.

    Args:
        fedvoc_clients:        list of FedVocClient (one per domain)
        fedavg_client:         a FedAvgClient with the global model
        test_texts_per_domain: list of test text lists, one per domain
        top_k:                 top-k predictions to evaluate
        max_texts:             cap per domain
    """
    domain_names = ["Shakespeare", "News", "Medical"]

    print(f"\n{'Domain':<14} {'FedVoc':>8} {'FedAvg':>8} {'Delta':>8}")
    print("-" * 42)

    for i, (name, test_texts) in enumerate(zip(domain_names, test_texts_per_domain)):
        fedvoc_acc = domain_accuracy(fedvoc_clients[i], test_texts, top_k, max_texts)

        # For FedAvg: temporarily give it the same test texts
        fedavg_acc = domain_accuracy(fedavg_client, test_texts, top_k, max_texts)

        delta = fedvoc_acc - fedavg_acc
        sign = "+" if delta >= 0 else ""
        print(f"{name:<14} {fedvoc_acc:>8.3f} {fedavg_acc:>8.3f} {sign}{delta:>7.3f}")

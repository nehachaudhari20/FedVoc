import torch
import torch.nn.functional as F
from collections import Counter


def build_domain_vocab(tokenizer, domain_texts, min_freq=5):
    counter = Counter()
    for text in domain_texts:
        counter.update(tokenizer.encode(text).tokens)
    return {
        tok for tok, count in counter.items()
        if count >= min_freq
        and tok not in ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
        and len(tok.replace("Ġ", "").replace("##", "")) >= 3
    }


def domain_accuracy(client, test_texts, top_k=3, max_texts=200, device=None):
    """
    Fraction of top-k predictions that are in-domain words.
    Higher = better domain-aware predictions.
    """
    device = device or client.device
    client.model.eval()

    pad_id = client.tokenizer.token_to_id("[PAD]")
    id_to_token = {v: k for k, v in client.tokenizer.get_vocab().items()}
    domain_vocab = build_domain_vocab(client.tokenizer, client.texts)
    domain_vocab_clean = {
        t.replace("Ġ", "").replace("##", "").lower() for t in domain_vocab
    }

    hits = 0
    total = 0

    with torch.no_grad():
        for text in test_texts[:max_texts]:
            ids = client.tokenizer.encode(text).ids[-79:]
            if len(ids) < 2:
                continue

            input_tensor = torch.tensor([ids]).to(device)
            mask = (input_tensor != pad_id).long()
            logits = client.model(input_tensor, mask)
            probs = F.softmax(logits[0, -1], dim=-1)
            topk_ids = torch.topk(probs, top_k).indices.tolist()

            for idx in topk_ids:
                token = id_to_token.get(idx, "[UNK]")
                clean = token.replace("Ġ", "").replace("##", "").lower()
                if clean in domain_vocab_clean:
                    hits += 1
            total += top_k

    return hits / max(total, 1)

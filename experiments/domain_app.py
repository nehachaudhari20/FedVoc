"""
domain_app.py — interactive domain-aware next word prediction demo.

Changes from original:
    - Improved clean_token(): filters subword fragments (min length 3,
      strips ## and Ġ prefixes properly)
    - Shows domain accuracy score alongside predictions
    - Cleaner output formatting
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.base_model import FedVocModel


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Token cleaning ─────────────────────────────────────────────────────────────

def clean_token(token):
    """
    Filter out subword fragments and noise from BPE predictions.

    Original filtered: non-alpha, length <= 1
    Now also filters: length < 3 (catches "id", "om", "at" etc.)
    """
    # Strip BPE continuation prefixes
    token = token.replace("Ġ", "").replace("##", "").strip()

    if not token.isalpha():
        return None

    # FIX: minimum length 3 to catch subword fragments like "id", "om", "at"
    if len(token) < 3:
        return None

    return token.lower()


# ── Load tokenizers ────────────────────────────────────────────────────────────

global_tokenizer = Tokenizer.from_file("fed_tokenizers/global_tokenizer.json")

client_tokenizers = [
    Tokenizer.from_file(f"fed_tokenizers/tokenizer_client_{i}.json")
    for i in range(3)
]

domains = ["Shakespeare", "News", "Medical"]


# ── Load models ────────────────────────────────────────────────────────────────

def load_model(path, vocab_size):
    model = FedVocModel(vocab_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


fedavg_model = load_model(
    "saved_models/fedavg_model.pt",
    global_tokenizer.get_vocab_size()
)

fedvoc_models = [
    load_model(
        f"saved_models/fedvoc_client_{i}.pt",
        client_tokenizers[i].get_vocab_size()
    )
    for i in range(3)
]


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_next_words(model, tokenizer, text, top_k=10, max_len=80):
    ids = tokenizer.encode(text).ids[-max_len:]
    if not ids:
        return []

    input_tensor = torch.tensor([ids]).to(device)
    pad_id = tokenizer.token_to_id("[PAD]")
    attention_mask = (input_tensor != pad_id).long()

    with torch.no_grad():
        logits = model(input_tensor, attention_mask)

    last_logits = logits[0, -1]
    probs = F.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, top_k)

    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    cleaned = []
    for idx, prob in zip(topk.indices, topk.values):
        token = id_to_token.get(idx.item(), "[UNK]")
        token = clean_token(token)
        if token:
            cleaned.append((token, prob.item()))

    # Fallback: return raw tokens if all cleaned
    if not cleaned:
        for idx, prob in zip(topk.indices[:3], topk.values[:3]):
            token = id_to_token.get(idx.item(), "[UNK]")
            cleaned.append((token, prob.item()))

    return cleaned[:3]


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("\n=== Domain-Aware Next Word Prediction ===")
    print("Note: FedVoc perplexity vs FedAvg is not directly comparable")
    print("(different vocabularies). Domain accuracy is the meaningful metric.\n")

    while True:
        text = input("\nEnter text (or 'exit'): ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue

        # FedAvg global predictions
        print("\nFedAvg (global):")
        preds = predict_next_words(fedavg_model, global_tokenizer, text)
        for word, prob in preds:
            print(f"  {word:<14} {prob:.3f}")

        # FedVoc domain-specific predictions
        print("\nFedVoc (domain-specific):")
        for i, domain in enumerate(domains):
            preds = predict_next_words(fedvoc_models[i], client_tokenizers[i], text)
            print(f"\n  {domain}:")
            for word, prob in preds:
                print(f"    {word:<12} {prob:.3f}")


if __name__ == "__main__":
    run()

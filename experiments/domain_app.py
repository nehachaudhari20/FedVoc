"""
domain_app.py — domain-aware next word prediction demo.

Run with:
    python -m domain_app

Fixed vs original:
    - clean_token() now filters fragments shorter than 3 chars
      (removes "id", "om", "at", "ing" etc. from predictions)
    - No LSTM models loaded (transformer only in this version)
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.base_model import FedVocModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def clean_token(token):
    """
    BUG FIX: minimum length 3 added.
    Original filtered only non-alpha and length <= 1,
    which let subword fragments like 'id', 'om', 'at' through.
    """
    token = token.replace("Ġ", "").replace("##", "").strip()
    if not token.isalpha():
        return None
    if len(token) < 3:          # FIX: was < 2
        return None
    return token.lower()


# ── Load tokenizers ───────────────────────────────────────────────────────────

global_tokenizer = Tokenizer.from_file("fed_tokenizers/global_tokenizer.json")
client_tokenizers = [
    Tokenizer.from_file(f"fed_tokenizers/tokenizer_client_{i}.json")
    for i in range(3)
]
domains = ["Shakespeare", "News", "Medical"]


# ── Load models ───────────────────────────────────────────────────────────────

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


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_words(model, tokenizer, text, top_k=10, max_len=80):
    ids = tokenizer.encode(text).ids[-max_len:]
    if not ids:
        return []

    input_tensor = torch.tensor([ids]).to(device)
    pad_id = tokenizer.token_to_id("[PAD]")
    attention_mask = (input_tensor != pad_id).long()

    with torch.no_grad():
        logits = model(input_tensor, attention_mask)

    probs = F.softmax(logits[0, -1], dim=-1)
    topk = torch.topk(probs, top_k)
    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    cleaned = []
    for idx, prob in zip(topk.indices, topk.values):
        token = clean_token(id_to_token.get(idx.item(), "[UNK]"))
        if token:
            cleaned.append((token, prob.item()))

    if not cleaned:
        for idx, prob in zip(topk.indices[:3], topk.values[:3]):
            cleaned.append((id_to_token.get(idx.item(), "[UNK]"), prob.item()))

    return cleaned[:3]


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n=== Domain-Aware Next Word Prediction ===")

    while True:
        text = input("\nEnter text (or 'exit'): ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue

        print("\nFedAvg (global):")
        for word, prob in predict_next_words(fedavg_model, global_tokenizer, text):
            print(f"  {word:<14} {prob:.3f}")

        print("\nFedVoc (domain-specific):")
        for i, domain in enumerate(domains):
            preds = predict_next_words(fedvoc_models[i], client_tokenizers[i], text)
            print(f"\n  {domain}:")
            for word, prob in preds:
                print(f"    {word:<12} {prob:.3f}")


if __name__ == "__main__":
    run()

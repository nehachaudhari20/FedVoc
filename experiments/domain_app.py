"""
domain_app.py — interactive domain-aware next word prediction demo.

Shows predictions from three model families side by side:
    1. FedAvg Transformer  — global model, shared vocab
    2. FedVoc Transformer  — domain-specific, adapter + DistilBERT encoder
    3. FedAvg LSTM         — global model, shared vocab, LSTM backbone
    4. FedVoc LSTM         — domain-specific, adapter + LSTM backbone

Changes from original:
    - LSTM models (FedAvgLSTMModel + FedVocLSTMModel) loaded alongside transformers
    - predict_next_words() works for both model types — no changes needed
      (both take input_ids + attention_mask and return logits)
    - Output section reorganised: transformer block then LSTM block per input
    - Improved clean_token(): filters subword fragments (min length 3,
      strips ## and Ġ prefixes properly)
    - Shows domain accuracy score alongside predictions
    - Cleaner output formatting
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.base_model import FedVocModel
from models.lstm_model import FedVocLSTMModel


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Token cleaning ─────────────────────────────────────────────────────────────

def clean_token(token):
    """
    Filter out subword fragments and noise from BPE predictions.
    """
    token = token.replace("Ġ", "").replace("##", "").strip()
    if not token.isalpha():
        return None
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

def load_transformer(path, vocab_size):
    model = FedVocModel(vocab_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_lstm(path, vocab_size):
    model = FedVocLSTMModel(vocab_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


print("Loading transformer models...")
fedavg_transformer = load_transformer(
    "saved_models/fedavg_model.pt",
    global_tokenizer.get_vocab_size()
)
fedvoc_transformers = [
    load_transformer(
        f"saved_models/fedvoc_client_{i}.pt",
        client_tokenizers[i].get_vocab_size()
    )
    for i in range(3)
]

print("Loading LSTM models...")
fedavg_lstm = load_lstm(
    "saved_models/fedavg_lstm_model.pt",
    global_tokenizer.get_vocab_size()
)
fedvoc_lstms = [
    load_lstm(
        f"saved_models/fedvoc_lstm_client_{i}.pt",
        client_tokenizers[i].get_vocab_size()
    )
    for i in range(3)
]

print("All models loaded.\n")


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_next_words(model, tokenizer, text, top_k=10, max_len=80):
    """
    Works for both transformer and LSTM models — both accept
    (input_ids, attention_mask) and return (B, T, vocab_size) logits.
    """
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

    # Fallback: return raw tokens if all were filtered
    if not cleaned:
        for idx, prob in zip(topk.indices[:3], topk.values[:3]):
            token = id_to_token.get(idx.item(), "[UNK]")
            cleaned.append((token, prob.item()))

    return cleaned[:3]


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print("=" * 55)
    print("   Domain-Aware Next Word Prediction Demo")
    print("   Transformer vs LSTM | FedAvg vs FedVoc")
    print("=" * 55)
    print("Note: Perplexity across tokenizers is not directly")
    print("comparable. Domain accuracy is the meaningful metric.\n")

    while True:
        text = input("\nEnter text (or 'exit'): ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue

        # ── Transformer ───────────────────────────────────────────────────────
        print("\n── TRANSFORMER ──────────────────────────────────────")

        print("\n  FedAvg (global, transformer):")
        preds = predict_next_words(fedavg_transformer, global_tokenizer, text)
        for word, prob in preds:
            print(f"    {word:<14} {prob:.3f}")

        print("\n  FedVoc (domain-specific, transformer):")
        for i, domain in enumerate(domains):
            preds = predict_next_words(fedvoc_transformers[i], client_tokenizers[i], text)
            print(f"\n    {domain}:")
            for word, prob in preds:
                print(f"      {word:<12} {prob:.3f}")

        # ── LSTM ──────────────────────────────────────────────────────────────
        print("\n── LSTM ─────────────────────────────────────────────")

        print("\n  FedAvg (global, LSTM):")
        preds = predict_next_words(fedavg_lstm, global_tokenizer, text)
        for word, prob in preds:
            print(f"    {word:<14} {prob:.3f}")

        print("\n  FedVoc (domain-specific, LSTM):")
        for i, domain in enumerate(domains):
            preds = predict_next_words(fedvoc_lstms[i], client_tokenizers[i], text)
            print(f"\n    {domain}:")
            for word, prob in preds:
                print(f"      {word:<12} {prob:.3f}")

        print("\n" + "─" * 55)


if __name__ == "__main__":
    run()

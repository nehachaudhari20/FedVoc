"""
domain_app_full.py — next word prediction demo showing all 4 models.

Run with:
    python -m domain_app_full

Shows predictions from:
    1. FedAvg Transformer  — global model, DistilBERT encoder
    2. FedVoc Transformer  — domain-specific, DistilBERT encoder
    3. FedAvg LSTM         — global model, LSTM encoder
    4. FedVoc LSTM         — domain-specific, LSTM encoder

Requires all 4 training scripts to have been run first:
    python -m run_fedavg
    python -m run_fedvoc
    python -m run_fedavg_lstm
    python -m run_fedvoc_lstm
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.base_model import FedVocModel
from models.lstm_model import FedVocLSTMModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

LSTM_D_MODEL = 128
LSTM_LAYERS = 1


def clean_token(token):
    token = token.replace("Ġ", "").replace("##", "").strip()
    if not token.isalpha():
        return None
    if len(token) < 3:
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

def load_transformer(path, vocab_size):
    model = FedVocModel(vocab_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_lstm(path, vocab_size):
    model = FedVocLSTMModel(
        vocab_size=vocab_size,
        d_model=LSTM_D_MODEL,
        num_layers=LSTM_LAYERS,
    ).to(device)
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


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_transformer(model, tokenizer, text, top_k=10, max_len=80):
    ids = tokenizer.encode(text).ids[-max_len:]
    if not ids:
        return []
    input_tensor = torch.tensor([ids]).to(device)
    pad_id = tokenizer.token_to_id("[PAD]")
    mask = (input_tensor != pad_id).long()
    with torch.no_grad():
        logits = model(input_tensor, mask)
    return _topk_tokens(logits[0, -1], tokenizer, top_k)


def predict_lstm(model, tokenizer, text, top_k=10, max_len=80):
    ids = tokenizer.encode(text).ids[-max_len:]
    if not ids:
        return []
    input_tensor = torch.tensor([ids]).to(device)
    with torch.no_grad():
        logits, _ = model(input_tensor, hidden=None)
    return _topk_tokens(logits[0, -1], tokenizer, top_k)


def _topk_tokens(last_logits, tokenizer, top_k):
    probs = F.softmax(last_logits, dim=-1)
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
    print("=" * 58)
    print("   Domain-Aware Next Word Prediction")
    print("   Transformer vs LSTM  |  FedAvg vs FedVoc")
    print("=" * 58)

    while True:
        text = input("\nEnter text (or 'exit'): ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue

        # ── Transformer ───────────────────────────────────────────────────────
        print("\n── Transformer ──────────────────────────────────────────")

        print("\n  FedAvg (global):")
        for word, prob in predict_transformer(fedavg_transformer, global_tokenizer, text):
            print(f"    {word:<14} {prob:.3f}")

        print("\n  FedVoc (domain-specific):")
        for i, domain in enumerate(domains):
            preds = predict_transformer(fedvoc_transformers[i], client_tokenizers[i], text)
            print(f"\n    {domain}:")
            for word, prob in preds:
                print(f"      {word:<12} {prob:.3f}")

        # ── LSTM ──────────────────────────────────────────────────────────────
        print("\n── LSTM ─────────────────────────────────────────────────")

        print("\n  FedAvg (global):")
        for word, prob in predict_lstm(fedavg_lstm, global_tokenizer, text):
            print(f"    {word:<14} {prob:.3f}")

        print("\n  FedVoc (domain-specific):")
        for i, domain in enumerate(domains):
            preds = predict_lstm(fedvoc_lstms[i], client_tokenizers[i], text)
            print(f"\n    {domain}:")
            for word, prob in preds:
                print(f"      {word:<12} {prob:.3f}")

        print("\n" + "─" * 58)


if __name__ == "__main__":
    run()
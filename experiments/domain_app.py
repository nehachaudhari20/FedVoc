import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.base_model import FedVocModel


# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# -----------------------------
# CLEAN TOKEN
# -----------------------------
def clean_token(token):
    token = token.replace("Ġ", "").strip()

    # remove special junk
    if not token.isalpha():
        return None

    # remove extreme junk (single chars only)
    if len(token) == 1:
        return None

    return token.lower()


# -----------------------------
# LOAD TOKENIZERS
# -----------------------------
global_tokenizer = Tokenizer.from_file("fed_tokenizers/global_tokenizer.json")

client_tokenizers = [
    Tokenizer.from_file(f"fed_tokenizers/tokenizer_client_{i}.json")
    for i in range(3)
]

domains = ["Shakespeare", "News", "Medical"]


# -----------------------------
# LOAD MODELS
# -----------------------------
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


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_next_words(model, tokenizer, text, top_k=10, max_len=80):

    ids = tokenizer.encode(text).ids[-max_len:]

    if len(ids) == 0:
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

    if len(cleaned) == 0:
        fallback = []
        for idx, prob in zip(topk.indices, topk.values):
            token = id_to_token.get(idx.item(), "[UNK]")
            fallback.append((token, prob.item()))
        return fallback[:3]

    return cleaned[:3]


# -----------------------------
# MAIN APP
# -----------------------------
def run():

    print("\n=== Domain-Aware Next Word Prediction ===")

    while True:
        text = input("\nEnter text (or type 'exit'): ")

        if text.lower() == "exit":
            break

        # -------- FedAvg --------
        print("\n🔹 FedAvg (Global):")
        preds = predict_next_words(
            fedavg_model,
            global_tokenizer,
            text
        )

        for word, prob in preds:
            print(f"{word:<12} → {prob:.2f}")

        # -------- FedVoc --------
        print("\n🔹 FedVoc (Domain-specific):")

        for i, domain in enumerate(domains):
            preds = predict_next_words(
                fedvoc_models[i],
                client_tokenizers[i],
                text
            )

            print(f"\n{domain}:")
            for word, prob in preds:
                print(f"{word} ({prob:.3f})")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run()

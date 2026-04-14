"""
FedVoc LSTM training — lightweight version (~10-15 min on gaming GPU).

Run with:
    python -m run_fedvoc_lstm

Saves:
    saved_models/fedvoc_lstm_client_0.pt
    saved_models/fedvoc_lstm_client_1.pt
    saved_models/fedvoc_lstm_client_2.pt
    results/fedvoc_lstm_results.json
    results/fedvoc_lstm_convergence.png

Same bug fixes as transformer FedVoc:
    1. Global shared state = average of ALL clients, not just client 0
    2. Weighted FedAvg aggregation by dataset size
    3. Optimizer momentum preserved across rounds
    4. Adapter-only sharing (~8K params per round vs transformer's 50K)

Lightweight settings (same as transformer version):
    - 3000-sample cap per epoch
    - No FedProx
    - No pretrained weights
    - 15 rounds
    - d_model=128, layers=1 (~800K total params vs transformer's 66M)
"""

import json
import os

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from clients.client_fedvoc_lstm import FedVocLSTMClient
from utils.communication import count_parameters
from utils.data_loader import load_domain_clients
from utils.domain_eval import domain_accuracy
from utils.oov_eval import oov_rate

# ── Config ────────────────────────────────────────────────────────────────────

ROUNDS = 15
ENCODER_UNFREEZE_ROUND = 2
D_MODEL = 128
NUM_LAYERS = 1
DROPOUT = 0.3
RANK = 16

# ── Data ──────────────────────────────────────────────────────────────────────

clients_data = load_domain_clients()
clients = []

for i, (cid, data) in enumerate(clients_data.items()):
    tokenizer = Tokenizer.from_file(f"fed_tokenizers/tokenizer_client_{i}.json")
    client = FedVocLSTMClient(
        tokenizer=tokenizer,
        texts=data["train"],
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        rank=RANK,
    )
    client.test_texts = data["test"]
    clients.append(client)

# Dataset weights for aggregation — BUG FIX #2
dataset_sizes = [len(c.texts) for c in clients]
total_samples = sum(dataset_sizes)
client_weights = [s / total_samples for s in dataset_sizes]
print(f"Dataset sizes: {dataset_sizes}")
print(f"Client weights: {[round(w, 3) for w in client_weights]}")

# ── Global shared state init — BUG FIX #1 ────────────────────────────────────

all_init = [c.get_shared_weights() for c in clients]
global_shared = {}
for key in all_init[0]:
    global_shared[key] = sum(w[key] for w in all_init) / len(all_init)

# ── Training ──────────────────────────────────────────────────────────────────

round_losses = []
round_lrs = []

print(f"\nStarting FedVoc LSTM training — {ROUNDS} rounds, d_model={D_MODEL}, layers={NUM_LAYERS}")
print("Bug fixes: weighted agg + proper init + optimizer momentum + adapter-only sharing\n")

for round_idx in range(ROUNDS):
    print(f"--- Round {round_idx} ---")

    # BUG FIX #3: only rebuilds optimizer when freeze state actually changes
    for client in clients:
        client.set_encoder_frozen(round_idx < ENCODER_UNFREEZE_ROUND)
        client.initialize_shared_weights(global_shared)

    shared_updates = []
    total_round_loss = 0

    for i, client in enumerate(clients):
        loss = client.train_one_epoch()
        print(f"  Client {i} loss: {loss:.4f}  lr: {client.get_current_lr():.2e}")
        total_round_loss += loss
        shared_updates.append(client.get_shared_weights())

    # BUG FIX #2: weighted aggregation
    new_shared = {}
    for key in global_shared:
        new_shared[key] = sum(
            w * upd[key]
            for w, upd in zip(client_weights, shared_updates)
        )
    global_shared = new_shared

    avg_loss = total_round_loss / len(clients)
    round_losses.append(avg_loss)
    print(f"  Avg loss: {avg_loss:.4f}\n")

    for client in clients:
        client.step_scheduler()
    round_lrs.append(clients[0].get_current_lr())

print("FedVoc LSTM training complete.")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\nEvaluating FedVoc LSTM...")
eval_results = []

for i, client in enumerate(clients):
    loss, ppl = client.evaluate(client.test_texts)
    print(f"  Client {i} — Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")
    eval_results.append({"client": i, "loss": loss, "perplexity": ppl})

print("\nDomain accuracy (top-3):")
domain_accs = []
for i, client in enumerate(clients):
    acc = domain_accuracy(client, client.test_texts[:200])
    print(f"  Client {i}: {acc:.3f}")
    domain_accs.append(acc)

print("\nOOV rate on test sentences:")
oov_rates = []
for i, client in enumerate(clients):
    rate = oov_rate(client.tokenizer, client.test_texts[:500])
    print(f"  Client {i}: {rate:.4f}")
    oov_rates.append(rate)

comm_cost = count_parameters(global_shared)
print(f"\nFedVoc LSTM communication cost per round: {comm_cost:,} params")
print(f"FedAvg LSTM shares full model — FedVoc LSTM is much cheaper")

# ── Save results ──────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)
with open("results/fedvoc_lstm_results.json", "w") as f:
    json.dump({
        "round_losses": round_losses,
        "round_lrs":    round_lrs,
        "eval":         eval_results,
        "domain_acc":   domain_accs,
        "oov_rates":    oov_rates,
        "comm_cost":    comm_cost,
    }, f, indent=2)
print("Results saved to results/fedvoc_lstm_results.json")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(round_losses, marker="o", markersize=4)
ax1.set_title("FedVoc LSTM — convergence")
ax1.set_xlabel("Round")
ax1.set_ylabel("Avg loss")
ax1.grid(alpha=0.3)

ax2.plot(round_lrs)
ax2.set_title("LR schedule (cosine)")
ax2.set_xlabel("Round")
ax2.set_ylabel("LR")
ax2.set_yscale("log")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/fedvoc_lstm_convergence.png", dpi=120)
plt.close()
print("Convergence plot saved to results/fedvoc_lstm_convergence.png")

# ── Save models ───────────────────────────────────────────────────────────────

os.makedirs("saved_models", exist_ok=True)
for i, client in enumerate(clients):
    torch.save(
        client.model.state_dict(),
        f"saved_models/fedvoc_lstm_client_{i}.pt"
    )
print("FedVoc LSTM client models saved.")

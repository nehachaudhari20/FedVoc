"""
run_fedvoc_lstm.py — FedVoc training with LSTM language model.

Mirrors run_fedvoc.py exactly, with two differences:
  1. Uses FedVocLSTMClient instead of FedVocClient
  2. No pretrained encoder — LSTM trains from scratch

Shared weights per round: lstm + adapter (~8.4M params)
  Compare to transformer FedVoc: adapter only (~24K params)
  Compare to transformer FedAvg: full model (~74M params)

Run order (suggested):
  python run_fedavg.py
  python run_fedvoc.py
  python run_fedavg_lstm.py
  python run_fedvoc_lstm.py
  python utils/compare_results.py

Output:
  results/fedvoc_lstm_convergence.png
  results/fedvoc_lstm_results.json     ← consumed by compare_results.py
  saved_models/fedvoc_lstm_client_{i}.pt
"""

import json
import os

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from clients.client_fedvoc_lstm import FedVocLSTMClient
from utils.communication import count_parameters
from utils.data_loader import load_domain_clients
from utils.oov_eval import oov_rate
from utils.domain_eval import domain_accuracy


# ── Data ──────────────────────────────────────────────────────────────────────

clients_data = load_domain_clients()
clients      = []

for i, (cid, data) in enumerate(clients_data.items()):
    tokenizer = Tokenizer.from_file(f"fed_tokenizers/tokenizer_client_{i}.json")
    client    = FedVocLSTMClient(tokenizer, data["train"])
    client.test_texts = data["test"]
    clients.append(client)

# ── Global shared state initialisation ────────────────────────────────────────
# Average ALL clients' initial lstm + adapter weights (same fix as run_fedvoc.py).

all_init     = [c.get_shared_weights() for c in clients]
global_shared = {}
for key in all_init[0]:
    global_shared[key] = sum(w[key] for w in all_init) / len(all_init)

# ── Training loop ─────────────────────────────────────────────────────────────

ROUNDS               = 20
LSTM_UNFREEZE_ROUND  = 2     # freeze lstm for first N rounds → train adapter only
FEDPROX_MU           = 0.01

dataset_sizes  = [len(c.texts) for c in clients]
total_samples  = sum(dataset_sizes)
client_weights = [s / total_samples for s in dataset_sizes]

round_losses = []
round_lrs    = []

print("Starting FedVoc LSTM training (lstm + adapter sharing, no pretrained weights)...")

for round_idx in range(ROUNDS):
    print(f"\n--- Round {round_idx} ---")

    for client in clients:
        should_freeze = (round_idx < LSTM_UNFREEZE_ROUND)
        client.set_encoder_frozen(should_freeze)   # same API as FedVocClient
        client.initialize_shared_weights(global_shared)

    shared_updates   = []
    total_round_loss = 0

    for client in clients:
        loss = client.train_one_epoch(batch_size=16, mu=FEDPROX_MU)
        print(f"  Client loss: {loss:.4f}  lr: {client.get_current_lr():.2e}")

        total_round_loss += loss
        shared_updates.append(client.get_shared_weights())

    # Weighted aggregation
    new_shared = {}
    for key in global_shared:
        new_shared[key] = sum(
            w * update[key]
            for w, update in zip(client_weights, shared_updates)
        )
    global_shared = new_shared

    round_losses.append(total_round_loss / len(clients))

    for client in clients:
        client.step_scheduler()
    round_lrs.append(clients[0].get_current_lr())

print("\nFedVoc LSTM training complete.")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\nEvaluating FedVoc LSTM...")
eval_results = []

for i, client in enumerate(clients):
    loss, ppl = client.evaluate(client.test_texts)
    print(f"  Client {i} — Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")
    eval_results.append({"client": i, "loss": loss, "perplexity": ppl})

# Domain accuracy
print("\nDomain accuracy (top-3):")
domain_accs = []
for i, client in enumerate(clients):
    acc = domain_accuracy(client, client.test_texts[:200])
    print(f"  Client {i}: {acc:.3f}")
    domain_accs.append(acc)

# OOV rate
print("\nOOV Rate on test sentences:")
oov_rates = []
for i, client in enumerate(clients):
    rate = oov_rate(client.tokenizer, client.test_texts[:500])
    print(f"  Client {i}: {rate:.4f}")
    oov_rates.append(rate)

# Communication cost
comm_cost = count_parameters(global_shared)
print(f"\nFedVoc LSTM communication cost per round: {comm_cost:,} params")

# ── Save results for compare_results.py ───────────────────────────────────────

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

# ── Plots ─────────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(round_losses)
ax1.set_title("FedVoc LSTM convergence")
ax1.set_xlabel("Round")
ax1.set_ylabel("Avg loss")

ax2.plot(round_lrs)
ax2.set_title("Learning rate schedule")
ax2.set_xlabel("Round")
ax2.set_ylabel("LR")
ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("results/fedvoc_lstm_convergence.png")
plt.close()
print("Convergence plot saved to results/fedvoc_lstm_convergence.png")

# ── Save models ───────────────────────────────────────────────────────────────

os.makedirs("saved_models", exist_ok=True)
for i, client in enumerate(clients):
    torch.save(client.model.state_dict(), f"saved_models/fedvoc_lstm_client_{i}.pt")
print("FedVoc LSTM client models saved.")

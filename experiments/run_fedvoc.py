"""
FedVoc training — lightweight version (~20-25 min on gaming GPU).

Run with:
    python -m run_fedvoc

What was removed vs heavy version to hit the 30-min target:
    - Pretrained DistilBERT loading        → single biggest time saving (~60% faster)
    - Full dataset per epoch               → 3000-sample cap restored (4-5x faster)
    - FedProx proximal term                → removed (~10% overhead per batch)
    - Rounds 20 → 15                       → 25% fewer rounds

Bug fixes kept (zero compute cost — all logic changes):
    1. Global shared state = average of ALL clients, not just client 0
    2. Weighted FedAvg aggregation by dataset size
    3. Optimizer momentum preserved — only rebuilt on actual freeze state change
    4. Adapter-only sharing (not full 66M-param encoder)

Free improvements kept:
    5. Cosine LR scheduler
    6. LayerNorm + zero-init B in adapter
    7. Domain accuracy + OOV rate evaluation
"""

import json
import os

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from clients.client_fedvoc import FedVocClient
from utils.communication import count_parameters
from utils.data_loader import load_domain_clients
from utils.domain_eval import domain_accuracy
from utils.oov_eval import oov_rate

# ── Config ────────────────────────────────────────────────────────────────────

ROUNDS = 15                  # reduced from 20 — saves 25% time, still converges
ENCODER_UNFREEZE_ROUND = 2   # unfreeze encoder after round 2 (same as before)

# ── Data ──────────────────────────────────────────────────────────────────────

clients_data = load_domain_clients()
clients = []

for i, (cid, data) in enumerate(clients_data.items()):
    tokenizer = Tokenizer.from_file(f"fed_tokenizers/tokenizer_client_{i}.json")
    client = FedVocClient(tokenizer, data["train"])
    client.test_texts = data["test"]
    clients.append(client)

# Dataset weights for aggregation — BUG FIX #2
dataset_sizes = [len(c.texts) for c in clients]
total_samples = sum(dataset_sizes)
client_weights = [s / total_samples for s in dataset_sizes]
print(f"Dataset sizes: {dataset_sizes}")
print(f"Client weights: {[round(w, 3) for w in client_weights]}")

# ── Global shared state init — BUG FIX #1 ────────────────────────────────────
# Original: global_shared = clients[0].get_shared_weights()  ← wrong, biases client 0
# Fixed:    average of ALL clients' initial adapter weights

all_init = [c.get_shared_weights() for c in clients]
global_shared = {}
for key in all_init[0]:
    global_shared[key] = sum(w[key] for w in all_init) / len(all_init)

# ── Training ──────────────────────────────────────────────────────────────────

round_losses = []
round_lrs = []

print(f"\nStarting FedVoc training — {ROUNDS} rounds, 3000-sample cap, no pretrained encoder")
print("Bug fixes: weighted agg + proper init + optimizer momentum + adapter-only sharing\n")

for round_idx in range(ROUNDS):
    print(f"--- Round {round_idx} ---")

    # BUG FIX #3: set_encoder_frozen() only rebuilds optimizer when state changes
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

    # BUG FIX #2: weighted aggregation by dataset size
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

print("FedVoc training complete.")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\nEvaluating FedVoc...")
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
print(f"\nFedVoc communication cost per round: {comm_cost:,} params")
print(f"FedAvg shares ~74,146,184 params — FedVoc is {74146184 // comm_cost}× cheaper")

# ── Save results ──────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)
with open("results/fedvoc_results.json", "w") as f:
    json.dump({
        "round_losses": round_losses,
        "round_lrs":    round_lrs,
        "eval":         eval_results,
        "domain_acc":   domain_accs,
        "oov_rates":    oov_rates,
        "comm_cost":    comm_cost,
    }, f, indent=2)
print("Results saved to results/fedvoc_results.json")

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(round_losses, marker="o", markersize=4)
ax1.set_title("FedVoc convergence")
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
plt.savefig("results/fedvoc_convergence.png", dpi=120)
plt.close()
print("Convergence plot saved to results/fedvoc_convergence.png")

# ── Save models ───────────────────────────────────────────────────────────────

os.makedirs("saved_models", exist_ok=True)
for i, client in enumerate(clients):
    torch.save(client.model.state_dict(), f"saved_models/fedvoc_client_{i}.pt")
print("FedVoc client models saved.")

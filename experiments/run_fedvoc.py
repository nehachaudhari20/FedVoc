"""
run_fedvoc.py — improved FedVoc training script

Bug fixes vs original:
    1. Global shared state initialised by averaging ALL clients, not just client 0
    2. Weighted FedAvg aggregation by dataset size
    3. Optimizer momentum preserved — only rebuilt when freeze state changes
    4. Only adapter weights are shared (not full encoder)

Training improvements:
    5. Pretrained DistilBERT encoder warm-start
    6. Cosine LR scheduler stepped each round
    7. Full dataset used per epoch (removed 3000-sample cap)
    8. FedProx proximal term (mu=0.01)

Evaluation improvements:
    9. Domain accuracy metric alongside perplexity
   10. Fixed OOV rate (rate on test sentences, not type count)
"""

import copy
import os

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from clients.client_fedvoc import FedVocClient
from utils.communication import count_parameters
from utils.data_loader import load_domain_clients
from utils.oov_eval import oov_rate
from utils.domain_eval import domain_accuracy


# ── Data ──────────────────────────────────────────────────────────────────────

clients_data = load_domain_clients()
clients = []

for i, (cid, data) in enumerate(clients_data.items()):
    tokenizer = Tokenizer.from_file(f"fed_tokenizers/tokenizer_client_{i}.json")
    client = FedVocClient(tokenizer, data["train"])
    client.test_texts = data["test"]
    clients.append(client)

# ── Pretrained encoder warm-start ─────────────────────────────────────────────
# Load pretrained DistilBERT once, copy weights into every client's encoder.
# The encoder already knows language — adapter only needs to learn alignment.

print("Loading pretrained DistilBERT encoder into all clients...")
clients[0].model.load_pretrained_encoder()
pretrained_encoder_state = clients[0].model.encoder.state_dict()
for client in clients[1:]:
    client.model.encoder.load_state_dict(pretrained_encoder_state)
print("Pretrained encoder loaded.\n")

# ── Global shared state initialisation ────────────────────────────────────────
# FIX 1: average ALL clients' initial adapter weights, not just client 0.
# Original: global_shared = clients[0].get_shared_weights()  ← wrong

all_init = [c.get_shared_weights() for c in clients]
global_shared = {}
for key in all_init[0]:
    global_shared[key] = sum(w[key] for w in all_init) / len(all_init)

# ── Training loop ─────────────────────────────────────────────────────────────

ROUNDS = 20
ENCODER_UNFREEZE_ROUND = 2   # unfreeze encoder after this many rounds
FEDPROX_MU = 0.01            # proximal coefficient (0 = disabled)

# Dataset sizes for weighted aggregation (FIX 2)
dataset_sizes = [len(c.texts) for c in clients]
total_samples = sum(dataset_sizes)
client_weights = [s / total_samples for s in dataset_sizes]

round_losses = []
round_lrs = []

print("Starting FedVoc training (adapter-only sharing, pretrained encoder)...")

for round_idx in range(ROUNDS):
    print(f"\n--- Round {round_idx} ---")

    # FIX 3: toggle freeze only when it actually changes — optimizer preserved otherwise
    for client in clients:
        should_freeze = (round_idx < ENCODER_UNFREEZE_ROUND)
        client.set_encoder_frozen(should_freeze)
        client.initialize_shared_weights(global_shared)

    shared_updates = []
    total_round_loss = 0

    for client in clients:
        loss = client.train_one_epoch(batch_size=16, mu=FEDPROX_MU)
        print(f"  Client loss: {loss:.4f}  lr: {client.get_current_lr():.2e}")

        total_round_loss += loss
        shared_updates.append(client.get_shared_weights())

    # FIX 2: weighted aggregation by dataset size
    new_shared = {}
    for key in global_shared:
        new_shared[key] = sum(
            w * update[key]
            for w, update in zip(client_weights, shared_updates)
        )
    global_shared = new_shared

    avg_loss = total_round_loss / len(clients)
    round_losses.append(avg_loss)

    # Step LR scheduler for every client (improvement 6)
    for client in clients:
        client.step_scheduler()
    round_lrs.append(clients[0].get_current_lr())

print("\nFedVoc training complete.")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\nEvaluating FedVoc...")
for i, client in enumerate(clients):
    loss, ppl = client.evaluate(client.test_texts)
    print(f"  Client {i} — Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")

# Domain accuracy (new metric — measures in-domain prediction quality)
print("\nDomain accuracy (top-3):")
for i, client in enumerate(clients):
    acc = domain_accuracy(client, client.test_texts[:200])
    print(f"  Client {i}: {acc:.3f}")

# OOV rate (fixed — rate on test sentences, not type count)
print("\nOOV Rate on test sentences:")
for i, client in enumerate(clients):
    rate = oov_rate(client.tokenizer, client.test_texts[:500])
    print(f"  Client {i}: {rate:.4f}")

# Communication cost — adapter only
comm_cost = count_parameters(global_shared)
print(f"\nFedVoc communication cost per round: {comm_cost:,} params")
print(f"(FedAvg baseline was ~74,146,184 params — {74146184 // comm_cost}× reduction)")

# ── Plots ─────────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(round_losses)
ax1.set_title("FedVoc convergence")
ax1.set_xlabel("Round")
ax1.set_ylabel("Avg loss")

ax2.plot(round_lrs)
ax2.set_title("Learning rate schedule")
ax2.set_xlabel("Round")
ax2.set_ylabel("LR")
ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("results/fedvoc_convergence.png")
plt.close()
print("\nConvergence plot saved to results/fedvoc_convergence.png")

# ── Save models ───────────────────────────────────────────────────────────────

os.makedirs("saved_models", exist_ok=True)
for i, client in enumerate(clients):
    torch.save(client.model.state_dict(), f"saved_models/fedvoc_client_{i}.pt")

print("FedVoc client models saved.")
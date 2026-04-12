"""
run_fedavg.py — improved FedAvg baseline

Changes vs original:
    1. Pretrained DistilBERT encoder loaded for fair comparison with FedVoc
    2. Full dataset used per epoch (removed 3000-sample cap)
    3. Cosine LR scheduler stepped each round
    4. Weighted aggregation by dataset size (same as FedVoc)
"""

import os

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from clients.client_fedavg import FedAvgClient
from server.server_base import Server
from utils.communication import count_parameters
from utils.data_loader import load_domain_clients


# ── Data ──────────────────────────────────────────────────────────────────────

clients_data = load_domain_clients()
tokenizer = Tokenizer.from_file("fed_tokenizers/global_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

server = Server(vocab_size)
clients = []

for cid, data in clients_data.items():
    client = FedAvgClient(tokenizer, data["train"])
    client.test_texts = data["test"]
    clients.append(client)

# Dataset sizes for weighted aggregation
dataset_sizes = [len(c.texts) for c in clients]
total_samples = sum(dataset_sizes)
client_weights = [s / total_samples for s in dataset_sizes]

# ── Pretrained encoder warm-start ─────────────────────────────────────────────

print("Loading pretrained DistilBERT into FedAvg server model...")
server.global_model.load_pretrained_encoder()
print("Pretrained encoder loaded.\n")

# ── Training loop ─────────────────────────────────────────────────────────────

ROUNDS = 20
round_losses = []

print("Starting FedAvg training...")

for round_idx in range(ROUNDS):
    print(f"\n--- Round {round_idx} ---")

    client_weight_list = []
    total_round_loss = 0

    for client in clients:
        client.initialize_local_model(server.global_model)
        loss = client.train_one_epoch()
        print(f"  Client train loss: {loss:.4f}  lr: {client.scheduler.get_last_lr()[0]:.2e}")

        total_round_loss += loss
        client_weight_list.append(client.get_model_weights())

    # Weighted aggregation
    server.aggregate(client_weight_list, client_weights)

    round_losses.append(total_round_loss / len(clients))

    for client in clients:
        client.step_scheduler()

print("\nFedAvg training complete.")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\nEvaluating FedAvg...")
for i, client in enumerate(clients):
    client.initialize_local_model(server.global_model)
    loss, ppl = client.evaluate(client.test_texts)
    print(f"  Client {i} — Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")

comm_cost = count_parameters(server.global_model.state_dict())
print(f"\nFedAvg communication cost per round: {comm_cost:,} params")

# ── Plots ─────────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)
plt.plot(round_losses)
plt.title("FedAvg convergence")
plt.xlabel("Round")
plt.ylabel("Avg loss")
plt.savefig("results/fedavg_convergence.png")
plt.close()
print("Convergence plot saved to results/fedavg_convergence.png")

# ── Save models ───────────────────────────────────────────────────────────────

os.makedirs("saved_models", exist_ok=True)
torch.save(server.global_model.state_dict(), "saved_models/fedavg_model.pt")
print("FedAvg model saved.")

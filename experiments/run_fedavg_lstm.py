"""
FedAvg LSTM baseline training — lightweight version (~10-15 min on gaming GPU).

Run with:
    python -m run_fedavg_lstm

Saves:
    saved_models/fedavg_lstm_model.pt
    results/fedavg_lstm_results.json
    results/fedavg_lstm_convergence.png

Same improvements as transformer FedAvg:
    - Weighted aggregation by dataset size
    - Cosine LR scheduler
    - 3000-sample cap
    - 15 rounds
"""

import json
import os

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from clients.client_fedavg_lstm import FedAvgLSTMClient
from server.server_lstm import LSTMServer
from utils.communication import count_parameters
from utils.data_loader import load_domain_clients

# ── Config ────────────────────────────────────────────────────────────────────

ROUNDS = 15
D_MODEL = 128
NUM_LAYERS = 1
DROPOUT = 0.3
RANK = 16

# ── Data ──────────────────────────────────────────────────────────────────────

clients_data = load_domain_clients()
tokenizer = Tokenizer.from_file("fed_tokenizers/global_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

server = LSTMServer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    rank=RANK,
)

clients = []
for cid, data in clients_data.items():
    client = FedAvgLSTMClient(
        tokenizer=tokenizer,
        texts=data["train"],
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        rank=RANK,
    )
    client.test_texts = data["test"]
    clients.append(client)

# Dataset weights for weighted aggregation
dataset_sizes = [len(c.texts) for c in clients]
total_samples = sum(dataset_sizes)
client_weights = [s / total_samples for s in dataset_sizes]
print(f"Dataset sizes: {dataset_sizes}")
print(f"Client weights: {[round(w, 3) for w in client_weights]}")

# ── Training ──────────────────────────────────────────────────────────────────

round_losses = []

print(f"\nStarting FedAvg LSTM training — {ROUNDS} rounds, d_model={D_MODEL}, layers={NUM_LAYERS}\n")

for round_idx in range(ROUNDS):
    print(f"--- Round {round_idx} ---")

    client_weight_list = []
    total_round_loss = 0

    for i, client in enumerate(clients):
        client.initialize_local_model(server.global_model)
        loss = client.train_one_epoch()
        print(f"  Client {i} train loss: {loss:.4f}  lr: {client.scheduler.get_last_lr()[0]:.2e}")
        total_round_loss += loss
        client_weight_list.append(client.get_model_weights())

    server.aggregate(client_weight_list, client_weights)

    avg_loss = total_round_loss / len(clients)
    round_losses.append(avg_loss)
    print(f"  Avg loss: {avg_loss:.4f}\n")

    for client in clients:
        client.step_scheduler()

print("FedAvg LSTM training complete.")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\nEvaluating FedAvg LSTM...")
eval_results = []

for i, client in enumerate(clients):
    client.initialize_local_model(server.global_model)
    loss, ppl = client.evaluate(client.test_texts)
    print(f"  Client {i} — Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")
    eval_results.append({"client": i, "loss": loss, "perplexity": ppl})

comm_cost = count_parameters(server.global_model.state_dict())
print(f"\nFedAvg LSTM communication cost per round: {comm_cost:,} params")

# ── Save results ──────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)
with open("results/fedavg_lstm_results.json", "w") as f:
    json.dump({
        "round_losses": round_losses,
        "eval":         eval_results,
        "comm_cost":    comm_cost,
    }, f, indent=2)
print("Results saved to results/fedavg_lstm_results.json")

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.plot(round_losses, marker="o", markersize=4)
plt.title("FedAvg LSTM — convergence")
plt.xlabel("Round")
plt.ylabel("Avg loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fedavg_lstm_convergence.png", dpi=120)
plt.close()
print("Convergence plot saved to results/fedavg_lstm_convergence.png")

# ── Save model ────────────────────────────────────────────────────────────────

os.makedirs("saved_models", exist_ok=True)
torch.save(server.global_model.state_dict(), "saved_models/fedavg_lstm_model.pt")
print("FedAvg LSTM model saved.")
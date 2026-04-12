"""
run_fedavg_lstm.py — FedAvg baseline with LSTM language model.

Mirrors run_fedavg.py exactly, with two differences:
  1. Uses FedAvgLSTMClient instead of FedAvgClient
  2. No pretrained encoder warm-start (LSTM trains from scratch)

Run AFTER run_fedavg.py and run_fedvoc.py so compare_results.py
can load all four result JSON files for a side-by-side comparison.

Output:
  results/fedavg_lstm_convergence.png
  results/fedavg_lstm_results.json     ← consumed by compare_results.py
  saved_models/fedavg_lstm_model.pt
"""

import json
import os

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from clients.client_fedavg_lstm import FedAvgLSTMClient
from server.server_base import Server
from utils.communication import count_parameters
from utils.data_loader import load_domain_clients


# ── Data ──────────────────────────────────────────────────────────────────────

clients_data = load_domain_clients()
tokenizer    = Tokenizer.from_file("fed_tokenizers/global_tokenizer.json")
vocab_size   = tokenizer.get_vocab_size()

server  = Server(vocab_size)          # Server still uses FedVocModel internally
clients = []                          # but we only use it for aggregation shape

# We need a lightweight LSTM server model for aggregation.
# Reuse Server but override global_model with an LSTM instance.
from models.lstm_model import FedVocLSTMModel

class LSTMServer:
    """Minimal server that holds an LSTM global model and does weighted FedAvg."""

    def __init__(self, vocab_size):
        import copy
        self.global_model = FedVocLSTMModel(vocab_size)
        self._vocab_size  = vocab_size

    def aggregate(self, client_weights_list, dataset_weights=None):
        import copy
        n = len(client_weights_list)
        if dataset_weights is None:
            dataset_weights = [1.0 / n] * n

        new_state = copy.deepcopy(self.global_model.state_dict())
        for key in new_state.keys():
            new_state[key] = sum(
                w * weights[key]
                for w, weights in zip(dataset_weights, client_weights_list)
            )
        self.global_model.load_state_dict(new_state)

server = LSTMServer(vocab_size)

for cid, data in clients_data.items():
    client = FedAvgLSTMClient(tokenizer, data["train"])
    client.test_texts = data["test"]
    clients.append(client)

dataset_sizes  = [len(c.texts) for c in clients]
total_samples  = sum(dataset_sizes)
client_weights = [s / total_samples for s in dataset_sizes]

# ── Training loop ─────────────────────────────────────────────────────────────

ROUNDS      = 20
round_losses = []

print("Starting FedAvg LSTM training...")

for round_idx in range(ROUNDS):
    print(f"\n--- Round {round_idx} ---")

    client_weight_list = []
    total_round_loss   = 0

    for client in clients:
        client.initialize_local_model(server.global_model)
        loss = client.train_one_epoch()
        print(f"  Client train loss: {loss:.4f}  lr: {client.scheduler.get_last_lr()[0]:.2e}")

        total_round_loss += loss
        client_weight_list.append(client.get_model_weights())

    server.aggregate(client_weight_list, client_weights)
    round_losses.append(total_round_loss / len(clients))

    for client in clients:
        client.step_scheduler()

print("\nFedAvg LSTM training complete.")

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

# ── Save results for compare_results.py ───────────────────────────────────────

os.makedirs("results", exist_ok=True)
with open("results/fedavg_lstm_results.json", "w") as f:
    json.dump({
        "round_losses": round_losses,
        "eval":         eval_results,
        "comm_cost":    comm_cost,
    }, f, indent=2)

# ── Plots ─────────────────────────────────────────────────────────────────────

plt.plot(round_losses)
plt.title("FedAvg LSTM convergence")
plt.xlabel("Round")
plt.ylabel("Avg loss")
plt.savefig("results/fedavg_lstm_convergence.png")
plt.close()
print("Convergence plot saved to results/fedavg_lstm_convergence.png")

# ── Save model ────────────────────────────────────────────────────────────────

os.makedirs("saved_models", exist_ok=True)
torch.save(server.global_model.state_dict(), "saved_models/fedavg_lstm_model.pt")
print("FedAvg LSTM model saved.")

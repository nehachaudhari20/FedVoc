from utils.communication import count_parameters
from tokenizers import Tokenizer
from clients.client_fedavg import FedAvgClient
from server.server_base import Server
import matplotlib.pyplot as plt

from utils.data_loader import load_domain_clients

clients_data = load_domain_clients()
tokenizer = Tokenizer.from_file(
    "fed_tokenizers/global_tokenizer.json"
)

vocab_size = tokenizer.get_vocab_size()

server = Server(vocab_size)

clients = []

for cid, data in clients_data.items():
    client = FedAvgClient(tokenizer, data["train"])
    client.test_texts = data["test"]
    clients.append(client)

round_losses = []

print("Starting TRUE FedAvg baseline with DistilBERT...")

for round in range(12):
    print(f"\n--- Round {round} ---")

    client_weights = []
    total_round_loss = 0

    for client in clients:
        client.initialize_local_model(server.global_model)

        loss = client.train_one_epoch()
        print("Client train loss:", loss)

        total_round_loss += loss
        client_weights.append(client.get_model_weights())

    avg_loss = total_round_loss / len(clients)
    round_losses.append(avg_loss)

    server.aggregate(client_weights)

print("\nEvaluating FedAvg baseline...")

for i, client in enumerate(clients):
    loss, ppl = client.evaluate(client.test_texts)
    print(f"Client {i} Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")

comm_cost = count_parameters(server.global_model.state_dict())
print("\nFedAvg communication cost per round:", comm_cost)

plt.plot(round_losses)
plt.title("FedAvg Convergence")
plt.xlabel("Rounds")
plt.ylabel("Average Loss")
plt.savefig("results/fedavg_convergence.png")
plt.close()

print("\nConvergence plot saved to results/fedavg_convergence.png")

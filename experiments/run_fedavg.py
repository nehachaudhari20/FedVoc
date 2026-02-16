from utils.data_loader import load_shakespeare_clients
from tokenizers import Tokenizer
from clients.client_fedavg import FedAvgClient
from server.server_base import Server
import torch

clients_data = load_shakespeare_clients(num_clients=3)

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

print("Starting TRUE FedAvg baseline with DistilBERT...")

for round in range(8):
    print(f"\n--- Round {round} ---")

    client_weights = []

    for client in clients:
        client.initialize_local_model(server.global_model)

        loss = client.train_one_epoch()
        print("Client train loss:", loss)

        client_weights.append(client.get_model_weights())

    server.aggregate(client_weights)

print("\nEvaluating FedAvg baseline...")

for i, client in enumerate(clients):
    loss, ppl = client.evaluate(client.test_texts)
    print(f"Client {i} Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")

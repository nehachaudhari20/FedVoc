from utils.data_loader import load_shakespeare_clients
from tokenizers import Tokenizer
from clients.client_fedavg import FedAvgClient
from server.server_base import Server

clients_data = load_shakespeare_clients(num_clients=3)

# 🔥 Load GLOBAL tokenizer
tokenizer = Tokenizer.from_file(
    "fed_tokenizers/global_tokenizer.json"
)

vocab_size = tokenizer.get_vocab_size()

server = Server(vocab_size)

clients = []

for cid, texts in clients_data.items():
    client = FedAvgClient(tokenizer, texts)
    clients.append(client)

print("Starting TRUE FedAvg baseline with DistilBERT...")

for round in range(8):
    print(f"\n--- Round {round} ---")

    client_weights = []

    for client in clients:
        client.initialize_local_model(server.global_model)

        loss = client.train_one_epoch()
        print("Client loss:", loss)

        client_weights.append(client.get_model_weights())

    server.aggregate(client_weights)

print("\nTraining complete.")

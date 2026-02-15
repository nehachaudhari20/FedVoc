from utils.data_loader import load_shakespeare_clients
from tokenizers import Tokenizer
from clients.client_fedavg import FedAvgClient
from server.server_base import Server

clients_data = load_shakespeare_clients(num_clients=3)

clients = []
client_vocab_sizes = []

# 🔥 Load DIFFERENT tokenizer per client
for i, (cid, texts) in enumerate(clients_data.items()):
    tokenizer = Tokenizer.from_file(
        f"fed_tokenizers/tokenizer_client_{i}.json"
    )

    vocab_size = tokenizer.get_vocab_size()
    client_vocab_sizes.append(vocab_size)

    client = FedAvgClient(tokenizer, texts)
    clients.append(client)

print("Client vocab sizes:", client_vocab_sizes)

# ⚠️ Server uses first client's vocab size
server = Server(client_vocab_sizes[0])

print("Starting FedAvg with LOCAL tokenizers...")

for round in range(3):
    print(f"\n--- Round {round} ---")

    client_weights = []

    for client in clients:
        client.initialize_local_model(server.global_model)

        loss = client.train_one_epoch()
        print("Client loss:", loss)

        client_weights.append(client.get_model_weights())

    server.aggregate(client_weights)

print("\nTraining complete.")

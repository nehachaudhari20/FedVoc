from utils.data_loader import load_shakespeare_clients
from tokenizers import Tokenizer
from clients.client_fedavg import FedAvgClient
from server.server_base import Server

clients_data = load_shakespeare_clients()

server = Server()

clients = []

for i, (cid, texts) in enumerate(clients_data.items()):
    tokenizer = Tokenizer.from_file(
        f"fed_tokenizers/tokenizer_client_{i}.json"
    )

    client = FedAvgClient(tokenizer, texts)
    clients.append(client)

print("Starting TRUE FedAvg baseline...")

for round in range(5):
    print(f"\n--- Round {round} ---")

    client_weights = []

    for client in clients:
        # Send global model copy
        client.initialize_local_model(server.global_model)

        loss = client.train_one_epoch()
        print("Client loss:", loss)

        # Collect local updated model
        client_weights.append(client.get_model_weights())

    # Server aggregation
    server.aggregate(client_weights)

print("\nTraining complete.")

from utils.data_loader import load_shakespeare_clients
from tokenizers import Tokenizer
from clients.client_fedvoc import FedVocClient
import copy

clients_data = load_shakespeare_clients(num_clients=3)

clients = []

for i, (cid, texts) in enumerate(clients_data.items()):
    tokenizer = Tokenizer.from_file(
        f"fed_tokenizers/tokenizer_client_{i}.json"
    )

    client = FedVocClient(tokenizer, texts)
    clients.append(client)

# 🔥 Initialize global shared state (adapter + encoder)
global_shared = clients[0].get_shared_weights()

print("Starting FedVoc v2 training (adapter + encoder shared)...")

for round in range(8):
    print(f"\n--- Round {round} ---")

    shared_updates = []

    for client in clients:
        client.initialize_shared_weights(global_shared)

        loss = client.train_one_epoch()
        print("Client loss:", loss)

        shared_updates.append(client.get_shared_weights())

    # FedAvg on shared components
    new_shared = copy.deepcopy(global_shared)

    for key in new_shared.keys():
        new_shared[key] = sum(
            update[key] for update in shared_updates
        ) / len(shared_updates)

    global_shared = new_shared

print("\nFedVoc v2 training complete.")

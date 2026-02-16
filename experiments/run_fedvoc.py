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

# Initialize global adapter
global_adapter = copy.deepcopy(clients[0].model.adapter.state_dict())

print("Starting FedVoc training...")

for round in range(8):
    print(f"\n--- Round {round} ---")

    adapter_updates = []

    for client in clients:
        client.initialize_local_adapter(global_adapter)

        loss = client.train_one_epoch()
        print("Client loss:", loss)

        adapter_updates.append(client.get_adapter_weights())

    # FedAvg on adapter only
    new_adapter = copy.deepcopy(global_adapter)

    for key in new_adapter.keys():
        new_adapter[key] = sum(
            update[key] for update in adapter_updates
        ) / len(adapter_updates)

    global_adapter = new_adapter

print("\nFedVoc training complete.")

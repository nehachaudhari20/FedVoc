from utils.oov_eval import compute_oov_tokens
from utils.communication import count_parameters
from tokenizers import Tokenizer
from clients.client_fedvoc import FedVocClient
import copy
import matplotlib.pyplot as plt

from utils.data_loader import load_domain_clients

clients_data = load_domain_clients()
clients = []

for i, (cid, data) in enumerate(clients_data.items()):

    tokenizer = Tokenizer.from_file(
        f"fed_tokenizers/tokenizer_client_{i}.json"
    )

    client = FedVocClient(tokenizer, data["train"])
    client.test_texts = data["test"]
    clients.append(client)

# Initialize global shared state
global_shared = clients[0].get_shared_weights()

round_losses = []

print("Starting FedVoc v2 training (adapter + encoder shared)...")

for round in range(8):
    print(f"\n--- Round {round} ---")

    shared_updates = []
    total_round_loss = 0

    for client in clients:
        client.initialize_shared_weights(global_shared)

        loss = client.train_one_epoch()
        print("Client loss:", loss)

        total_round_loss += loss
        shared_updates.append(client.get_shared_weights())

    avg_loss = total_round_loss / len(clients)
    round_losses.append(avg_loss)

    new_shared = copy.deepcopy(global_shared)

    for key in new_shared.keys():
        new_shared[key] = sum(
            update[key] for update in shared_updates
        ) / len(shared_updates)

    global_shared = new_shared

print("\nFedVoc v2 training complete.")

# -------- Evaluation --------

print("\nEvaluating FedVoc v2...")

for i, client in enumerate(clients):
    loss, ppl = client.evaluate(client.test_texts)
    print(f"Client {i} Test Loss: {loss:.4f} | Perplexity: {ppl:.4f}")

# -------- OOV Analysis --------

print("\nOOV Analysis:")

for i, client in enumerate(clients):

    other_tokenizers = [
        clients[j].tokenizer for j in range(len(clients)) if j != i
    ]

    oov_tokens = compute_oov_tokens(
        client.tokenizer,
        other_tokenizers
    )

    print(f"Client {i} OOV token count:", len(oov_tokens))

# -------- Communication Cost --------

comm_cost = count_parameters(global_shared)
print("\nFedVoc communication cost per round:", comm_cost)

# -------- Save Convergence Plot --------

plt.plot(round_losses)
plt.title("FedVoc Convergence")
plt.xlabel("Rounds")
plt.ylabel("Average Loss")
plt.savefig("results/fedvoc_convergence.png")
plt.close()

print("\nConvergence plot saved to results/fedvoc_convergence.png")

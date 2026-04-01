# plot_perplexity
import matplotlib.pyplot as plt
import os

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Data (FINAL results)
clients = ["Client 0", "Client 1", "Client 2"]

fedavg_ppl = [5.94, 139.15, 195.29]
fedvoc_ppl = [6.44, 227.41, 240.73]

x = range(len(clients))
width = 0.35

plt.figure()

plt.bar([i - width/2 for i in x], fedavg_ppl, width=width, label="FedAvg")
plt.bar([i + width/2 for i in x], fedvoc_ppl, width=width, label="FedVoc")

plt.xlabel("Clients")
plt.ylabel("Perplexity")
plt.title("Perplexity Comparison: FedAvg vs FedVoc")
plt.xticks(x, clients)
plt.legend()

plt.savefig("results/perplexity_comparison.png")
plt.show()
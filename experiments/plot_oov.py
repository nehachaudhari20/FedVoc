# plot_oov
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

clients = ["Client 0", "Client 1", "Client 2"]
oov_counts = [2680, 2594, 2735]

plt.figure()

plt.bar(clients, oov_counts)

plt.xlabel("Clients")
plt.ylabel("OOV Token Count")
plt.title("Vocabulary Heterogeneity (OOV Tokens)")

plt.savefig("results/oov_tokens.png")
plt.show()
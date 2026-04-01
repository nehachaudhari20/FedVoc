# plot_communication
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

methods = ["FedAvg", "FedVoc"]
params = [74146184, 66461184]

plt.figure()

plt.bar(methods, params)

plt.xlabel("Method")
plt.ylabel("Parameters per Round")
plt.title("Communication Cost Comparison")

plt.savefig("results/communication_cost.png")
plt.show()
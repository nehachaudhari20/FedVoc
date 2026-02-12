# FedVoc

**Federated Vocabulary Alignment for Transformer-Based Language Models**

---

## 📌 Overview

**FedVoc** addresses the challenge of federated learning (FL) for NLP, where clients have heterogeneous vocabularies. It introduces a framework that:

* Supports client-specific tokenizers and vocabularies
* Aligns local embeddings into a shared semantic space
* Mitigates embedding drift and improves convergence in federated transformers

---

## 🧠 Research Contributions

* **Vocabulary Drift Formalization**: Quantifies token distribution skew and vocabulary overlap.
* **Embedding Divergence Analysis**: Measures semantic drift during FL.
* **FedVoc Framework**: Introduces low-rank alignment adapters for vocabulary alignment.
* **Communication Efficiency**: Shares alignment parameters, reducing communication overhead.
* **Empirical Validation**: Demonstrates improved perplexity, OOV recall, and convergence.

---

## 🗃️ Repository Structure

```bash
fedvoc/
├── data/            # Datasets
├── tokenizers/      # Client-specific tokenizers
├── models/          # Model architectures
├── clients/         # Client FL logic
├── server/          # Server aggregation
├── experiments/     # Experiment runners
├── utils/           # Evaluation and utilities
├── results/         # Logs and plots
├── config.yaml      # Configurations
└── README.md        # Documentation
```

---

## ⚙️ Setup

### 1️⃣ Python Version

```bash
Python 3.10
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

To simulate vocabulary drift, generate client-specific tokenizers:

```bash
python tokenizers/build_tokenizers.py
```

---

## 🚀 Running Experiments

### 🔹 Baseline — FedAvg

```bash
python experiments/run_fedavg.py
```

### 🔹 Proposed Method — FedVoc

```bash
python experiments/run_fedvoc.py
```

---

## 🧪 Evaluation Metrics

FedVoc is evaluated using:

* **Perplexity**
* **OOV Recall**
* **Embedding Drift**
* **FL Convergence**

Run evaluation:

```bash
python utils/evaluate.py
```

---

## 📈 Results & Visualization

Results are stored in:

```
results/
├── logs/
├── round_metrics.json
└── plots/
```





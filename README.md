# EcoFoundation

**EcoFoundation** is a modular framework for analyzing spatial transcriptomics data using graph neural networks (GNNs), spatial statistics, and machine learning. It supports exploration of cellular ecosystems by integrating SPATA2 outputs, CytoSPACE-inferred single-cell positions, and customizable GNN architectures.

---

## 📌 Features

- 🔁 **Graph Neural Networks**: Custom GAT layers with edge features for modeling spatial connectivity.
- 🧠 **Graph Encoder**: Encodes spatial transcriptomics into meaningful latent representations.
- 🎯 **Prediction Heads**: Extendable MLP-based heads for classification or regression.
- 🧩 **Batch Effect Removal**: Domain discriminator for adversarial training setups.
- 🌍 **Ecosystem Visualization**: R functions to visualize cellular subgraphs using Voronoi tiling.
- ⏹️ **Early Stopping**: Monitoring to halt training when validation loss plateaus.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/heilandd/EcoFoundation.git
cd EcoFoundation
```

🛠️ Example Workflow
	1.	Process your spatial transcriptomics data (e.g., with SPATA2 + CytoSPACE)
	2.	Extract subgraphs of interest
	3.	Encode subgraphs using GraphEncoder
	4.	Train a classifier (e.g., with MLP) to predict cell state or perturbation
	5.	Visualize ecosystems using plotEcosystem() in R


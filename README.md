# CLA3GC---Linear time Attributed Graph Clustering via Collaborative Learning of Adaptive Anchors

## 🧩 Features

- Graph Filtering: Low-pass filtering for noise reduction and feature smoothing.

- Anchor Sampling: K-means and mini-batch K-means for efficient anchor selection.

- Multi-view Support: Handles multi-attribute and multi-relational graphs.

- Efficient Optimization: Iterative optimization of anchor and representation matrices.

- Evaluation Metrics: ACC, NMI, ARI, F1, and Purity for clustering performance.


<img width="1534" height="398" alt="image" src="https://github.com/user-attachments/assets/936e90b7-15fa-4cd9-9ff1-92fee369e047" />



## 📦 Project Structure

```
.
├── CLA3GC.py                 # Main training and reproduction script
├── anchor.py                 # Anchor sampling methods (k-means, mini-batch, random)
├── clustering.py             # Clustering algorithms (single/multi-view)
├── data_loader.py            # Data loading and preprocessing for various datasets
├── data_preprocess.py        # Graph preprocessing utilities
├── graph_filtering.py        # Low-pass graph filtering (sparse and dense)
├── metrics.py                # Clustering evaluation metrics
├── utils.py                  # Normalization and dimension reduction utilities
└── README.md
```

## 🗂️ Supported Datasets

Single-view Graphs: `Cora`
Topological Multi-view Graphs: `ACM`
Attribute Multi-view Graphs: `AMAP`

For other datasets, please contact the authors via guog92764@gmail.com

## 🚀 Run Clustering

You can modify the dataset name in the __main__ block of CLA3GC.py, and run directly.


## 📜 License

This project is for academic use only. Please contact the authors for commercial use.






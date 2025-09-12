# $CLA^3GC$: Linear time Attributed Graph Clustering via Collaborative Learning of Adaptive Anchors

This is the code of paper: Linear-time Attributed Graph Clustering via Collaborative Learning of Adaptive Anchors.

## 🚀 Run $CLA^3GC$

You can modify the `dataname` in the __main__ block of CLA3GC.py, and run directly.

## 🧩 Framework

The framework of CLA3GC is shown in the following figure.

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

For other datasets, please contact the authors via guog92764@gmail.com.

| **Datasets** | **Nodes** | **Edges** | **Features** | **Classes** |
|--------------|-----------|-----------|--------------|-------------|
| Cora | 2,708 | 13,264 | 1,433 | 7 |
| Citeseer | 3,327 | 12,431 | 3,703 | 6 |
| PubMed | 19,717 | 108,365 | 500 | 3 |
| ACM | 3,025 | 29,281/2,210,761 | 1,830 | 3 |
| DBLP | 4,057 | 11,113/5,000,495/676,335 | 334 | 4 |
| AMAP | 7,487 | 119,043 | 745/7,487 | 8 |
| AMAC | 13,381 | 259,159 | 767/13,381 | 10 |
| OGBN-Arxiv | 169,343 | 1,327,142 | 128 | 40 |
| OGBN-Products | 2,449,029 | 61,859,140 | 100 | 47 |

## 📜 License

This project is for academic use only. Please contact the authors for commercial use.






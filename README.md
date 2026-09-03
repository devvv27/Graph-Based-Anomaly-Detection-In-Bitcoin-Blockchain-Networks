# Graph-Based Bitcoin Transaction Classification

This project investigates illicit Bitcoin transaction detection on the Elliptic transaction graph. It combines transaction-level feature models with graph neural networks (GNNs), temporal evaluation, graph-derived features, and post-hoc AML typology analysis.

## Data model

The notebooks expect the public Elliptic dataset:

- `elliptic_txs_features.csv`: transaction identifiers, time steps, and 165 transaction features.
- `elliptic_txs_classes.csv`: `licit`, `illicit`, and `unknown` labels.
- `elliptic_txs_edgelist.csv`: directed transaction relationships.

Unknown labels are excluded from supervised training and evaluation. The graph is built over labeled transactions and the edge list is restricted to transactions represented in that graph. The later experiments contain 46,564 labeled nodes and 36,624 retained edges, with 4,545 illicit and 42,019 licit examples.

## Core methodology

1. **Preprocessing and feature regimes**
   Transaction IDs are joined with labels and temporal metadata. The notebooks compare local features (`LF`, the first 93 feature columns) with the complete feature set (`AF`, all 165 columns). Spark is used in the prototype and in the latest experiment for scalable preprocessing and Parquet persistence.

2. **Temporal evaluation**
   The main protocol trains on time steps `<= 34` and evaluates on later steps (`> 34`). This is intended to approximate deployment on future transactions and reduce temporal leakage. The earliest prototype also contains a random stratified split, whose results are not directly comparable with the temporal experiments.

3. **Tabular baselines**
   Logistic Regression, Random Forest, MLP, and CatBoost provide non-graph reference models. Class imbalance is handled with weighted objectives or focal loss where supported. F1 is the primary reported illicit-class metric, with per-time-step analysis in the improved notebooks.

4. **Graph neural networks**
   The primary graph model is a two-layer, multi-head Graph Attention Network (`GATConv`) with 128-dimensional hidden representations. Attention-weighted neighborhood aggregation produces node embeddings and an illicit/licit classifier. Training variants use focal loss or weighted cross-entropy, Adam-based optimization, dropout, learning-rate scheduling, checkpointing, and early stopping.

5. **Hybrid and ensemble classification**
   GAT embeddings are concatenated with transaction features and, in the latest experiment, class probabilities and train-subgraph graph features. CatBoost, Random Forest, and XGBoost act as downstream classifiers. The latest pipeline also trains a GraphSAGE branch and combines model probabilities through soft voting. Graph features are computed from the training subgraph to avoid using future graph structure.

6. **Explainability and AML typologies**
   The explainability experiments apply `GNNExplainer` to selected nodes, producing feature and edge masks for local subgraphs. NetworkX visualizations and rule-based interpretation map transaction neighborhoods to AML patterns including Fan-in, Fan-out, Layering, Pass-through, and Structuring. GNN feature attribution is compared with Random Forest feature importance.

## Experiment progression

| Folder | Role |
| --- | --- |
| `temporal-hybrid` | First temporal GAT, embedding-plus-feature hybrids, and serialized baseline models. |
| `loss-feature-analysis` | Loss-function comparison and local-versus-aggregated feature importance analysis. |
| `explainer-prototype` | Initial GNNExplainer and AML typology implementation. |
| `aml-explainability` | Cleaned and formalized GNNExplainer/AML typology notebook. |
| `leakage-safe-ensemble` | Scalable GAT plus GraphSAGE ensemble with train-derived graph features and downstream boosting. |

The root notebooks are the random-split prototype (`Emdsem.ipynb`) and the unexecuted improved temporal template (`Emdsem_improved.ipynb`). Model checkpoints and training logs in the experiment folders are generated artifacts, not additional algorithmic stages.

## Implementation boundary

The executable notebooks verify GAT, GraphSAGE, focal or weighted loss, train-derived graph features, CatBoost/XGBoost downstream models, soft voting, temporal metrics, and GNNExplainer. `Novelty.txt` additionally describes SGAT-BC, gated two-hop layers, five-model SGAT bagging, label smoothing, seven topology features, F2 threshold tuning, AdamW, cosine warm restarts, and gradient clipping. Those additions should be treated as proposed/report-level methodology unless implemented in a notebook or source module.
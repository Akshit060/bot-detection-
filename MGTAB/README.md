## MGTAB GNN Trainer

Train a GNN on the tensors in this folder: `features.pt`, `edge_index.pt`, optional `edge_type.pt`, `edge_weight.pt`, and labels `labels_bot.pt` (binary 0/1) and `labels_stance.pt` (class indices).

### Install

1. Create a virtual environment (recommended)
2. Install PyTorch matching your CUDA/CPU, then install requirements. Example (CPU only):

```bash
pip install -r requirements.txt
```

For CUDA, first install the correct `torch` wheel, then install the rest.

### Run

```bash
python train_gnn.py --epochs 100 --hidden-dim 128 --layers 2 --dropout 0.2
```

Key flags:
- `--data-dir`: directory containing the `.pt` tensors (default `MGTAB`)
- `--device`: `cuda` or `cpu` (auto-detected by default)

The script supports:
- RGCN (if `edge_type.pt` exists) or GCN otherwise
- Multi-task training: bot detection (BCEWithLogits) and stance classification (CrossEntropy)
- Random masks for train/val/test splits



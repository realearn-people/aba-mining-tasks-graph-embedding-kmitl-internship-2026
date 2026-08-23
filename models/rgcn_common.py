"""
Training/eval loop for R-GCN on the heterogeneous ABA graph (graph_construction_hetero.py).
Mirrors gnn_common.py's structure and metrics exactly, so results are directly
comparable to the plain GCN/GAT/GraphSAGE runs — same classifier head, same
class-weighted loss, same Accuracy/macro-P/R/F1/macro-AUC(OvR) methodology.
The only structural difference: the encoder is RGCNConv, which is relation-aware
(uses edge_type), instead of the untyped GCN/GAT/SAGE convs.

Claude-Assisted
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.nn import RGCNConv

from graph_construction_hetero import ABAHeteroGraph
from graph_construction import RELATIONS
from gnn_common import compute_metrics  # same metric implementation as the plain GNNs

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RGCNEdgeClassifier(nn.Module):
    def __init__(self, in_dim, num_relations, hidden_dim=128, num_classes=3,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.convs.append(RGCNConv(d_in, hidden_dim, num_relations, num_bases=None))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x, edge_index, edge_type):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index, edge_type)
            h = norm(h)
            h = F.relu(h)
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def classify(self, z, head_idx, tail_idx):
        feat = torch.cat([z[head_idx], z[tail_idx]], dim=-1)
        return self.classifier(feat)


def train_and_evaluate(model_name="RGCN", hidden_dim=128, num_layers=2, dropout=0.5,
                        lr=0.005, weight_decay=5e-4, max_epochs=300, patience=30,
                        min_epochs=15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print(f"Training {model_name} (heterogeneous graph, typed edges)")
    print("=" * 60)

    graph = ABAHeteroGraph()
    print(f"Nodes: {graph.num_nodes:,}  |  Relations: {graph.num_relations}  |  "
          f"Edges (both directions): {graph.edge_index.shape[1]:,}")
    print(f"Train: {len(graph.df_train):,}  Val: {len(graph.df_val):,}  Test: {len(graph.df_test):,}")

    x = graph.node_features.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)
    edge_type = graph.edge_type.to(DEVICE)

    train_h, train_t, train_y = [t.to(DEVICE) for t in graph.split_tensors(graph.df_train)]
    val_h, val_t, val_y = [t.to(DEVICE) for t in graph.split_tensors(graph.df_val)]
    test_h, test_t, test_y = [t.to(DEVICE) for t in graph.split_tensors(graph.df_test)]

    class_weights = graph.class_weights().to(DEVICE)
    print(f"Class weights (sqrt-balanced, train): "
          f"{dict(zip(RELATIONS, class_weights.cpu().numpy().round(3)))}")

    model = RGCNEdgeClassifier(
        in_dim=graph.num_nodes, num_relations=graph.num_relations,
        hidden_dim=hidden_dim, num_classes=len(RELATIONS),
        num_layers=num_layers, dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_left = patience

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, edge_index, edge_type)
        logits = model.classify(z, train_h, train_t)
        loss = criterion(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x, edge_index, edge_type)
            val_logits = model.classify(z, val_h, val_t)
            val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
            val_metrics, _ = compute_metrics(val_y.cpu().numpy(), val_probs)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  loss={loss.item():.4f}  "
                  f"val_acc={val_metrics['Accuracy']:.4f}  val_macroF1={val_metrics['F1']:.4f}")

        if val_metrics["F1"] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics["F1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = patience
        elif epoch >= min_epochs:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch}, "
                      f"best val macro-F1 {best_val_f1:.4f})")
                break

    elapsed = time.time() - start_time
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index, edge_type)
        test_logits = model.classify(z, test_h, test_t)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
    test_metrics, test_pred = compute_metrics(test_y.cpu().numpy(), test_probs)

    print("\n" + "=" * 60)
    print(f"{model_name} — Test results (best epoch {best_epoch}, {elapsed:.1f}s)")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v if v is None else round(v, 4)}")

    # per-relation breakdown — SUPPORT is expected to be near-trivial (node-type
    # alone determines it), the real signal is CONTRARY_TO vs NOT_CONTRARY
    print("\n  Per-relation (test):")
    y_true = test_y.cpu().numpy()
    for i, rel in enumerate(RELATIONS):
        mask = y_true == i
        if mask.sum() > 0:
            acc_i = (test_pred[mask] == i).mean()
            print(f"    {rel:<14}: n={mask.sum():5d}  acc={acc_i:.4f}")

    _save_outputs(model_name, graph, z, test_probs, test_pred, best_epoch, elapsed, test_metrics)
    return test_metrics


def _save_outputs(model_name, graph, z, test_probs, test_pred, best_epoch, elapsed, test_metrics):
    out_dir = ROOT / "outputs" / f"{model_name.lower()}_gnn_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = z.detach().cpu().numpy().astype(np.float32)
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(emb)

    rows = []
    for name, idx in graph.entity_to_id.items():
        row = {"entity_id": idx, "entity_name": name, "node_type": graph.node_type_name[idx],
               "x": round(float(coords_2d[idx, 0]), 6), "y": round(float(coords_2d[idx, 1]), 6)}
        for d, val in enumerate(emb[idx]):
            row[f"emb_{d}"] = round(float(val), 6)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "entity_embeddings.csv", index=False)

    df_pred = graph.df_test.copy().reset_index(drop=True)
    df_pred["predicted"] = [RELATIONS[i] for i in test_pred]
    df_pred["correct"] = df_pred["predicted"] == df_pred["relation"]
    for i, rel in enumerate(RELATIONS):
        df_pred[f"prob_{rel}"] = test_probs[:, i]
    df_pred.to_csv(out_dir / "test_predictions.csv", index=False)

    import json
    payload = {
        "model": model_name, "best_epoch": best_epoch, "training_time_sec": round(elapsed, 2),
        "metrics": test_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved: {out_dir}/entity_embeddings.csv")
    print(f"Saved: {out_dir}/test_predictions.csv")
    print(f"Saved: {out_dir}/metrics.json")

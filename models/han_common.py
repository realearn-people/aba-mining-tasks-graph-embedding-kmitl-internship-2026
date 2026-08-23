"""
Training/eval loop for HAN (Heterogeneous Graph Attention Network) on the same
typed ABA graph as rgcn_common.py — same node types, same edge types, same
split, same classifier head, same metrics. Only the encoder differs:

  R-GCN — one linear weight matrix per relation, messages summed (no attention).
  HAN   — two-level attention: (1) node-level attention over neighbors within
          each relation (a GAT-style score per edge), (2) semantic-level
          attention that learns how much each relation type should count
          toward a node's final embedding, instead of a fixed sum.

PyG's HANConv treats each (src_type, edge_name, dst_type) triple as a
single-hop "metapath" — the standard way to apply HAN when you don't hand-craft
longer composite metapaths, and it reuses graph_construction_hetero.py's edge
typing unchanged, so this is a direct apples-to-apples swap of the encoder.

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
from torch_geometric.nn import HANConv

from graph_construction_hetero import ABAHeteroGraph
from graph_construction import RELATIONS
from gnn_common import compute_metrics  # same metric implementation as the other GNNs

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_hetero_tensors(graph: ABAHeteroGraph):
    """Convert ABAHeteroGraph's flat (global-id) edge_index/edge_type into the
    x_dict / edge_index_dict / metadata form HANConv expects, using LOCAL
    per-type indices — built directly from the same edges R-GCN trained on,
    so the two encoders see an identical graph."""
    node_type_of = {}
    local_id_of = {}
    x_dict, global_id_dict = {}, {}
    for t in graph.NODE_TYPES:
        idx = [i for i in range(graph.num_nodes) if graph.node_type_name[i] == t]
        x_dict[t] = graph.node_features[idx]
        global_id_dict[t] = torch.tensor(idx, dtype=torch.long)
        for local_i, gi in enumerate(idx):
            node_type_of[gi] = t
            local_id_of[gi] = local_i

    buckets = {}
    src_all = graph.edge_index[0].tolist()
    dst_all = graph.edge_index[1].tolist()
    rel_all = graph.edge_type.tolist()
    for s, d, r in zip(src_all, dst_all, rel_all):
        rel_name = graph.relation_names[r]
        key = (node_type_of[s], rel_name, node_type_of[d])
        buckets.setdefault(key, ([], []))
        buckets[key][0].append(local_id_of[s])
        buckets[key][1].append(local_id_of[d])

    edge_index_dict = {
        key: torch.tensor(np.stack([ss, dd]), dtype=torch.long)
        for key, (ss, dd) in buckets.items()
    }
    metadata = (list(graph.NODE_TYPES), list(edge_index_dict.keys()))
    return x_dict, edge_index_dict, metadata, global_id_dict


class HANEdgeClassifier(nn.Module):
    def __init__(self, in_dim, metadata, hidden_dim=128, num_classes=3,
                 num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.node_types = metadata[0]
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            self.convs.append(HANConv(d_in, hidden_dim, heads=heads,
                                       dropout=dropout, metadata=metadata))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x_dict, edge_index_dict):
        h_dict = x_dict
        for conv in self.convs:
            out_dict = conv(h_dict, edge_index_dict)
            # HANConv can return None for a type with no incoming edges in a batch
            h_dict = {
                k: (F.relu(v) if v is not None
                    else h_dict[k].new_zeros(h_dict[k].shape[0], self.hidden_dim))
                for k, v in out_dict.items()
            }
        return h_dict

    def classify_global(self, h_dict, global_id_dict, num_nodes, hidden_dim, head_idx, tail_idx):
        z = h_dict[self.node_types[0]].new_zeros(num_nodes, hidden_dim)
        for t in self.node_types:
            z[global_id_dict[t]] = h_dict[t]
        feat = torch.cat([z[head_idx], z[tail_idx]], dim=-1)
        return self.classifier(feat), z


def train_and_evaluate(model_name="HAN", hidden_dim=128, num_layers=2, dropout=0.5,
                        heads=4, lr=0.005, weight_decay=5e-4, max_epochs=300,
                        patience=30, min_epochs=15, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print(f"Training {model_name} (heterogeneous graph, typed edges, attention encoder)")
    print("=" * 60)

    graph = ABAHeteroGraph()
    x_dict, edge_index_dict, metadata, global_id_dict = build_hetero_tensors(graph)
    print(f"Nodes: {graph.num_nodes:,}  |  Node types: {metadata[0]}  |  "
          f"Edge types: {len(metadata[1])}  |  Edges (both directions): {graph.edge_index.shape[1]:,}")
    print(f"Train: {len(graph.df_train):,}  Val: {len(graph.df_val):,}  Test: {len(graph.df_test):,}")

    x_dict = {k: v.to(DEVICE) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in edge_index_dict.items()}
    global_id_dict = {k: v.to(DEVICE) for k, v in global_id_dict.items()}

    train_h, train_t, train_y = [t.to(DEVICE) for t in graph.split_tensors(graph.df_train)]
    val_h, val_t, val_y = [t.to(DEVICE) for t in graph.split_tensors(graph.df_val)]
    test_h, test_t, test_y = [t.to(DEVICE) for t in graph.split_tensors(graph.df_test)]

    class_weights = graph.class_weights().to(DEVICE)
    print(f"Class weights (sqrt-balanced, train): "
          f"{dict(zip(RELATIONS, class_weights.cpu().numpy().round(3)))}")

    model = HANEdgeClassifier(
        in_dim=graph.num_nodes, metadata=metadata, hidden_dim=hidden_dim,
        num_classes=len(RELATIONS), num_layers=num_layers, dropout=dropout, heads=heads,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward_logits(h_idx, t_idx):
        h_dict = model.encode(x_dict, edge_index_dict)
        logits, z = model.classify_global(h_dict, global_id_dict, graph.num_nodes,
                                           hidden_dim, h_idx, t_idx)
        return logits, z

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_left = patience

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = forward_logits(train_h, train_t)
        loss = criterion(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = forward_logits(val_h, val_t)
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
        test_logits, z = forward_logits(test_h, test_t)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
    test_metrics, test_pred = compute_metrics(test_y.cpu().numpy(), test_probs)

    print("\n" + "=" * 60)
    print(f"{model_name} — Test results (best epoch {best_epoch}, {elapsed:.1f}s)")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v if v is None else round(v, 4)}")

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

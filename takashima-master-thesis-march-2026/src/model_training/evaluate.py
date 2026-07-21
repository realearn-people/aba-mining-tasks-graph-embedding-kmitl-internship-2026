import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_relation_ranking_metrics(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Relation-ranking evaluation.

    For each test sample, the true relation is ranked against all other possible
    relations using the model's predicted probability/score per class.

    Args:
        probs:  (N,) for binary  — probability of class 1
                (N, K) for K-class — probability per class
        y_true: (N,) integer class labels

    Returns:
        dict with mean_rank, mrr, hits@1_micro, hits@1_macro, hits@3, hits@10
    """
    probs  = np.asarray(probs)
    y_true = np.asarray(y_true, dtype=int)

    if probs.ndim == 1:
        # binary: build (N, 2) matrix — class-0 score = 1-p, class-1 score = p
        probs_2d = np.stack([1.0 - probs, probs], axis=1)
    else:
        probs_2d = probs

    num_classes = probs_2d.shape[1]
    ranks = []
    for i, true_label in enumerate(y_true):
        true_score = probs_2d[i, true_label]
        rank = 1 + int(np.sum(
            [probs_2d[i, c] > true_score for c in range(num_classes) if c != true_label]
        ))
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)

    # Macro Hits@1: compute Hits@1 per class then average — unaffected by class imbalance.
    classes = np.unique(y_true)
    per_class_h1 = [float(np.mean(ranks[y_true == c] <= 1)) for c in classes]
    hits_at_1_macro = float(np.mean(per_class_h1))

    return {
        'mean_rank':    float(np.mean(ranks)),
        'mrr':          float(np.mean(1.0 / ranks)),
        'hits@1_micro': float(np.mean(ranks <= 1)),
        'hits@1_macro': hits_at_1_macro,
        'hits@3':       float(np.mean(ranks <= 3)),
        'hits@10':      float(np.mean(ranks <= 10)),
    }

compute_ranking_metrics = compute_relation_ranking_metrics


def evaluate_model(model, data, test_edges, node_to_idx, device: str = 'cpu', num_classes: int = 2):
    """モデルを評価"""
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        edge_pairs = [(node_to_idx[u], node_to_idx[v]) for (u, v), _ in test_edges]
        x_dev = data.x.to(device)
        edge_index_dev = data.edge_index.to(device)
        edge_type_dev = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        predictions = model(x_dev, edge_index_dev, edge_type_dev, edge_pairs)
        predictions_np = predictions.detach().cpu().numpy()

    y_true = np.array([label for _, label in test_edges])

    if num_classes == 2:
        y_pred = (predictions_np > 0.5).astype(int)
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
            'auc':       roc_auc_score(y_true, predictions_np) if len(set(y_true.tolist())) > 1 else 0,
        }
        metrics.update(compute_relation_ranking_metrics(predictions_np, y_true))
    else:
        # predictions_np: (N, num_classes) raw logits from R-GCN.
        # sklearn's roc_auc_score requires probabilities (rows sum to 1), so apply softmax first.
        e = np.exp(predictions_np - predictions_np.max(axis=1, keepdims=True))
        probs_np = e / e.sum(axis=1, keepdims=True)

        y_pred = np.argmax(probs_np, axis=1)
        try:
            auc = roc_auc_score(y_true, probs_np, multi_class='ovr', average='macro') \
                  if len(set(y_true.tolist())) > 1 else 0.0
        except Exception:
            auc = 0.0
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
            'auc':       auc,
        }
        metrics.update(compute_relation_ranking_metrics(probs_np, y_true))

    return metrics, predictions_np


def evaluate_baseline(baseline, test_edges, num_classes: int = 2):
    """ベースラインを評価"""
    edge_pairs = [(u, v) for (u, v), _ in test_edges]
    predictions = baseline.predict(edge_pairs)  # 1-D (binary) or 2-D (multi-class)

    y_true = np.array([label for _, label in test_edges])

    if num_classes == 2:
        predictions_1d = np.array(predictions)
        y_pred = (predictions_1d > 0.5).astype(int)
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
            'auc':       roc_auc_score(y_true, predictions_1d) if len(set(y_true.tolist())) > 1 else 0,
        }
        metrics.update(compute_relation_ranking_metrics(predictions_1d, y_true))
    else:
        predictions_2d = np.array(predictions)  # (N, num_classes)
        y_pred = np.argmax(predictions_2d, axis=1)
        try:
            auc = roc_auc_score(y_true, predictions_2d, multi_class='ovr', average='macro') \
                  if len(set(y_true.tolist())) > 1 else 0.0
        except Exception:
            auc = 0.0
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
            'auc':       auc,
        }
        metrics.update(compute_relation_ranking_metrics(predictions_2d, y_true))
        predictions = predictions_2d

    return metrics, predictions
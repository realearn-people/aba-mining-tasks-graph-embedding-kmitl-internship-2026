import random
from typing import List, Tuple, Dict, Set


Edge = Tuple[Tuple[str, str], int]  # ((u, v), label)


def split_train_val(
    train_edges: List[Edge],
    val_ratio: float,
    seed: int
) -> Tuple[List[Edge], List[Edge]]:
    """
    学習エッジから検証エッジを固定分割する（エッジレベルランダム）。
    以降のサイズ変更では val は固定し、 train_base のみを縮小する。
    """
    rng = random.Random(seed)
    shuffled = train_edges.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(n * val_ratio)) if n > 0 else 0
    val_edges = shuffled[:n_val]
    train_base = shuffled[n_val:]
    return train_base, val_edges


def sample_node_induced(
    edges: List[Edge],
    pct: float,
    seed: int
) -> Tuple[List[Edge], Set[str], Dict[str, int]]:
    """
    ノード誘導サンプリングにより、与えられたエッジ集合を縮小する。
    - edges: 対象エッジ（学習用ベース）
    - pct: 残すノード割合（0.2, 0.4, ... 1.0）
    - seed: 乱数シード
    Returns:
      - フィルタ後エッジ
      - 抽出ノード集合
      - 統計（edges, pos, neg, nodes）
    """
    if pct <= 0.0 or pct > 1.0:
        raise ValueError(f"pct must be in (0, 1], got {pct}")
    # 対象ノード集合
    node_set: Set[str] = set()
    for (u, v), _ in edges:
        node_set.add(u)
        node_set.add(v)
    nodes = sorted(node_set)
    if len(nodes) == 0:
        return [], set(), {"edges": 0, "pos": 0, "neg": 0, "nodes": 0}
    # ノード抽出
    rng = random.Random(seed)
    k = max(1, int(len(nodes) * pct))
    sampled_nodes = set(rng.sample(nodes, k))
    # エッジフィルタ
    filtered: List[Edge] = []
    pos = neg = 0
    for (u, v), y in edges:
        if u in sampled_nodes and v in sampled_nodes:
            filtered.append(((u, v), y))
            if y == 1:
                pos += 1
            else:
                neg += 1
    stats = {"edges": len(filtered), "pos": pos, "neg": neg, "nodes": len(sampled_nodes)}
    return filtered, sampled_nodes, stats


def qc_minimum_counts(
    edges: List[Edge],
    min_positive: int,
    min_negative: int
) -> bool:
    """最小正負件数のQCを満たすかを返す。"""
    pos = sum(1 for _, y in edges if y == 1)
    neg = sum(1 for _, y in edges if y == 0)
    return pos >= min_positive and neg >= min_negative



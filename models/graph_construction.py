"""
Plain, homogeneous ABA graph construction.

Builds a single-node-type, single-edge-type graph from the same
hotel_contrary_dataset_support.csv used by the KGE models:
  - nodes  = unique ABA entities (heads/tails)
  - edges  = every (head, tail) pair, undirected, relation label dropped
             (that's what makes it "plain" / homogeneous — GCN/GAT/GraphSAGE
             see one edge type; the 3-way relation label is only the thing
             being predicted, not part of the message-passing structure)

The train/val/test split is the identical stratified-by-(domain, relation)
split used by the KGE + LR scripts (same ratios, same seed), so downstream
metrics are comparable across all model families.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "input_data" / "hotel_contrary_dataset_support.csv"

RELATIONS = ["CONTRARY_TO", "NOT_CONTRARY", "SUPPORT"]
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}


def stratified_split(df, train_ratio=0.8, val_ratio=0.1, random_seed=42):
    """Identical to the split used in sup_*.py / lr_baseline.py."""
    train_dfs, val_dfs, test_dfs = [], [], []

    for (domain, relation), group in df.groupby(["domain", "relation"]):
        n = len(group)
        if n < 10:
            train_dfs.append(group)
            continue

        train_val, test = train_test_split(
            group, test_size=1 - train_ratio - val_ratio, random_state=random_seed,
        )
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_size, random_state=random_seed)

        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

    df_train = pd.concat(train_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_val = pd.concat(val_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_test = pd.concat(test_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return df_train, df_val, df_test


class ABAHomogeneousGraph:
    """Plain homogeneous graph + train/val/test relation-classification edges."""

    def __init__(self, csv_path: Path = CSV_PATH, random_seed: int = 42):
        self.random_seed = random_seed

        df = pd.read_csv(csv_path)
        df = df.drop_duplicates(subset=["head", "relation", "tail"]).reset_index(drop=True)
        self.df = df

        # ── node vocabulary ──────────────────────────────────────────────
        entities = sorted(set(df["head"]) | set(df["tail"]))
        self.entity_to_id = {e: i for i, e in enumerate(entities)}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.num_nodes = len(entities)

        # ── message-passing edges: every triple, both directions, no relation label ──
        src = df["head"].map(self.entity_to_id).to_numpy()
        dst = df["tail"].map(self.entity_to_id).to_numpy()
        edge_index = np.vstack([
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ])
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        # ── one-hot node features (matches Takashima R-GCN's default input scheme) ──
        self.node_features = torch.eye(self.num_nodes, dtype=torch.float32)

        # ── stratified split (identical to KGE / LR scripts) ─────────────
        self.df_train, self.df_val, self.df_test = stratified_split(
            df, random_seed=random_seed
        )

    def split_tensors(self, split_df: pd.DataFrame):
        """Return (head_idx, tail_idx, label_idx) tensors for a split dataframe."""
        head_idx = torch.tensor(split_df["head"].map(self.entity_to_id).to_numpy(), dtype=torch.long)
        tail_idx = torch.tensor(split_df["tail"].map(self.entity_to_id).to_numpy(), dtype=torch.long)
        label_idx = torch.tensor(split_df["relation"].map(REL_TO_ID).to_numpy(), dtype=torch.long)
        return head_idx, tail_idx, label_idx

    def class_weights(self):
        """Sqrt inverse-frequency weights over df_train — softer than LR's fully
        balanced weighting, which made gradient-descent training oscillate
        (SUPPORT is ~100x rarer than NOT_CONTRARY)."""
        counts = self.df_train["relation"].value_counts()
        n = len(self.df_train)
        k = len(RELATIONS)
        w = torch.tensor(
            [(n / (k * counts.get(r, 1))) ** 0.5 for r in RELATIONS], dtype=torch.float32
        )
        return w

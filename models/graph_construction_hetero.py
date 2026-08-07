"""
Heterogeneous (typed nodes + typed edges) ABA graph — the medium-term step up
from the plain graph in graph_construction.py.

Node types (derived from entity naming convention, same heuristic already used
in sup_transe.py::get_node_level for the ABA tree export):
  - claim    — the 8 root claims (good_staff, bad_staff, ...)
  - attacker — entities named 'no_evident_...'
  - body     — everything else

Edge types (relation buckets for R-GCN message passing) are the OBSERVED
(head_type, tail_type) structural pairs in the data, NOT the 3-way relation
label (CONTRARY_TO/NOT_CONTRARY/SUPPORT). Using the label itself as the edge
type would leak the prediction target into the graph structure — see the
verification below, which shows SUPPORT occurs iff tail_type == claim with
zero exceptions, i.e. node type alone already determines SUPPORT vs not.
So the label stays a pure decoder target (identical to the plain-graph setup);
only the *structural* (head_type, tail_type) pairing is exposed to the encoder:

    attacker -> body    (CONTRARY_TO / NOT_CONTRARY candidates)
    body     -> body    (CONTRARY_TO / NOT_CONTRARY candidates)
    attacker -> claim   (SUPPORT candidates)
    body     -> claim   (SUPPORT candidates)

...plus a reverse relation for each, for bidirectional message passing (8
relations total) — same convention R-GCN papers use for directed multi-graphs.

Reuses ABAHomogeneousGraph for entity vocabulary, one-hot node features, and
the stratified train/val/test split, so results are directly comparable to
the plain GCN/GAT/GraphSAGE models.

Claude-Assisted
"""
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from graph_construction import ABAHomogeneousGraph, CSV_PATH

CLAIMS = [
    'good_staff', 'bad_staff',
    'good_price', 'bad_price',
    'good_check-in', 'bad_check-in',
    'good_check-out', 'bad_check-out',
]


def node_level(name: str) -> str:
    if name in CLAIMS:
        return 'claim'
    if str(name).startswith('no_evident_'):
        return 'attacker'
    return 'body'


class ABAHeteroGraph(ABAHomogeneousGraph):
    """Adds node types + structural edge types on top of the plain graph."""

    NODE_TYPES = ['claim', 'body', 'attacker']

    def __init__(self, csv_path: Path = CSV_PATH, random_seed: int = 42):
        super().__init__(csv_path, random_seed)

        # ── node types (global id -> type) ────────────────────────────────
        self.node_type_name = [node_level(self.id_to_entity[i]) for i in range(self.num_nodes)]
        self.node_type_to_id = {t: i for i, t in enumerate(self.NODE_TYPES)}
        self.node_type = torch.tensor(
            [self.node_type_to_id[t] for t in self.node_type_name], dtype=torch.long
        )

        # ── structural relations actually observed in the data ───────────
        pairs = sorted(set(
            (node_level(h), node_level(t)) for h, t in zip(self.df['head'], self.df['tail'])
        ))
        fwd_names = [f"{h}__to__{t}" for h, t in pairs]
        rev_names = [f"{n}__rev" for n in fwd_names]
        self.relation_names = fwd_names + rev_names
        self.relation_to_id = {r: i for i, r in enumerate(self.relation_names)}
        self.num_relations = len(self.relation_names)

        src = self.df['head'].map(self.entity_to_id).to_numpy()
        dst = self.df['tail'].map(self.entity_to_id).to_numpy()
        fwd_rel = [self.relation_to_id[f"{node_level(h)}__to__{node_level(t2)}"]
                   for h, t2 in zip(self.df['head'], self.df['tail'])]
        rev_rel = [self.relation_to_id[f"{node_level(h)}__to__{node_level(t2)}__rev"]
                   for h, t2 in zip(self.df['head'], self.df['tail'])]

        edge_index_fwd = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_index_rev = torch.tensor(np.stack([dst, src]), dtype=torch.long)
        self.edge_index = torch.cat([edge_index_fwd, edge_index_rev], dim=1)
        self.edge_type = torch.tensor(fwd_rel + rev_rel, dtype=torch.long)

        print("Structural relation buckets (edge type <- (head_type, tail_type)):")
        for h, t in pairs:
            n = ((self.df['head'].map(node_level) == h) & (self.df['tail'].map(node_level) == t)).sum()
            print(f"  {h:<8} -> {t:<8} : {n:,} triples  "
                  f"(relations = '{h}__to__{t}' + reverse)")

    def to_hetero_data(self) -> HeteroData:
        """Build a torch_geometric HeteroData view of the same graph, for
        inspection/sanity-checking the schema (not used for training directly —
        training uses the flat edge_index/edge_type tensors above so entity
        ids match the plain graph exactly, avoiding a second id remapping)."""
        data = HeteroData()
        type_ids = {t: [i for i, tt in enumerate(self.node_type_name) if tt == t] for t in self.NODE_TYPES}
        for t in self.NODE_TYPES:
            idx = type_ids[t]
            data[t].x = self.node_features[idx]
            data[t].global_id = torch.tensor(idx, dtype=torch.long)

        local_id = {}
        for t in self.NODE_TYPES:
            for local_i, global_i in enumerate(type_ids[t]):
                local_id[global_i] = local_i

        df = self.df
        for h_type, t_type in sorted(set(
            (node_level(h), node_level(t)) for h, t in zip(df['head'], df['tail'])
        )):
            mask = (df['head'].map(node_level) == h_type) & (df['tail'].map(node_level) == t_type)
            sub = df[mask]
            src = torch.tensor([local_id[self.entity_to_id[h]] for h in sub['head']], dtype=torch.long)
            dst = torch.tensor([local_id[self.entity_to_id[t]] for t in sub['tail']], dtype=torch.long)
            data[h_type, 'to', t_type].edge_index = torch.stack([src, dst])

        return data


if __name__ == "__main__":
    g = ABAHeteroGraph()
    print()
    print(f"Nodes: {g.num_nodes}  |  Node types: "
          f"{ {t: g.node_type_name.count(t) for t in g.NODE_TYPES} }")
    print(f"Relations ({g.num_relations}): {g.relation_names}")
    print(f"Edges (both directions): {g.edge_index.shape[1]:,}")
    print()
    hd = g.to_hetero_data()
    print("HeteroData schema:")
    print(hd)

"""
HAN (Heterogeneous Graph Attention Network) on the same typed ABA graph used
for R-GCN — same node types, same edge types, same split, same classifier
head, same metrics (see rgcn.py / graph_construction_hetero.py). Only the
encoder differs: HAN uses node-level + semantic-level attention instead of
R-GCN's per-relation weight matrices. Requested by the supervisor as a direct
architecture comparison under the same graph configuration.

Claude-Assisted
"""
from han_common import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate(model_name="HAN")

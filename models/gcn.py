"""
GCN on the plain, homogeneous ABA graph — 3-class relation classification
(CONTRARY_TO, NOT_CONTRARY, SUPPORT). See graph_construction.py for the graph
and gnn_common.py for the shared training/eval loop.

Claude-Assisted
"""
from torch_geometric.nn import GCNConv

from gnn_common import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate(model_name="GCN", conv_cls=GCNConv)

"""
GAT on the plain, homogeneous ABA graph — 3-class relation classification
(CONTRARY_TO, NOT_CONTRARY, SUPPORT). See graph_construction.py for the graph
and gnn_common.py for the shared training/eval loop.

Claude-Assisted
"""
from torch_geometric.nn import GATConv

from gnn_common import train_and_evaluate

if __name__ == "__main__":
    # concat=False keeps the per-layer output at hidden_dim regardless of head count,
    # so it can be stacked with BatchNorm1d the same way as GCN/GraphSAGE.
    train_and_evaluate(
        model_name="GAT",
        conv_cls=GATConv,
        conv_kwargs=dict(heads=4, concat=False, dropout=0.3),
    )

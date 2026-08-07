"""
R-GCN on the heterogeneous ABA graph (typed nodes + typed edges) — 3-class
relation classification (CONTRARY_TO, NOT_CONTRARY, SUPPORT). See
graph_construction_hetero.py for the graph and rgcn_common.py for the
training/eval loop.

Claude-Assisted
"""
from rgcn_common import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate(model_name="RGCN")

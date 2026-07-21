"""
pytest設定とフィクスチャ
"""

import pytest
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data


@pytest.fixture
def mock_graph():
    """モックグラフを作成"""
    G = nx.DiGraph()
    
    # ノードを追加
    nodes = [
        "良い部屋でした",
        "清潔感があります",
        "スタッフが親切",
        "朝食が美味しい",
        "Wi-Fiが遅い",
        "エアコンが古い"
    ]
    
    for node in nodes:
        G.add_node(node)
    
    # Inferenceエッジを追加
    G.add_edge("良い部屋でした", "清潔感があります", relation='inference')
    G.add_edge("スタッフが親切", "朝食が美味しい", relation='inference')
    
    # Attackエッジを追加
    G.add_edge("Wi-Fiが遅い", "良い部屋でした", relation='attack')
    G.add_edge("エアコンが古い", "清潔感があります", relation='attack')
    
    return G


@pytest.fixture
def mock_embeddings():
    """モックエンベディングを作成"""
    nodes = [
        "良い部屋でした",
        "清潔感があります",
        "スタッフが親切",
        "朝食が美味しい",
        "Wi-Fiが遅い",
        "エアコンが古い"
    ]
    
    # ランダムな768次元ベクトル
    embeddings = {node: np.random.randn(768).astype(np.float32) for node in nodes}
    return embeddings


@pytest.fixture
def mock_attack_edges():
    """モックAttackエッジを作成"""
    return [
        ("Wi-Fiが遅い", "良い部屋でした"),
        ("エアコンが古い", "清潔感があります")
    ]


@pytest.fixture
def mock_negative_edges():
    """モックネガティブエッジを作成"""
    return [
        ("朝食が美味しい", "Wi-Fiが遅い"),
        ("スタッフが親切", "エアコンが古い")
    ]


@pytest.fixture
def mock_pytorch_geometric_data():
    """モックPyTorch Geometricデータを作成"""
    # 6ノード、4エッジ
    x = torch.randn(6, 768)  # 6ノード、768次元特徴量
    
    edge_index = torch.tensor([
        [0, 1, 2, 3],  # source nodes
        [1, 0, 3, 2]   # target nodes
    ], dtype=torch.long)
    
    edge_attr = torch.zeros(4, dtype=torch.long)  # すべてinference (0)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


@pytest.fixture
def mock_node_to_idx():
    """モックノードインデックスマッピングを作成"""
    nodes = [
        "良い部屋でした",
        "清潔感があります",
        "スタッフが親切",
        "朝食が美味しい",
        "Wi-Fiが遅い",
        "エアコンが古い"
    ]
    return {node: i for i, node in enumerate(nodes)}


@pytest.fixture
def mock_cv_splits(mock_attack_edges, mock_negative_edges):
    """モックCross-validation分割を作成"""
    # ラベル付きエッジを作成
    all_edges = [(edge, 1) for edge in mock_attack_edges] + \
                [(edge, 0) for edge in mock_negative_edges]
    
    # 2-fold分割
    mid = len(all_edges) // 2
    
    fold1_test = all_edges[:mid]
    fold1_train = all_edges[mid:]
    
    fold2_test = all_edges[mid:]
    fold2_train = all_edges[:mid]
    
    return [(fold1_train, fold1_test), (fold2_train, fold2_test)]


@pytest.fixture
def mock_config():
    """モック設定を作成"""
    return {
        'data': {
            'input_graph': 'data/output/aba_graph_room_combined.pkl',
            'base_output_dir': 'data/training_results',
            'experiment_id': 'test_experiment',
            'output_dir': 'data/training_results/test_experiment',  # テスト用の固定パス
            'seed': 42
        },
        'negative_sampling': {
            'hard_negatives': {'enabled': True, 'ratio': 0.4},
            'structural_negatives': {'enabled': True, 'ratio': 0.3},
            'random_negatives': {'enabled': True, 'ratio': 0.3}
        },
        'cross_validation': {
            'n_folds': 2,
            'split_strategy': 'edge_level_random',
            'val_split_ratio': 0.2
        },
        'models': {
            'rgcn': {
                'enabled': True,
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1,
                'learning_rate': 0.001,
                'num_epochs': 10,
                'verbose': False
            },
            'improved_bert': {'enabled': False},
            'cross_encoder_bert': {'enabled': False},
            'random_baseline': {'enabled': True},
            'bert_cosine': {'enabled': True},
            'tfidf_lr': {'enabled': True}
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc']
        },
        'visualization': {
            'enabled': False,
            'save_plots': False,
            'show_plots': False
        },
        'compute': {
            'device': 'cpu'
        }
    }


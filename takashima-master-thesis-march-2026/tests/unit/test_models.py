"""
モデル定義のユニットテスト
"""

import pytest
import torch
import numpy as np

from src.model_defs.models import (
    AttackLinkPredictor,
    RandomBaseline,
    BERTCosineSimilarityBaseline,
    TFIDFLogisticRegressionBaseline
)


class TestAttackLinkPredictor:
    """AttackLinkPredictorのテスト"""
    
    def test_initialization(self):
        """モデルの初期化をテスト"""
        model = AttackLinkPredictor(input_dim=768, hidden_dim=128, num_layers=2)
        assert model is not None
        assert model.num_layers == 2
    
    def test_forward_pass(self, mock_pytorch_geometric_data):
        """順伝播をテスト"""
        model = AttackLinkPredictor(input_dim=768, hidden_dim=128, num_layers=2)
        
        # エッジペアを作成
        edge_pairs = [(0, 1), (2, 3)]
        
        # 順伝播
        predictions = model(
            mock_pytorch_geometric_data.x,
            mock_pytorch_geometric_data.edge_index,
            mock_pytorch_geometric_data.edge_attr,
            edge_pairs
        )
        
        # 予測の形状を確認
        assert predictions.shape == (2,)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)
    
    def test_parameter_count(self):
        """パラメータ数をテスト"""
        model = AttackLinkPredictor(input_dim=768, hidden_dim=128, num_layers=2)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0


class TestRandomBaseline:
    """RandomBaselineのテスト"""
    
    def test_predict(self):
        """予測をテスト"""
        baseline = RandomBaseline()
        edge_pairs = [('a', 'b'), ('c', 'd'), ('e', 'f')]
        
        predictions = baseline.predict(edge_pairs)
        
        assert len(predictions) == 3
        assert np.all(predictions >= 0) and np.all(predictions <= 1)


class TestBERTCosineSimilarityBaseline:
    """BERTCosineSimilarityBaselineのテスト"""
    
    def test_predict(self, mock_embeddings, mock_node_to_idx):
        """予測をテスト"""
        baseline = BERTCosineSimilarityBaseline(mock_embeddings, mock_node_to_idx)
        
        edge_pairs = [
            ("良い部屋でした", "清潔感があります"),
            ("スタッフが親切", "朝食が美味しい")
        ]
        
        predictions = baseline.predict(edge_pairs)
        
        assert len(predictions) == 2
        # コサイン類似度は-1から1の範囲
        assert np.all(predictions >= -1) and np.all(predictions <= 1)


class TestTFIDFLogisticRegressionBaseline:
    """TFIDFLogisticRegressionBaselineのテスト"""
    
    def test_fit_and_predict(self, mock_attack_edges, mock_negative_edges):
        """学習と予測をテスト"""
        baseline = TFIDFLogisticRegressionBaseline()
        
        # 訓練データを作成
        train_edges = [(edge, 1) for edge in mock_attack_edges] + \
                     [(edge, 0) for edge in mock_negative_edges]
        
        all_nodes = list(set([node for edge, _ in train_edges for node in edge]))
        
        # 学習
        baseline.fit(train_edges, all_nodes)
        
        # 予測
        test_edges = mock_attack_edges[:1]
        predictions = baseline.predict(test_edges)
        
        assert len(predictions) == 1
        assert np.all(predictions >= 0) and np.all(predictions <= 1)


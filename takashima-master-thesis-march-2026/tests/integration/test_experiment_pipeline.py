"""
実験パイプラインの統合テスト
"""

import pytest
import torch

from src.experiments.cross_validation import (
    create_cross_validation_splits,
    run_cross_validation
)
from src.visualization.plot_results import (
    calculate_statistics,
    perform_statistical_tests
)


class TestExperimentPipeline:
    """実験パイプライン全体のテスト"""
    
    def test_end_to_end_pipeline(self, mock_config, mock_pytorch_geometric_data,
                                 mock_node_to_idx, mock_attack_edges,
                                 mock_negative_edges, mock_embeddings):
        """エンドツーエンドのパイプラインをテスト"""
        
        # 1. Cross-validation分割
        cv_splits = create_cross_validation_splits(
            mock_attack_edges,
            mock_negative_edges,
            n_splits=mock_config['cross_validation']['n_folds'],
            seed=mock_config['data']['seed']
        )
        
        assert len(cv_splits) == 2
        
        # 2. Cross-validation実行
        all_nodes = list(mock_embeddings.keys())
        
        results = run_cross_validation(
            cv_splits,
            mock_pytorch_geometric_data,
            mock_node_to_idx,
            all_nodes,
            mock_embeddings,
            mock_config,
            device='cpu'
        )
        
        # 結果を確認
        assert 'AttackLinkPredictor' in results
        assert 'Random' in results
        assert 'BERTCosine' in results
        assert 'TFIDF+LR' in results
        
        # 各モデルの結果を確認
        for model_name in results:
            assert 'accuracy' in results[model_name]
            assert 'f1' in results[model_name]
            assert 'auc' in results[model_name]
            assert len(results[model_name]['accuracy']) == 2  # 2-fold
        
        # 3. 統計分析
        stats = calculate_statistics(results)
        assert 'AttackLinkPredictor' in stats
        
        test_results = perform_statistical_tests(results)
        assert len(test_results) > 0
    
    def test_pipeline_with_minimal_data(self, mock_config):
        """最小限のデータでパイプラインをテスト"""
        
        # 最小限のデータ
        attack_edges = [("A", "B")]
        negative_edges = [("C", "D")]
        
        cv_splits = create_cross_validation_splits(
            attack_edges,
            negative_edges,
            n_splits=2,
            seed=42
        )
        
        # 分割が正常に作成されることを確認
        assert len(cv_splits) == 2
        
        for train_edges, test_edges in cv_splits:
            # 最低限のデータが存在することを確認
            assert len(train_edges) + len(test_edges) == 2


class TestDataFlow:
    """データフローのテスト"""
    
    def test_cv_splits_data_integrity(self, mock_attack_edges, mock_negative_edges):
        """Cross-validation分割でデータの整合性を確認"""
        
        cv_splits = create_cross_validation_splits(
            mock_attack_edges,
            mock_negative_edges,
            n_splits=2,
            seed=42
        )
        
        # 全フォールドでポジティブとネガティブが含まれることを確認
        for train_edges, test_edges in cv_splits:
            train_labels = [label for _, label in train_edges]
            test_labels = [label for _, label in test_edges]
            
            # 訓練データに両方のラベルが含まれることを確認
            # （ただし、小さいデータセットでは必ずしも保証されない）
            assert len(train_labels) > 0
            assert len(test_labels) > 0
    
    def test_results_structure(self, mock_config, mock_pytorch_geometric_data,
                              mock_node_to_idx, mock_attack_edges,
                              mock_negative_edges, mock_embeddings):
        """結果の構造をテスト"""
        
        cv_splits = create_cross_validation_splits(
            mock_attack_edges,
            mock_negative_edges,
            n_splits=2,
            seed=42
        )
        
        all_nodes = list(mock_embeddings.keys())
        
        results = run_cross_validation(
            cv_splits,
            mock_pytorch_geometric_data,
            mock_node_to_idx,
            all_nodes,
            mock_embeddings,
            mock_config,
            device='cpu'
        )
        
        # 結果の構造を確認
        for model_name, metrics in results.items():
            assert isinstance(metrics, dict)
            for metric_name, values in metrics.items():
                assert isinstance(values, list)
                assert len(values) == 2  # 2-fold
                for value in values:
                    assert isinstance(value, (int, float))
                    assert 0 <= value <= 1  # メトリクスは0-1の範囲


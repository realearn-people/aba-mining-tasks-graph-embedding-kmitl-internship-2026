"""
Cross-validationのユニットテスト
"""

import pytest

from src.experiments.cross_validation import create_cross_validation_splits


class TestCrossValidationSplits:
    """Cross-validation分割のテスト"""
    
    def test_create_splits(self, mock_attack_edges, mock_negative_edges):
        """分割の作成をテスト"""
        cv_splits = create_cross_validation_splits(
            mock_attack_edges,
            mock_negative_edges,
            n_splits=2,
            seed=42
        )
        
        # 分割数を確認
        assert len(cv_splits) == 2
        
        # 各フォールドを確認
        for train_edges, test_edges in cv_splits:
            assert len(train_edges) > 0
            assert len(test_edges) > 0
            
            # ラベルの確認
            for edge, label in train_edges:
                assert label in [0, 1]
            
            for edge, label in test_edges:
                assert label in [0, 1]
    
    def test_no_overlap(self, mock_attack_edges, mock_negative_edges):
        """訓練データとテストデータが重複していないことを確認"""
        cv_splits = create_cross_validation_splits(
            mock_attack_edges,
            mock_negative_edges,
            n_splits=2,
            seed=42
        )
        
        for train_edges, test_edges in cv_splits:
            train_set = set([edge for edge, _ in train_edges])
            test_set = set([edge for edge, _ in test_edges])
            
            # 重複がないことを確認
            assert len(train_set & test_set) == 0
    
    def test_all_data_used(self, mock_attack_edges, mock_negative_edges):
        """全データが使用されることを確認"""
        total_edges = len(mock_attack_edges) + len(mock_negative_edges)
        
        cv_splits = create_cross_validation_splits(
            mock_attack_edges,
            mock_negative_edges,
            n_splits=2,
            seed=42
        )
        
        for train_edges, test_edges in cv_splits:
            # 訓練+テストが全データに等しいことを確認
            assert len(train_edges) + len(test_edges) == total_edges


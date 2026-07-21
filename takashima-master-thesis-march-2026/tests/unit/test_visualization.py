"""
可視化・統計分析のユニットテスト
"""

import pytest
import numpy as np

from src.visualization.plot_results import (
    calculate_statistics,
    perform_statistical_tests
)


class TestStatistics:
    """統計計算のテスト"""
    
    def test_calculate_statistics(self):
        """統計量の計算をテスト"""
        results = {
            'Model1': {
                'accuracy': [0.8, 0.85, 0.82],
                'f1': [0.75, 0.78, 0.76],
                'auc': [0.88, 0.90, 0.89]
            },
            'Model2': {
                'accuracy': [0.70, 0.72, 0.71],
                'f1': [0.65, 0.68, 0.66],
                'auc': [0.78, 0.80, 0.79]
            }
        }
        
        stats = calculate_statistics(results)
        
        # Model1の統計量を確認
        assert 'Model1' in stats
        assert 'accuracy' in stats['Model1']
        assert abs(stats['Model1']['accuracy']['mean'] - 0.823) < 0.01
        assert stats['Model1']['accuracy']['std'] > 0
    
    def test_empty_results(self):
        """空の結果の処理をテスト"""
        results = {
            'Model1': {
                'accuracy': [],
                'f1': [],
                'auc': []
            }
        }
        
        stats = calculate_statistics(results)
        
        # 空の場合はすべて0
        assert stats['Model1']['accuracy']['mean'] == 0.0
        assert stats['Model1']['accuracy']['std'] == 0.0


class TestStatisticalTests:
    """統計的検定のテスト"""
    
    def test_perform_statistical_tests(self):
        """統計的検定の実行をテスト"""
        results = {
            'Model1': {
                'accuracy': [0.8, 0.85, 0.82, 0.84, 0.83],
                'f1': [0.75, 0.78, 0.76, 0.77, 0.76],
                'auc': [0.88, 0.90, 0.89, 0.91, 0.90]
            },
            'Model2': {
                'accuracy': [0.70, 0.72, 0.71, 0.73, 0.72],
                'f1': [0.65, 0.68, 0.66, 0.69, 0.67],
                'auc': [0.78, 0.80, 0.79, 0.81, 0.80]
            }
        }
        
        test_results = perform_statistical_tests(results)
        
        # 比較結果を確認
        assert 'Model1_vs_Model2' in test_results
        assert 'accuracy' in test_results['Model1_vs_Model2']
        assert 'p_value' in test_results['Model1_vs_Model2']['accuracy']
        assert 'significant' in test_results['Model1_vs_Model2']['accuracy']
    
    def test_insufficient_data(self):
        """データ不足の場合の処理をテスト"""
        results = {
            'Model1': {
                'accuracy': [0.8],
                'f1': [0.75],
                'auc': [0.88]
            },
            'Model2': {
                'accuracy': [0.70],
                'f1': [0.65],
                'auc': [0.78]
            }
        }
        
        test_results = perform_statistical_tests(results)
        
        # データ不足の場合はNaN
        assert np.isnan(test_results['Model1_vs_Model2']['accuracy']['p_value'])
        assert test_results['Model1_vs_Model2']['accuracy']['significant'] == False


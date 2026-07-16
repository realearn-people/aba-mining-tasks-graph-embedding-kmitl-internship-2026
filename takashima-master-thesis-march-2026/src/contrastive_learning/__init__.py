"""
対照学習モジュール

Attack Link Predictionのための対照学習実装
"""

from .pretrain_embeddings import (
    AttackContrastiveLearning,
    pretrain_embeddings_with_contrastive_learning
)
from .embedding_quality import (
    evaluate_embedding_quality,
    visualize_embeddings_tsne,
    calculate_separation_score,
    calculate_clustering_score,
    plot_similarity_distributions
)

__all__ = [
    'AttackContrastiveLearning',
    'pretrain_embeddings_with_contrastive_learning',
    'evaluate_embedding_quality',
    'visualize_embeddings_tsne',
    'calculate_separation_score',
    'calculate_clustering_score',
    'plot_similarity_distributions'
]


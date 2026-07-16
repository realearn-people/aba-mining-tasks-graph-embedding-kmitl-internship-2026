import os
import json
import numpy as np
import pytest

from src.contrastive_learning import (
    AttackContrastiveLearning,
    pretrain_embeddings_with_contrastive_learning,
    evaluate_embedding_quality,
    visualize_embeddings_tsne,
    calculate_separation_score,
    calculate_clustering_score,
    plot_similarity_distributions,
)


def _make_contrastive_config(num_epochs: int = 10, output_dim: int = 16, hidden_dim: int = 32, lr: float = 1e-3, batch_size: int = 8):
    return {
        'contrastive_learning': {
            'enabled': True,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'temperature': 0.07,
            'dropout': 0.1,
            'num_epochs': num_epochs,
            'learning_rate': lr,
            'batch_size': batch_size,
            'evaluate_quality': False,
        },
        'data': {
            'seed': 123,
        },
        'compute': {
            'device': 'cpu'
        }
    }


def test_attack_contrastive_learning_improves_similarity_gap(mock_embeddings, mock_attack_edges, mock_negative_edges, mock_node_to_idx):
    nodes_sorted = sorted(mock_node_to_idx.keys())
    embedding_matrix = np.array([mock_embeddings[n] for n in nodes_sorted])

    model = AttackContrastiveLearning(
        input_dim=embedding_matrix.shape[1], hidden_dim=32, output_dim=16, dropout=0.0
    )

    # 初期埋め込みをエンコードして基準メトリクスを測定
    import torch
    with torch.no_grad():
        node_emb = model(torch.tensor(embedding_matrix, dtype=torch.float32))
    attack_pairs_idx = [(mock_node_to_idx[u], mock_node_to_idx[v]) for u, v in mock_attack_edges]
    non_attack_pairs_idx = [(mock_node_to_idx[u], mock_node_to_idx[v]) for u, v in mock_negative_edges]
    loss0, metrics0 = model.compute_loss(node_emb, attack_pairs_idx, non_attack_pairs_idx)

    # 簡単な学習を行い、ギャップの改善を期待
    optim = __import__('torch').optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        node_emb = model(torch.tensor(embedding_matrix, dtype=torch.float32))
        loss, _ = model.compute_loss(node_emb, attack_pairs_idx, non_attack_pairs_idx)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        node_emb = model(torch.tensor(embedding_matrix, dtype=torch.float32))
    _, metrics1 = model.compute_loss(node_emb, attack_pairs_idx, non_attack_pairs_idx)

    assert np.isfinite(loss0.item())
    assert set(metrics0.keys()) == {
        'attack_loss', 'non_attack_loss', 'attack_avg_sim', 'non_attack_avg_sim', 'attack_std_sim', 'non_attack_std_sim'
    }
    gap0 = metrics0['non_attack_avg_sim'] - metrics0['attack_avg_sim']
    gap1 = metrics1['non_attack_avg_sim'] - metrics1['attack_avg_sim']
    assert gap1 >= gap0 - 1e-6  # 改善または維持


def test_pretrain_embeddings_with_contrastive_learning_saves_and_is_deterministic(tmp_path, mock_embeddings, mock_attack_edges, mock_negative_edges, mock_node_to_idx):
    config = _make_contrastive_config(num_epochs=5, output_dim=16, hidden_dim=32)

    out_dir = tmp_path / 'contrastive_artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)

    optimized1, history1 = pretrain_embeddings_with_contrastive_learning(
        initial_embeddings=mock_embeddings,
        attack_edges=mock_attack_edges,
        non_attack_edges=mock_negative_edges,
        node_to_idx=mock_node_to_idx,
        config=config,
        device='cpu',
        verbose=False,
        output_dir=str(out_dir),
        seed=123,
        deterministic=True,
        save_artifacts=True,
    )

    # 成果物が保存されていること
    assert (out_dir / 'embeddings_after_contrastive.npz').exists()
    assert (out_dir / 'contrastive_history.json').exists()
    assert (out_dir / 'contrastive_model.pt').exists()

    # 再実行で同一結果（決定論性）
    optimized2, history2 = pretrain_embeddings_with_contrastive_learning(
        initial_embeddings=mock_embeddings,
        attack_edges=mock_attack_edges,
        non_attack_edges=mock_negative_edges,
        node_to_idx=mock_node_to_idx,
        config=config,
        device='cpu',
        verbose=False,
        output_dir=None,
        seed=123,
        deterministic=True,
        save_artifacts=False,
    )

    for n in mock_node_to_idx.keys():
        assert np.allclose(optimized1[n], optimized2[n])
    assert len(history1['loss']) == len(history2['loss'])


def test_evaluate_embedding_quality_outputs(mock_embeddings, mock_attack_edges, mock_negative_edges, mock_node_to_idx):
    # t-SNE を回避するため output_dir=None で評価（小規模データでは既定perplexityが不適）
    results = evaluate_embedding_quality(
        embeddings=mock_embeddings,
        attack_edges=mock_attack_edges,
        non_attack_edges=mock_negative_edges,
        node_to_idx=mock_node_to_idx,
        output_dir=None,
        verbose=False,
    )

    assert 'separation' in results
    assert 'clustering_score' in results
    sep = results['separation']
    for k in [
        'attack_mean_similarity', 'non_attack_mean_similarity', 'similarity_gap',
        'attack_std_similarity', 'non_attack_std_similarity',
        'attack_min_similarity', 'attack_max_similarity',
        'non_attack_min_similarity', 'non_attack_max_similarity'
    ]:
        assert k in sep


def test_visualize_embeddings_tsne_saves_image(tmp_path, mock_embeddings, mock_attack_edges, mock_node_to_idx):
    # 小規模データに適したperplexityを指定
    out_file = tmp_path / 'tsne.png'
    fig = visualize_embeddings_tsne(
        embeddings=mock_embeddings,
        attack_edges=mock_attack_edges,
        node_to_idx=mock_node_to_idx,
        save_path=str(out_file),
        title='test tsne',
        perplexity=2,
        random_state=42,
    )
    assert out_file.exists()
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_similarity_distributions_saves_image(tmp_path):
    attack = np.random.uniform(-1, 0.2, size=100)
    non_attack = np.random.uniform(-0.2, 1, size=100)
    out_file = tmp_path / 'dist.png'
    fig = plot_similarity_distributions(attack, non_attack, save_path=str(out_file), title='dist')
    assert out_file.exists()
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)


def test_calculate_separation_and_clustering_scores(mock_embeddings, mock_attack_edges, mock_negative_edges, mock_node_to_idx):
    nodes_sorted = sorted(mock_node_to_idx.keys())
    X = np.array([mock_embeddings[n] for n in nodes_sorted])
    attack_pairs_idx = [(mock_node_to_idx[u], mock_node_to_idx[v]) for u, v in mock_attack_edges]
    non_attack_pairs_idx = [(mock_node_to_idx[u], mock_node_to_idx[v]) for u, v in mock_negative_edges]

    sep = calculate_separation_score(X, attack_pairs_idx, non_attack_pairs_idx)
    assert isinstance(sep['similarity_gap'], float)

    # Attack含有ノードを1とする2値ラベル
    attack_nodes = set()
    for u, v in mock_attack_edges:
        attack_nodes.add(u)
        attack_nodes.add(v)
    labels = np.array([1 if n in attack_nodes else 0 for n in nodes_sorted])
    score = calculate_clustering_score(X, labels, metric='cosine')
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0



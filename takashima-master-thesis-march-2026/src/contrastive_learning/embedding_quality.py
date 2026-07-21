"""
埋め込み品質評価モジュール

対照学習後の埋め込み空間の品質を評価
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import os


def calculate_clustering_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = 'cosine'
) -> float:
    """
    Silhouette係数を計算（クラスタリング品質評価）
    
    Args:
        embeddings: ノード埋め込み [num_nodes, embedding_dim]
        labels: クラスラベル [num_nodes]（0: 非Attack, 1: Attack）
        metric: 距離メトリック
        
    Returns:
        silhouette_score: -1（悪い）から+1（良い）
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    score = silhouette_score(embeddings, labels, metric=metric)
    # numpy.float ではなく Python float を返す
    return float(score)


def calculate_separation_score(
    embeddings: np.ndarray,
    attack_pairs: List[Tuple[int, int]],
    non_attack_pairs: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    Attack/非Attack関係の埋め込み空間での分離度を計算
    
    Args:
        embeddings: ノード埋め込み [num_nodes, embedding_dim]
        attack_pairs: Attack関係のペアインデックス
        non_attack_pairs: 非Attack関係のペアインデックス
        
    Returns:
        分離スコア辞書
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Attack関係の類似度
    attack_similarities = []
    for u_idx, v_idx in attack_pairs:
        u_emb = embeddings[u_idx].reshape(1, -1)
        v_emb = embeddings[v_idx].reshape(1, -1)
        sim = cosine_similarity(u_emb, v_emb)[0, 0]
        attack_similarities.append(sim)
    
    # 非Attack関係の類似度
    non_attack_similarities = []
    for u_idx, v_idx in non_attack_pairs:
        u_emb = embeddings[u_idx].reshape(1, -1)
        v_emb = embeddings[v_idx].reshape(1, -1)
        sim = cosine_similarity(u_emb, v_emb)[0, 0]
        non_attack_similarities.append(sim)
    
    attack_sim_array = np.array(attack_similarities)
    non_attack_sim_array = np.array(non_attack_similarities)
    
    # 統計量計算（明示的にPython floatに変換）
    separation_metrics = {
        'attack_mean_similarity': float(np.mean(attack_sim_array).item()),
        'attack_std_similarity': float(np.std(attack_sim_array).item()),
        'attack_min_similarity': float(np.min(attack_sim_array).item()),
        'attack_max_similarity': float(np.max(attack_sim_array).item()),
        'non_attack_mean_similarity': float(np.mean(non_attack_sim_array).item()),
        'non_attack_std_similarity': float(np.std(non_attack_sim_array).item()),
        'non_attack_min_similarity': float(np.min(non_attack_sim_array).item()),
        'non_attack_max_similarity': float(np.max(non_attack_sim_array).item()),
        'similarity_gap': float((np.mean(non_attack_sim_array) - np.mean(attack_sim_array)).item())
    }
    
    return separation_metrics


def visualize_embeddings_tsne(
    embeddings: Dict[str, np.ndarray],
    attack_edges: List[Tuple[str, str]],
    node_to_idx: Dict[str, int],
    save_path: str = None,
    title: str = "t-SNE Visualization of Node Embeddings",
    figsize: Tuple[int, int] = (12, 8),
    perplexity: int = 30,
    random_state: int = 42
) -> plt.Figure:
    """
    t-SNEによる埋め込み空間の可視化
    
    Args:
        embeddings: ノード埋め込み辞書
        attack_edges: Attack関係のエッジ
        node_to_idx: ノードインデックスマッピング
        save_path: 保存先パス
        title: グラフタイトル
        figsize: 図のサイズ
        perplexity: t-SNEのperplexityパラメータ
        random_state: 乱数シード
        
    Returns:
        matplotlib Figure
    """
    # 埋め込みを行列に変換
    all_nodes = sorted(node_to_idx.keys())
    embedding_matrix = np.array([embeddings[node] for node in all_nodes])
    
    # ノードラベルを作成（Attack関係に含まれるノードか否か）
    attack_nodes = set()
    for u, v in attack_edges:
        attack_nodes.add(u)
        attack_nodes.add(v)
    
    node_labels = np.array([1 if node in attack_nodes else 0 for node in all_nodes])
    
    # t-SNE実行
    print(f"  t-SNE実行中（perplexity={perplexity}）...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    
    # 可視化
    fig, ax = plt.subplots(figsize=figsize)
    
    # 非Attack関係ノード
    mask_non_attack = node_labels == 0
    ax.scatter(
        embeddings_2d[mask_non_attack, 0],
        embeddings_2d[mask_non_attack, 1],
        c='blue',
        label='Non-Attack Nodes',
        alpha=0.6,
        s=50
    )
    
    # Attack関係ノード
    mask_attack = node_labels == 1
    ax.scatter(
        embeddings_2d[mask_attack, 0],
        embeddings_2d[mask_attack, 1],
        c='red',
        label='Attack Nodes',
        alpha=0.6,
        s=50
    )
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  t-SNE可視化を保存: {save_path}")
    
    return fig


def plot_similarity_distributions(
    attack_similarities: np.ndarray,
    non_attack_similarities: np.ndarray,
    save_path: str = None,
    title: str = "Cosine Similarity Distributions"
) -> plt.Figure:
    """
    Attack/非Attack関係の類似度分布をプロット
    
    Args:
        attack_similarities: Attack関係の類似度
        non_attack_similarities: 非Attack関係の類似度
        save_path: 保存先パス
        title: グラフタイトル
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ヒストグラム
    axes[0].hist(attack_similarities, bins=50, alpha=0.7, label='Attack', color='red', edgecolor='black')
    axes[0].hist(non_attack_similarities, bins=50, alpha=0.7, label='Non-Attack', color='blue', edgecolor='black')
    axes[0].set_xlabel('Cosine Similarity', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Similarity Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # ボックスプロット
    axes[1].boxplot(
        [attack_similarities, non_attack_similarities],
        labels=['Attack', 'Non-Attack'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1].set_title('Similarity Box Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  類似度分布を保存: {save_path}")
    
    return fig


def evaluate_embedding_quality(
    embeddings: Dict[str, np.ndarray],
    attack_edges: List[Tuple[str, str]],
    non_attack_edges: List[Tuple[str, str]],
    node_to_idx: Dict[str, int],
    output_dir: str = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    埋め込み品質の総合評価
    
    Args:
        embeddings: ノード埋め込み辞書
        attack_edges: Attack関係のエッジ
        non_attack_edges: 非Attack関係のエッジ
        node_to_idx: ノードインデックスマッピング
        output_dir: 出力ディレクトリ
        verbose: 詳細出力フラグ
        
    Returns:
        評価結果辞書
    """
    if verbose:
        print("\n" + "="*70)
        print("📊 埋め込み品質評価を開始...")
        print("="*70)
    
    # 埋め込みを行列に変換
    all_nodes = sorted(node_to_idx.keys())
    embedding_matrix = np.array([embeddings[node] for node in all_nodes])
    
    # エッジをインデックスに変換
    attack_pairs = [(node_to_idx[u], node_to_idx[v]) for u, v in attack_edges]
    non_attack_pairs = [(node_to_idx[u], node_to_idx[v]) for u, v in non_attack_edges]
    
    results = {}
    
    # 1. 分離度スコア
    if verbose:
        print("\n📏 Attack/非Attack関係の分離度を計算中...")
    
    separation_metrics = calculate_separation_score(
        embedding_matrix,
        attack_pairs,
        non_attack_pairs
    )
    results['separation'] = separation_metrics
    
    if verbose:
        print(f"  Attack関係平均類似度: {separation_metrics['attack_mean_similarity']:+.3f} ± {separation_metrics['attack_std_similarity']:.3f}")
        print(f"  非Attack関係平均類似度: {separation_metrics['non_attack_mean_similarity']:+.3f} ± {separation_metrics['non_attack_std_similarity']:.3f}")
        print(f"  類似度ギャップ: {separation_metrics['similarity_gap']:+.3f} (大きいほど良い)")
    
    # 2. クラスタリング係数
    if verbose:
        print("\n📐 クラスタリング係数を計算中...")
    
    # ノードラベルを作成
    attack_nodes = set()
    for u, v in attack_edges:
        attack_nodes.add(u)
        attack_nodes.add(v)
    
    node_labels = np.array([1 if node in attack_nodes else 0 for node in all_nodes])
    
    if len(np.unique(node_labels)) >= 2:
        clustering_score = calculate_clustering_score(embedding_matrix, node_labels, metric='cosine')
        results['clustering_score'] = float(clustering_score)
        
        if verbose:
            print(f"  Silhouette Score: {clustering_score:.3f} (範囲: [-1, +1], 高いほど良い)")
    else:
        results['clustering_score'] = None
        if verbose:
            print("  ⚠️ クラスが1つしかないためSilhouette Scoreを計算できません")
    
    # 3. t-SNE可視化
    if verbose:
        print("\n🎨 t-SNE可視化を生成中...")
    
    if output_dir:
        tsne_path = os.path.join(output_dir, 'tsne_visualization.png')
        fig_tsne = visualize_embeddings_tsne(
            embeddings,
            attack_edges,
            node_to_idx,
            save_path=tsne_path,
            title="t-SNE Visualization (After Contrastive Learning)"
        )
        plt.close(fig_tsne)
        results['tsne_plot_path'] = tsne_path
    
    # 4. 類似度分布プロット
    if verbose:
        print("\n📊 類似度分布を可視化中...")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    attack_sims = []
    for u_idx, v_idx in attack_pairs:
        sim = cosine_similarity(
            embedding_matrix[u_idx].reshape(1, -1),
            embedding_matrix[v_idx].reshape(1, -1)
        )[0, 0]
        attack_sims.append(sim)
    
    non_attack_sims = []
    for u_idx, v_idx in non_attack_pairs:
        sim = cosine_similarity(
            embedding_matrix[u_idx].reshape(1, -1),
            embedding_matrix[v_idx].reshape(1, -1)
        )[0, 0]
        non_attack_sims.append(sim)
    
    if output_dir:
        dist_path = os.path.join(output_dir, 'similarity_distributions.png')
        fig_dist = plot_similarity_distributions(
            np.array(attack_sims),
            np.array(non_attack_sims),
            save_path=dist_path,
            title="Cosine Similarity Distributions (After Contrastive Learning)"
        )
        plt.close(fig_dist)
        results['distribution_plot_path'] = dist_path
    
    if verbose:
        print("\n" + "="*70)
        print("✅ 埋め込み品質評価完了!")
        print("="*70)
    
    return results


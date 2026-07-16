"""
BERT埋め込みの対照学習による事前学習

Attack関係にあるノードペアを遠ざけ、非Attack関係のペアを近づける
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import os
import json
import random


class AttackContrastiveLearning(nn.Module):
    """
    Attack Link Prediction用の対照学習モジュール
    
    設計思想:
    - Attack関係（Direct Negation）のペアは最大距離を持つべき
    - 非Attack関係のペアは近接すべき
    - コサイン類似度ベースの距離メトリック
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 入力次元（BERT埋め込み次元）
            hidden_dim: 隠れ層次元
            output_dim: 出力埋め込み次元
            temperature: 温度パラメータ（小さいほど鋭い分布）
            dropout: ドロップアウト率
        """
        super(AttackContrastiveLearning, self).__init__()
        
        self.temperature = temperature
        
        # エンコーダー（BERT埋め込みを変換）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        埋め込みをエンコード
        
        Args:
            x: 入力埋め込み [batch_size, input_dim]
            
        Returns:
            エンコードされた埋め込み [batch_size, output_dim]
        """
        h = self.encoder(x)
        # L2正規化（コサイン類似度計算のため）
        h = F.normalize(h, p=2, dim=1)
        return h
    
    def compute_loss(
        self,
        node_embeddings: torch.Tensor,
        attack_pairs: List[Tuple[int, int]],
        non_attack_pairs: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        対照学習損失を計算
        
        Args:
            node_embeddings: ノード埋め込み [num_nodes, output_dim]
            attack_pairs: Attack関係のペア（遠ざける）
            non_attack_pairs: 非Attack関係のペア（近づける）
            
        Returns:
            loss: 損失値
            metrics: 統計情報
        """
        if len(attack_pairs) == 0 or len(non_attack_pairs) == 0:
            raise ValueError("attack_pairsとnon_attack_pairsは空であってはいけません")
        
        # Attack関係のペアの類似度（遠ざける = 類似度を最小化）
        attack_similarities = []
        for u_idx, v_idx in attack_pairs:
            u_emb = node_embeddings[u_idx]
            v_emb = node_embeddings[v_idx]
            sim = F.cosine_similarity(u_emb.unsqueeze(0), v_emb.unsqueeze(0))
            attack_similarities.append(sim)
        
        attack_sim_tensor = torch.stack(attack_similarities)
        
        # 非Attack関係のペアの類似度（近づける = 類似度を最大化）
        non_attack_similarities = []
        for u_idx, v_idx in non_attack_pairs:
            u_emb = node_embeddings[u_idx]
            v_emb = node_embeddings[v_idx]
            sim = F.cosine_similarity(u_emb.unsqueeze(0), v_emb.unsqueeze(0))
            non_attack_similarities.append(sim)
        
        non_attack_sim_tensor = torch.stack(non_attack_similarities)
        
        # 損失計算
        # Attack関係: 類似度を最小化（= -1に近づける）
        # 目標: sim -> -1, 損失 = (sim - (-1))^2 = (sim + 1)^2
        attack_loss = torch.mean((attack_sim_tensor + 1.0) ** 2)
        
        # 非Attack関係: 類似度を最大化（= +1に近づける）
        # 目標: sim -> +1, 損失 = (sim - 1)^2
        non_attack_loss = torch.mean((non_attack_sim_tensor - 1.0) ** 2)
        
        # 総損失
        total_loss = attack_loss + non_attack_loss
        
        # 統計情報
        metrics = {
            'attack_loss': attack_loss.item(),
            'non_attack_loss': non_attack_loss.item(),
            'attack_avg_sim': attack_sim_tensor.mean().item(),
            'non_attack_avg_sim': non_attack_sim_tensor.mean().item(),
            'attack_std_sim': attack_sim_tensor.std().item(),
            'non_attack_std_sim': non_attack_sim_tensor.std().item()
        }
        
        return total_loss, metrics


def pretrain_embeddings_with_contrastive_learning(
    initial_embeddings: Dict[str, np.ndarray],
    attack_edges: List[Tuple[str, str]],
    non_attack_edges: List[Tuple[str, str]],
    node_to_idx: Dict[str, int],
    config: Dict,
    device: str = 'cpu',
    verbose: bool = True,
    output_dir: str = None,
    seed: int = None,
    deterministic: bool = True,
    save_artifacts: bool = True
) -> Dict[str, np.ndarray]:
    """
    対照学習によるBERT埋め込みの事前学習
    
    Args:
        initial_embeddings: 初期BERT埋め込み {node: embedding}
        attack_edges: Attack関係のエッジ
        non_attack_edges: 非Attack関係のエッジ
        node_to_idx: ノードインデックスマッピング
        config: 設定辞書
        device: 計算デバイス
        verbose: 詳細出力フラグ
        
    Returns:
        学習後の埋め込み {node: embedding}
    """
    if verbose:
        print("\n" + "="*70)
        print("🎯 対照学習による埋め込み事前学習を開始...")
        print("="*70)
    
    # 設定取得
    contrastive_config = config.get('contrastive_learning', {})
    hidden_dim = contrastive_config.get('hidden_dim', 256)
    output_dim = contrastive_config.get('output_dim', 128)
    temperature = contrastive_config.get('temperature', 0.07)
    dropout = contrastive_config.get('dropout', 0.1)
    num_epochs = contrastive_config.get('num_epochs', 50)
    learning_rate = contrastive_config.get('learning_rate', 0.001)
    batch_size = contrastive_config.get('batch_size', 128)
    # シード決定（優先順位: 引数 > config['data']['seed']）
    if seed is None:
        seed = config.get('data', {}).get('seed', None)
    # シード固定
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    # 初期埋め込みをTensorに変換
    all_nodes = sorted(node_to_idx.keys())
    embedding_dim = len(list(initial_embeddings.values())[0])
    
    embedding_matrix = np.array([initial_embeddings[node] for node in all_nodes])
    x_initial = torch.tensor(embedding_matrix, dtype=torch.float32).to(device)
    
    if verbose:
        print(f"\n📊 データ統計:")
        print(f"  ノード数: {len(all_nodes)}")
        print(f"  Attack関係ペア数: {len(attack_edges)}")
        print(f"  非Attack関係ペア数: {len(non_attack_edges)}")
        print(f"  初期埋め込み次元: {embedding_dim}")
        print(f"  出力埋め込み次元: {output_dim}")
    
    # エッジをインデックスに変換
    attack_pairs = [(node_to_idx[u], node_to_idx[v]) for u, v in attack_edges]
    non_attack_pairs = [(node_to_idx[u], node_to_idx[v]) for u, v in non_attack_edges]
    
    # モデル初期化
    model = AttackContrastiveLearning(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        temperature=temperature,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n⚙️  モデル情報:")
        print(f"  総パラメータ数: {total_params:,}")
        print(f"  学習可能パラメータ数: {trainable_params:,}")
        print(f"  学習率: {learning_rate}")
        print(f"  エポック数: {num_epochs}")
        print(f"  温度パラメータ: {temperature}")
        if seed is not None:
            print(f"  乱数シード: {seed} (deterministic={deterministic})")
    
    # 学習ループ
    model.train()
    history = {
        'loss': [],
        'attack_loss': [],
        'non_attack_loss': [],
        'attack_avg_sim': [],
        'non_attack_avg_sim': []
    }
    
    if verbose:
        print(f"\n🚀 学習開始...")
        print("-" * 70)
    
    for epoch in range(num_epochs):
        # 全ノードをエンコード
        node_embeddings = model(x_initial)
        
        # バッチサンプリング（大規模データ対応）
        if len(attack_pairs) > batch_size:
            attack_indices = np.random.choice(len(attack_pairs), batch_size, replace=False)
            attack_batch = [attack_pairs[i] for i in attack_indices]
        else:
            attack_batch = attack_pairs
        
        if len(non_attack_pairs) > batch_size:
            non_attack_indices = np.random.choice(len(non_attack_pairs), batch_size, replace=False)
            non_attack_batch = [non_attack_pairs[i] for i in non_attack_indices]
        else:
            non_attack_batch = non_attack_pairs
        
        # 損失計算
        optimizer.zero_grad()
        loss, metrics = model.compute_loss(node_embeddings, attack_batch, non_attack_batch)
        loss.backward()
        optimizer.step()
        
        # 履歴記録
        history['loss'].append(loss.item())
        history['attack_loss'].append(metrics['attack_loss'])
        history['non_attack_loss'].append(metrics['non_attack_loss'])
        history['attack_avg_sim'].append(metrics['attack_avg_sim'])
        history['non_attack_avg_sim'].append(metrics['non_attack_avg_sim'])
        
        # 進捗表示
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"  Epoch {epoch:3d}/{num_epochs}: "
                  f"Loss={loss.item():.4f}, "
                  f"Attack_sim={metrics['attack_avg_sim']:+.3f}, "
                  f"NonAttack_sim={metrics['non_attack_avg_sim']:+.3f}")
    
    if verbose:
        print("-" * 70)
        print("✅ 対照学習完了!")
        print(f"\n📈 最終結果:")
        print(f"  最終損失: {history['loss'][-1]:.4f}")
        print(f"  Attack関係平均類似度: {history['attack_avg_sim'][-1]:+.3f} (目標: -1.0)")
        print(f"  非Attack関係平均類似度: {history['non_attack_avg_sim'][-1]:+.3f} (目標: +1.0)")
    
    # 学習済み埋め込みを取得
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x_initial).cpu().numpy()
    
    # 辞書形式に変換
    optimized_embeddings = {
        node: final_embeddings[idx] for node, idx in node_to_idx.items()
    }
    
    # 成果物保存
    if output_dir and save_artifacts:
        os.makedirs(output_dir, exist_ok=True)
        # 埋め込み（ノード配列と行列をnpzで保存）
        try:
            all_nodes_sorted = sorted(node_to_idx.keys())
            np.savez_compressed(
                os.path.join(output_dir, 'embeddings_after_contrastive.npz'),
                nodes=np.array(all_nodes_sorted),
                embeddings=final_embeddings
            )
            if verbose:
                print(f"\n💾 埋め込みを保存: {os.path.join(output_dir, 'embeddings_after_contrastive.npz')}\n")
        except Exception as e:
            if verbose:
                print(f"⚠️ 埋め込み保存に失敗: {e}")
        # 学習履歴
        try:
            history_path = os.path.join(output_dir, 'contrastive_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"💾 学習履歴を保存: {history_path}")
        except Exception as e:
            if verbose:
                print(f"⚠️ 学習履歴保存に失敗: {e}")
        # モデル重み
        try:
            model_path = os.path.join(output_dir, 'contrastive_model.pt')
            torch.save({'state_dict': model.state_dict(),
                        'input_dim': embedding_dim,
                        'hidden_dim': hidden_dim,
                        'output_dim': output_dim,
                        'dropout': dropout,
                        'temperature': temperature,
                        'seed': seed}, model_path)
            if verbose:
                print(f"💾 モデル重みを保存: {model_path}")
        except Exception as e:
            if verbose:
                print(f"⚠️ モデル保存に失敗: {e}")

    return optimized_embeddings, history


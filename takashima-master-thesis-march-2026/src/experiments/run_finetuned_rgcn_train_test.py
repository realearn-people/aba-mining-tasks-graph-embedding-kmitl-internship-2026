"""
FinetunedBertRgcnMlp 単一の8:2（train:test）分割で学習・評価するランナー
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.run_robust_experiment import (
    load_config,
    determine_experiment_id,
    setup_output_directory,
    prepare_data,
    generate_negatives,
)
from src.model_defs.models import FinetunedBertRgcnMlp
from src.model_training.train import train_model
from src.model_training.evaluate import evaluate_model


def split_train_val(edges: List[Tuple], val_ratio: float, seed: int) -> Tuple[List[Tuple], List[Tuple]]:
    rnd = random.Random(seed)
    shuffled = edges.copy()
    rnd.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 1
    val_edges = shuffled[:n_val]
    train_edges = shuffled[n_val:]
    return train_edges, val_edges


def run_train_test(
    config_path: str,
    hypara_path: str,
    args=None
):
    # 設定読込
    config = load_config(config_path)
    # 対照学習は無効化（明示）
    if 'contrastive_learning' in config:
        config['contrastive_learning']['enabled'] = False

    # 実験IDと出力先
    experiment_id = determine_experiment_id(config, args)
    output_dir = setup_output_directory(config, experiment_id)
    config['data']['output_dir'] = output_dir
    config['data']['experiment_id'] = experiment_id

    # デバイス
    if config['compute']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['compute']['device'])
    print(f"\n💻 使用デバイス: {device}")

    # 乱数シード
    seed = int(config['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # データ準備
    (original_graph, inference_graph, attack_edges, all_nodes,
     node_embeddings, node_to_idx, embedding_matrix, data) = prepare_data(config)

    # ネガティブサンプリング
    all_negatives = generate_negatives(
        original_graph, all_nodes, attack_edges,
        embedding_matrix, node_to_idx, inference_graph, config
    )

    # 全エッジ（ラベル付き）を作成し、8:2で hold-out
    all_edges_with_labels = []
    for e in attack_edges:
        all_edges_with_labels.append((e, 1))
    for e in all_negatives:
        all_edges_with_labels.append((e, 0))
    random.shuffle(all_edges_with_labels)

    test_size = int(0.2 * len(all_edges_with_labels))
    test_edges = all_edges_with_labels[:test_size]
    base_train_edges = all_edges_with_labels[test_size:]
    print(f"\n📊 Hold-out 分割: train={len(base_train_edges)}, test={len(test_edges)}")

    # 検証分割（訓練内から val_split_ratio）
    val_ratio = float(config['cross_validation']['val_split_ratio'])
    train_edges, val_edges = split_train_val(base_train_edges, val_ratio, seed=seed)
    print(f"    ┗ train={len(train_edges)}, val={len(val_edges)} (val_ratio={val_ratio})")

    # ハイパラのロード
    with open(hypara_path, 'r', encoding='utf-8') as f:
        hp = json.load(f)

    # 値を取り出し（単一値を想定。配列なら先頭使用）
    def first_or_self(x, default=None):
        if isinstance(x, list) and len(x) > 0:
            return x[0]
        return x if x is not None else default

    hidden_dim = int(first_or_self(hp.get('hidden_dim'), 128))
    num_layers = int(first_or_self(hp.get('num_layers'), 2))
    dropout_link = float(first_or_self(hp.get('dropout_link'), 0.5))
    learning_rate = float(first_or_self(hp.get('learning_rate'), 3e-5))
    num_epochs = int(first_or_self(hp.get('num_epochs'), 50))
    max_length = int(first_or_self(hp.get('max_length'), 128))
    lora_r = int(first_or_self(hp.get('lora_r'), 8))
    lora_alpha = int(first_or_self(hp.get('lora_alpha'), 16))
    lora_dropout = float(first_or_self(hp.get('lora_dropout'), 0.1))

    # モデル初期化
    bert_cfg = config['models']['improved_bert']
    model = FinetunedBertRgcnMlp(
        all_nodes=all_nodes,
        model_name=bert_cfg['model_name'],
        max_length=max_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_relations=1,
        dropout_link=dropout_link,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # 学習
    history = train_model(
        model,
        data,
        train_edges,
        node_to_idx,
        num_epochs=num_epochs,
        lr=learning_rate,
        model_name="FinetunedBertRgcnMlp(train-test)",
        verbose=True,
        validation_edges=val_edges,
        device=str(device)
    )

    # 評価
    metrics, _preds = evaluate_model(model, data, test_edges, node_to_idx, device=str(device))
    print(f"\n✅ 評価完了: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

    # 保存
    out = {
        'config_used': {
            'hypara_path': hypara_path,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout_link': dropout_link,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'max_length': max_length,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
        },
        'split': {
            'train': len(train_edges),
            'val': len(val_edges),
            'test': len(test_edges),
        },
        'metrics': metrics,
        'training_history': history,
    }
    save_path = os.path.join(output_dir, 'train_test_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"💾 結果保存: {save_path}")
    print(f"📁 出力先: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run FinetunedBertRgcnMlp on single 80/20 train-test split.")
    parser.add_argument('--config', type=str, default='config/robust_experiment.yaml')
    parser.add_argument('--hypara', type=str, required=True, help='Path to hypara.json')
    parser.add_argument('--experiment-id', type=str, default=None, dest='experiment_id')
    args = parser.parse_args()
    run_train_test(args.config, args.hypara, args)


if __name__ == '__main__':
    main()
















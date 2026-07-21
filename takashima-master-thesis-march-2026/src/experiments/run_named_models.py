"""
Named Models Experiment Runner

以下の6モデルを固定セットとして5-fold CVで学習・評価し、結果を保存・可視化します。

- FreezedBertRgcnMlp
- FreezedBertMlp
- FinetunedBertMlp
- FinetunedBertCosSim
- TfidfLr
- Random

備考:
- 対照学習（contrastive learning）は `--use-contrastive` オプションで有効化可能
- 設定は既存の robust_experiment.yaml を流用します（ハイパラ・入出力位置など）。
"""

import os
import sys
import subprocess
import argparse

def _should_disable_cuda() -> bool:
    try:
        import yaml as _yaml
        _p = argparse.ArgumentParser(add_help=False)
        _p.add_argument('--config', default=None)
        _args, _ = _p.parse_known_args()
        if _args.config:
            with open(_args.config) as _f:
                _cfg = _yaml.safe_load(_f)
            _dev = _cfg.get('compute', {}).get('device', 'auto')
            if _dev == 'cpu':
                print("[device] config device=cpu → disabling CUDA")
                return True
            if _dev == 'cuda':
                return False
    except Exception:
        pass
    try:
        _out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            timeout=5
        )
        _free_mib = int(_out.decode().strip().split('\n')[0])
        if _free_mib < 1024:
            print(f"[device] GPU has only {_free_mib} MiB free → disabling CUDA")
            return True
        print(f"[device] GPU has {_free_mib} MiB free → CUDA enabled")
        return False
    except Exception:
        print("[device] nvidia-smi unavailable → disabling CUDA")
        return True

if _should_disable_cuda():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
# ─────────────────────────────────────────────────────────────────────────────

import json
import itertools
import random as _random
from typing import Dict, List, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

# プロジェクトルートをパスに追加（スクリプト直実行対応）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.cross_validation import create_cross_validation_splits, create_cross_validation_splits_multiclass
from src.experiments.run_robust_experiment import (
    load_config,
    determine_experiment_id,
    setup_output_directory,
    prepare_data,
    generate_negatives,
    apply_contrastive_learning,
)
from src.model_defs.models import (
    FreezedBertRgcnMlp,
    FreezedBertMlp,
    FinetunedBertMlp,
    FinetunedBertCosSim,
    TfidfLr,
    Random,
    FinetunedBertRgcnMlp,
)
from src.model_training.train import train_model
from src.model_training.train_bert import (
    ABADataset,
    train_bert_model,
    evaluate_bert_model,
)
from src.model_training.evaluate import evaluate_model, evaluate_baseline


def _make_bert_loader(subset, batch_size, num_workers, pin_memory, num_classes, is_train=True):
    """DataLoader with WeightedRandomSampler to prevent majority-class collapse (binary and multi-class)."""
    if not is_train:
        return DataLoader(subset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    labels = [subset.dataset.edges[i][1] for i in subset.indices]
    from collections import Counter
    counts = Counter(labels)
    if len(counts) < 2:
        return DataLoader(subset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    # Each class gets weight 1/count so every class is equally represented per batch
    class_w = {c: 1.0 / n for c, n in counts.items()}
    weights = [class_w[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(subset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)


from src.visualization.plot_results import (
    calculate_statistics,
    perform_statistical_tests,
    plot_box_plots,
    plot_bar_charts,
    plot_comprehensive_analysis,
    display_results_table,
    save_results_to_file,
)


def run_named_models_experiment(config_path: str, args=None):
    # 設定の読み込み
    config = load_config(config_path)

    # 実験ID（デフォルト名を補助）
    experiment_id = determine_experiment_id(config, args)
    # 明示指定があればサフィックスを付けずにそのまま使用。未指定時のみ既定名を用いる
    if not experiment_id:
        experiment_id = "named_models"

    # 出力先
    output_dir = setup_output_directory(config, experiment_id)
    config['data']['output_dir'] = output_dir
    config['data']['experiment_id'] = experiment_id

    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"\n💻 使用デバイス: {device}")
    # DataLoader最適化
    dl_workers = int(config.get('compute', {}).get('num_workers', 0))
    pin_mem = (str(device) == 'cuda')

    # データ準備（BERTノード埋め込みを生成し特徴に使用）
    (
        original_graph,
        inference_graph,
        attack_edges,
        not_contrary_edges,
        support_edge_pairs,
        all_nodes,
        node_embeddings,
        node_to_idx,
        embedding_matrix,
        data,
    ) = prepare_data(config)

    num_classes = int(config['data'].get('num_classes', 2))

    # ネガティブサンプリング / ラベル付きネガティブ収集
    if num_classes == 2 and (not_contrary_edges or support_edge_pairs):
        # Use actual labeled NOT_CONTRARY + SUPPORT pairs as negatives
        all_negatives = list(not_contrary_edges) + list(support_edge_pairs)
        print(f"\n📊 Labeled negatives: {len(not_contrary_edges):,} NOT_CONTRARY + "
              f"{len(support_edge_pairs):,} SUPPORT = {len(all_negatives):,} total")
    else:
        all_negatives = generate_negatives(
            original_graph, all_nodes, attack_edges,
            embedding_matrix, node_to_idx, inference_graph, config,
        )

    # 対照学習による埋め込み最適化（オプション）
    use_contrastive = args and hasattr(args, 'use_contrastive') and args.use_contrastive
    if use_contrastive:
        # 対照学習を有効化
        if 'contrastive_learning' not in config:
            config['contrastive_learning'] = {}
        config['contrastive_learning']['enabled'] = True
        
        # 対照学習を適用
        optimized_embeddings, contrastive_history = apply_contrastive_learning(
            node_embeddings,
            attack_edges,
            all_negatives,
            node_to_idx,
            config,
            output_dir,
            device=str(device)
        )
        
        # 最適化された埋め込みで行列を更新
        embedding_matrix = np.array([optimized_embeddings[node] for node in all_nodes])
        x = torch.tensor(embedding_matrix, dtype=torch.float32)
        data.x = x
        print(f"\n✅ 対照学習後の埋め込みでグラフデータを更新しました")
        print(f"   更新後の埋め込み次元: {embedding_matrix.shape}")
    else:
        # 対照学習を無効化（明示的に）
        if 'contrastive_learning' in config:
            config['contrastive_learning']['enabled'] = False
        print("\n⚠️  対照学習は使用しません（ベースライン実験）")

    # CV分割
    if num_classes == 3:
        cv_splits = create_cross_validation_splits_multiclass(
            edges_by_class={
                0: support_edge_pairs,
                1: not_contrary_edges,
                2: attack_edges,
            },
            n_splits=int(config['cross_validation']['n_folds']),
            seed=int(config['data']['seed']),
        )
    else:
        cv_splits = create_cross_validation_splits(
            attack_edges,
            all_negatives,
            n_splits=int(config['cross_validation']['n_folds']),
            seed=int(config['data']['seed']),
        )

    # 結果入れ物（全モデルのデフォルト）
    all_models = [
        'FreezedBertRgcnMlp',
        'FinetunedBertRgcnMlp',
        'FreezedBertMlp',
        'FinetunedBertMlp',
        'FinetunedBertCosSim',
        'TfidfLr',
        'Random',
    ]
    # --only-model が指定されていれば単一モデルのみ実行
    model_names = all_models
    if args and hasattr(args, 'only_model') and args.only_model:
        if args.only_model not in all_models:
            raise ValueError(f"Unknown model '{args.only_model}'. Choose from: {', '.join(all_models)}")
        model_names = [args.only_model]
    _ALL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc',
                    'mean_rank', 'mrr', 'hits@1_micro', 'hits@1_macro', 'hits@3', 'hits@10']
    results: Dict[str, Dict[str, List[float]]] = {
        name: {m: [] for m in _ALL_METRICS} for name in model_names
    }

    def ensure_results_key(key: str):
        if key not in results:
            results[key] = {m: [] for m in _ALL_METRICS}

    print(f"\n{'='*70}")
    print("固定セット（6モデル）で5-fold Cross-Validation を開始...")
    print(f"対象モデル: {', '.join(model_names)}")
    print(f"{'='*70}\n")

    # BERT設定（ImprovedBERT設定を流用）
    bert_cfg = config['models']['improved_bert']
    val_split_ratio = float(config['cross_validation']['val_split_ratio'])

    # 事前結果からFinetunedBertMlpの最良ハイパラを抽出（任意）
    selected_ftb = None
    if args and hasattr(args, 'best_from_results') and args.best_from_results:
        try:
            with open(args.best_from_results, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            # 統計から最大F1平均のモデルを選択（キー形式: FinetunedBertMlp[lr=...,do=...,ml=...,bs=...])
            stats = prev.get('statistics', {})
            best_key = None
            best_f1 = -1.0
            for model_name, metrics in stats.items():
                if isinstance(model_name, str) and model_name.startswith('FinetunedBertMlp['):
                    f1m = metrics.get('f1', {}).get('mean', None)
                    if f1m is not None and f1m > best_f1:
                        best_f1 = f1m
                        best_key = model_name
            if best_key:
                # 例: FinetunedBertMlp[lr=3e-05,do=0.2,ml=128,bs=32]
                try:
                    inside = best_key.split('[', 1)[1].rstrip(']')
                    parts = {kv.split('=')[0]: kv.split('=')[1] for kv in inside.split(',')}
                    selected_ftb = {
                        'learning_rate': float(parts['lr']),
                        'dropout': float(parts['do']),
                        'max_length': int(parts['ml']),
                        'batch_size': int(parts['bs'])
                    }
                    print(f"\n🔎 既存結果から最良FinetunedBertMlpを使用: {best_key} (F1={best_f1:.4f})")
                except Exception as pe:
                    print(f"⚠️ 最良ハイパラの解析に失敗しました: {pe}")
        except Exception as e:
            print(f"⚠️ best_from_results の読込に失敗しました: {e}")

    # 事前結果からFreezedBertRgcnMlpの最良ハイパラを抽出（任意）
    selected_rgcn = None
    if args and hasattr(args, 'best_rgcn_from_results') and args.best_rgcn_from_results:
        try:
            with open(args.best_rgcn_from_results, 'r', encoding='utf-8') as f:
                prev_r = json.load(f)
            stats_r = prev_r.get('statistics', {})
            best_key_r = None
            best_f1_r = -1.0
            for model_name, metrics in stats_r.items():
                if isinstance(model_name, str) and model_name.startswith('FreezedBertRgcnMlp['):
                    f1m = metrics.get('f1', {}).get('mean', None)
                    if f1m is not None and f1m > best_f1_r:
                        best_f1_r = f1m
                        best_key_r = model_name
            if best_key_r:
                try:
                    inside = best_key_r.split('[', 1)[1].rstrip(']')
                    parts = {kv.split('=')[0]: kv.split('=')[1] for kv in inside.split(',')}
                    selected_rgcn = {
                        'hidden_dim': int(parts['hd']),
                        'num_layers': int(parts['layers']),
                        'dropout_link': float(parts['dr']),
                        'learning_rate': float(parts['lr']),
                        'num_epochs': int(parts.get('ep',  int(config['models']['rgcn']['num_epochs'])))
                    }
                    print(f"\n🔎 既存結果から最良FreezedBertRgcnMlpを使用: {best_key_r} (F1={best_f1_r:.4f})")
                except Exception as pe:
                    print(f"⚠️ RGCN最良ハイパラの解析に失敗しました: {pe}")
        except Exception as e:
            print(f"⚠️ best_rgcn_from_results の読込に失敗しました: {e}")

    for fold_idx, (train_edges, test_edges) in enumerate(cv_splits):
        print(f"\n📊 Fold {fold_idx + 1}/{len(cv_splits)}")
        print("-" * 50)

        # 1) FreezedBertRgcnMlp（R-GCN）
        if 'FreezedBertRgcnMlp' in model_names:
            rgcn_cfg = config['models']['rgcn']
            # train/val split
            train_size = int((1 - val_split_ratio) * len(train_edges))
            shuffled = train_edges.copy()
            import random as _random
            _random.shuffle(shuffled)
            rgcn_train_edges = shuffled[:train_size]
            rgcn_val_edges = shuffled[train_size:]

            # 既存結果のベスト設定があればそれを使用
            if selected_rgcn is not None:
                print("🔥 FreezedBertRgcnMlp（best_from_results）を学習中...")
                rgcn_model = FreezedBertRgcnMlp(
                    input_dim=data.x.shape[1],
                    hidden_dim=int(selected_rgcn['hidden_dim']),
                    num_layers=int(selected_rgcn['num_layers']),
                    num_relations=1, 
                    dropout_link=float(selected_rgcn['dropout_link']),
                    num_classes=num_classes,
                )
                _ = train_model(
                    rgcn_model, data, rgcn_train_edges, node_to_idx,
                    num_epochs=int(selected_rgcn['num_epochs']),
                    lr=float(selected_rgcn['learning_rate']),
                    model_name="FreezedBertRgcnMlp(best)",
                    verbose=bool(rgcn_cfg.get('verbose', True)),
                    validation_edges=rgcn_val_edges,
                    device=str(device), num_classes=num_classes,
                )
                metrics, _preds = evaluate_model(rgcn_model, data, test_edges, node_to_idx,
                                                 device=str(device), num_classes=num_classes)
                key = (
                    f"FreezedBertRgcnMlp[hd={selected_rgcn['hidden_dim']},"
                    f"layers={selected_rgcn['num_layers']},dr={selected_rgcn['dropout_link']},"
                    f"lr={selected_rgcn['learning_rate']},ep={selected_rgcn['num_epochs']}]"
                )
                ensure_results_key(key)
                for k, v in metrics.items():
                    results[key][k].append(v)
                print(f"結果[{key}]: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

            # RGCNスイープ（only-modelがRGCNの時やsweep指定時に有効）
            elif args and hasattr(args, 'sweep_config') and args.sweep_config and (not args.only_model or args.only_model == 'FreezedBertRgcnMlp'):
                with open(args.sweep_config, 'r', encoding='utf-8') as f:
                    sweep = json.load(f)

                hidden_dims = sweep.get('hidden_dim', [int(rgcn_cfg['hidden_dim'])])
                num_layers_list = sweep.get('num_layers', [int(rgcn_cfg['num_layers'])])
                dropouts_link = sweep.get('dropout_link', [0.5])
                lrs = sweep.get('learning_rate', [float(rgcn_cfg['learning_rate'])])
                num_epochs_list = sweep.get('num_epochs', [int(rgcn_cfg['num_epochs'])])

                combos = list(itertools.product(hidden_dims, num_layers_list, dropouts_link, lrs, num_epochs_list))
                strategy = getattr(args, 'search_strategy', 'grid')
                max_trials = getattr(args, 'max_trials', None)
                if strategy == 'random':
                    _random.shuffle(combos)
                if max_trials is not None:
                    try:
                        combos = combos[:int(max_trials)]
                    except Exception:
                        pass

                print(f"🔥 FreezedBertRgcnMlp スイープ開始: 試行数={len(combos)}")
                for (hd_v, nl_v, dr_v, lr_v, ne_v) in combos:
                    rgcn_model = FreezedBertRgcnMlp(
                        input_dim=data.x.shape[1],
                        hidden_dim=int(hd_v),
                        num_layers=int(nl_v),
                        num_relations=1, 
                        dropout_link=float(dr_v),
                        num_classes=num_classes,
                    )
                    _ = train_model(
                        rgcn_model,
                        data,
                        rgcn_train_edges,
                        node_to_idx,
                        num_epochs=int(ne_v),
                        lr=float(lr_v),
                        model_name="FreezedBertRgcnMlp(sweep)",
                        verbose=bool(rgcn_cfg.get('verbose', True)),
                        validation_edges=rgcn_val_edges,
                        device=str(device),
                        num_classes=num_classes,
                    )
                    metrics, _preds = evaluate_model(rgcn_model, data, test_edges, node_to_idx, device=str(device), num_classes=num_classes)
                    key = f"FreezedBertRgcnMlp[hd={hd_v},layers={nl_v},dr={dr_v},lr={lr_v},ep={ne_v}]"
                    ensure_results_key(key)
                    for k, v in metrics.items():
                        results[key][k].append(v)
                    print(f"結果[{key}]: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
            else:
                print("🔥 FreezedBertRgcnMlp を学習中...")
                rgcn_model = FreezedBertRgcnMlp(
                    input_dim=data.x.shape[1],
                    hidden_dim=int(rgcn_cfg['hidden_dim']),
                    num_layers=int(rgcn_cfg['num_layers']),
                    num_relations=1, 
                    dropout_link=0.5,
                    num_classes=num_classes,
                )
                _ = train_model(
                    rgcn_model,
                    data,
                    rgcn_train_edges,
                    node_to_idx,
                    num_epochs=int(rgcn_cfg['num_epochs']),
                    lr=float(rgcn_cfg['learning_rate']),
                    model_name="FreezedBertRgcnMlp",
                    verbose=bool(rgcn_cfg.get('verbose', True)),
                    validation_edges=rgcn_val_edges,
                    device=str(device),
                    num_classes=num_classes,
                )
                metrics, _preds = evaluate_model(rgcn_model, data, test_edges, node_to_idx, device=str(device), num_classes=num_classes)
                for k, v in metrics.items():
                    results['FreezedBertRgcnMlp'][k].append(v)
                print(f"結果: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

        # 1b) FinetunedBertRgcnMlp（BERT微調整 + R-GCN）
        if 'FinetunedBertRgcnMlp' in model_names:
            rgcn_cfg = config['models']['rgcn']
            # train/val split（R-GCN と同様にエッジレベルで分割）
            train_size = int((1 - val_split_ratio) * len(train_edges))
            shuffled = train_edges.copy()
            import random as _random  # 局所利用（上と同様）
            _random.shuffle(shuffled)
            ftrgcn_train_edges = shuffled[:train_size]
            ftrgcn_val_edges = shuffled[train_size:]

            # スイープ設定がある場合はグリッド/ランダム探索
            if args and hasattr(args, 'sweep_config') and args.sweep_config and (not args.only_model or args.only_model == 'FinetunedBertRgcnMlp'):
                with open(args.sweep_config, 'r', encoding='utf-8') as f:
                    sweep = json.load(f)

                hidden_dims = sweep.get('hidden_dim', [int(rgcn_cfg['hidden_dim'])])
                num_layers_list = sweep.get('num_layers', [int(rgcn_cfg['num_layers'])])
                dropouts_link = sweep.get('dropout_link', [0.5])
                lrs = sweep.get('learning_rate', [3e-5])
                num_epochs_list = sweep.get('num_epochs', [int(rgcn_cfg['num_epochs'])])
                max_lengths = sweep.get('max_length', [int(bert_cfg['max_length'])])
                # LoRA hyperparameters (optional in sweep)
                lora_rs = sweep.get('lora_r', [8])
                lora_alphas = sweep.get('lora_alpha', [16])
                lora_dropouts = sweep.get('lora_dropout', [0.1])

                combos = list(itertools.product(
                    hidden_dims,
                    num_layers_list,
                    dropouts_link,
                    lrs,
                    num_epochs_list,
                    max_lengths,
                    lora_rs,
                    lora_alphas,
                    lora_dropouts
                ))
                strategy = getattr(args, 'search_strategy', 'grid')
                max_trials = getattr(args, 'max_trials', None)
                if strategy == 'random':
                    _random.shuffle(combos)
                if max_trials is not None:
                    try:
                        combos = combos[:int(max_trials)]
                    except Exception:
                        pass

                print(f"\n🤖 FinetunedBertRgcnMlp スイープ開始: 試行数={len(combos)}")
                for (hd_v, nl_v, dr_v, lr_v, ne_v, ml_v, lora_r_v, lora_alpha_v, lora_dropout_v) in combos:
                    model_trial = FinetunedBertRgcnMlp(
                        all_nodes=all_nodes,
                        model_name=bert_cfg['model_name'],
                        max_length=int(ml_v),
                        hidden_dim=int(hd_v),
                        num_layers=int(nl_v),
                        num_relations=1, 
                        num_classes=num_classes,
                        lora_r=int(lora_r_v),
                        lora_alpha=int(lora_alpha_v),
                        lora_dropout=float(lora_dropout_v),
                    )
                    _ = train_model(
                        model_trial,
                        data,
                        ftrgcn_train_edges,
                        node_to_idx,
                        num_epochs=int(ne_v),
                        lr=float(lr_v),
                        model_name="FinetunedBertRgcnMlp(sweep)",
                        verbose=bool(rgcn_cfg.get('verbose', True)),
                        validation_edges=ftrgcn_val_edges,
                        device=str(device),
                        num_classes=num_classes,
                    )
                    metrics, _preds = evaluate_model(model_trial, data, test_edges, node_to_idx, device=str(device), num_classes=num_classes)
                    key = (
                        f"FinetunedBertRgcnMlp[hd={hd_v},layers={nl_v},dr={dr_v},"
                        f"lr={lr_v},ep={ne_v},ml={ml_v},lora_r={lora_r_v},lora_a={lora_alpha_v},lora_do={lora_dropout_v}]"
                    )
                    ensure_results_key(key)
                    for k, v in metrics.items():
                        results[key][k].append(v)
                    print(f"結果[{key}]: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
            else:
                print("\n🤖 FinetunedBertRgcnMlp を学習中...")
                model_ftrgcn = FinetunedBertRgcnMlp(
                    all_nodes=all_nodes,
                    model_name=bert_cfg['model_name'],
                    max_length=int(bert_cfg['max_length']),
                    hidden_dim=int(rgcn_cfg['hidden_dim']),
                    num_layers=int(rgcn_cfg['num_layers']),
                    num_relations=1, 
                    dropout_link=0.5,
                    num_classes=num_classes,
                )
                _ = train_model(
                    model_ftrgcn,
                    data,
                    ftrgcn_train_edges,
                    node_to_idx,
                    num_epochs=int(rgcn_cfg['num_epochs']),
                    lr=3e-5,
                    model_name="FinetunedBertRgcnMlp",
                    verbose=bool(rgcn_cfg.get('verbose', True)),
                    validation_edges=ftrgcn_val_edges,
                    device=str(device),
                    num_classes=num_classes,
                )
                metrics, _preds = evaluate_model(model_ftrgcn, data, test_edges, node_to_idx, device=str(device), num_classes=num_classes)
                for k, v in metrics.items():
                    results['FinetunedBertRgcnMlp'][k].append(v)
                print(f"結果: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

        # 2) FreezedBertMlp
        if 'FreezedBertMlp' in model_names:
            ds_train = ABADataset(train_edges, all_nodes)
            ds_test = ABADataset(test_edges, all_nodes)
            tr_size = int((1 - val_split_ratio) * len(ds_train))
            va_size = len(ds_train) - tr_size
            if tr_size >= 1 and va_size >= 1:
                tr_subset, va_subset = torch.utils.data.random_split(ds_train, [tr_size, va_size])
            else:
                tr_subset, va_subset = ds_train, ds_test

            # スイープ設定がある場合はグリッド/ランダム探索を実行
            if args and hasattr(args, 'sweep_config') and args.sweep_config and (not args.only_model or args.only_model == 'FreezedBertMlp'):
                with open(args.sweep_config, 'r', encoding='utf-8') as f:
                    sweep = json.load(f)

                lrs = sweep.get('learning_rate', [float(bert_cfg['learning_rate'])])
                dropouts = sweep.get('dropout', [float(bert_cfg['dropout'])])
                max_lengths = sweep.get('max_length', [int(bert_cfg['max_length'])])
                train_bsz = sweep.get('batch_size', [int(bert_cfg['batch_size'])])
                val_bsz = sweep.get('val_batch_size', [int(bert_cfg['val_batch_size'])])
                num_epochs_list = sweep.get('num_epochs', [int(bert_cfg['num_epochs'])])
                schedulers = sweep.get('scheduler', [bert_cfg.get('scheduler', None)])

                combos = list(itertools.product(lrs, dropouts, max_lengths, train_bsz, val_bsz, num_epochs_list, schedulers))
                strategy = getattr(args, 'search_strategy', 'grid')
                max_trials = getattr(args, 'max_trials', None)
                if strategy == 'random':
                    _random.shuffle(combos)
                if max_trials is not None:
                    try:
                        max_trials = int(max_trials)
                        combos = combos[:max_trials]
                    except Exception:
                        pass

                print("\n🤖 FreezedBertMlp スイープ開始: 試行数=\n" + str(len(combos)))
                for (lr_v, do_v, ml_v, tr_bs_v, va_bs_v, ne_v, sch_v) in combos:
                    dl_tr = _make_bert_loader(tr_subset, batch_size=int(tr_bs_v), num_workers=dl_workers, pin_memory=pin_mem, num_classes=num_classes)
                    dl_va = DataLoader(va_subset, batch_size=int(va_bs_v), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                    dl_te = DataLoader(ds_test, batch_size=int(va_bs_v), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)

                    model_trial = FreezedBertMlp(
                        model_name=bert_cfg['model_name'],
                        dropout=float(do_v),
                        max_length=int(ml_v),
                        device=str(device),
                        num_classes=num_classes,
                    )

                    sched_conf = None
                    if sch_v and isinstance(sch_v, dict) and sch_v.get('type', '').lower() == 'steplr':
                        sched_conf = {
                            'type': 'StepLR',
                            'step_size': int(sch_v.get('step_size', 5)),
                            'gamma': float(sch_v.get('gamma', 0.7)),
                        }

                    _ = train_bert_model(
                        model_trial,
                        dl_tr,
                        dl_va,
                        num_epochs=int(ne_v),
                        lr=float(lr_v),
                        device=str(device),
                        model_name=f"FreezedBertMlp (sweep Fold {fold_idx+1})",
                        early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                        verbose=True,
                        scheduler_config=sched_conf,
                        num_classes=num_classes,
                    )
                    met, _ = evaluate_bert_model(model_trial, dl_te, device=str(device), num_classes=num_classes)
                    variant_key = f"FreezedBertMlp[lr={lr_v},do={do_v},ml={ml_v},bs={tr_bs_v}]"
                    ensure_results_key(variant_key)
                    for k, v in met.items():
                        results[variant_key][k].append(v)
                    print(f"結果[{variant_key}]: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")
            else:
                print("\n🤖 FreezedBertMlp を学習中...")
                dl_tr = _make_bert_loader(tr_subset, batch_size=int(bert_cfg['batch_size']), num_workers=dl_workers, pin_memory=pin_mem, num_classes=num_classes)
                dl_va = DataLoader(va_subset, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                dl_te = DataLoader(ds_test, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                model_freezed = FreezedBertMlp(
                    model_name=bert_cfg['model_name'],
                    dropout=float(bert_cfg['dropout']),
                    max_length=int(bert_cfg['max_length']),
                    device=str(device),
                    num_classes=num_classes,
                )
                sched = None
                if bert_cfg.get('scheduler'):
                    sched = {
                        'type': bert_cfg['scheduler']['type'],
                        'step_size': int(bert_cfg['scheduler']['step_size']),
                        'gamma': float(bert_cfg['scheduler']['gamma']),
                    }
                _ = train_bert_model(
                    model_freezed,
                    dl_tr,
                    dl_va,
                    num_epochs=int(bert_cfg['num_epochs']),
                    lr=float(bert_cfg['learning_rate']),
                    device=str(device),
                    model_name=f"FreezedBertMlp (Fold {fold_idx+1})",
                    early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                    verbose=True,
                    scheduler_config=sched,
                    num_classes=num_classes,
                )
                met, _ = evaluate_bert_model(model_freezed, dl_te, device=str(device), num_classes=num_classes)
                for k, v in met.items():
                    results['FreezedBertMlp'][k].append(v)
                print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 3) FinetunedBertMlp
        if 'FinetunedBertMlp' in model_names:
            # このモデル用のデータセット/分割を準備
            ds_train = ABADataset(train_edges, all_nodes)
            ds_test = ABADataset(test_edges, all_nodes)
            tr_size = int((1 - val_split_ratio) * len(ds_train))
            va_size = len(ds_train) - tr_size
            if tr_size >= 1 and va_size >= 1:
                tr_subset, va_subset = torch.utils.data.random_split(ds_train, [tr_size, va_size])
            else:
                tr_subset, va_subset = ds_train, ds_test

            # best_from_results が指定されている場合は単一設定で実行
            if selected_ftb is not None:
                print("\n🤖 FinetunedBertMlp（best_from_results）を学習中...")
                dl_tr = _make_bert_loader(tr_subset, batch_size=int(selected_ftb['batch_size']), num_workers=dl_workers, pin_memory=pin_mem, num_classes=num_classes)
                dl_va = DataLoader(va_subset, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                dl_te = DataLoader(ds_test, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)

                model_trial = FinetunedBertMlp(
                    model_name=bert_cfg['model_name'],
                    dropout=float(selected_ftb['dropout']),
                    max_length=int(selected_ftb['max_length']),
                    device=str(device),
                    num_classes=num_classes,
                )
                _ = train_bert_model(
                    model_trial,
                    dl_tr,
                    dl_va,
                    num_epochs=int(bert_cfg['num_epochs']),
                    lr=float(selected_ftb['learning_rate']),
                    device=str(device),
                    model_name=f"FinetunedBertMlp (best Fold {fold_idx+1})",
                    early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                    verbose=True,
                    scheduler_config=None,
                    num_classes=num_classes,
                )
                met, _ = evaluate_bert_model(model_trial, dl_te, device=str(device), num_classes=num_classes)
                key = f"FinetunedBertMlp[lr={selected_ftb['learning_rate']},do={selected_ftb['dropout']},ml={selected_ftb['max_length']},bs={selected_ftb['batch_size']}]"
                ensure_results_key(key)
                for k, v in met.items():
                    results[key][k].append(v)
                print(f"結果[{key}]: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

            # スイープ設定がある場合はグリッド/ランダム探索を実行
            elif args and hasattr(args, 'sweep_config') and args.sweep_config:
                with open(args.sweep_config, 'r', encoding='utf-8') as f:
                    sweep = json.load(f)

                lrs = sweep.get('learning_rate', [float(bert_cfg['learning_rate'])])
                dropouts = sweep.get('dropout', [float(bert_cfg['dropout'])])
                max_lengths = sweep.get('max_length', [int(bert_cfg['max_length'])])
                train_bsz = sweep.get('batch_size', [int(bert_cfg['batch_size'])])
                val_bsz = sweep.get('val_batch_size', [int(bert_cfg['val_batch_size'])])
                num_epochs_list = sweep.get('num_epochs', [int(bert_cfg['num_epochs'])])
                schedulers = sweep.get('scheduler', [bert_cfg.get('scheduler', None)])

                combos = list(itertools.product(lrs, dropouts, max_lengths, train_bsz, val_bsz, num_epochs_list, schedulers))
                strategy = getattr(args, 'search_strategy', 'grid')
                max_trials = getattr(args, 'max_trials', None)
                if strategy == 'random':
                    _random.shuffle(combos)
                if max_trials is not None:
                    try:
                        max_trials = int(max_trials)
                        combos = combos[:max_trials]
                    except Exception:
                        pass

                print(f"\n🤖 FinetunedBertMlp スイープ開始: 試行数={len(combos)}")
                for (lr_v, do_v, ml_v, tr_bs_v, va_bs_v, ne_v, sch_v) in combos:
                    # データローダーを試行ごとに再構築
                    dl_tr = _make_bert_loader(tr_subset, batch_size=int(tr_bs_v), num_workers=dl_workers, pin_memory=pin_mem, num_classes=num_classes)
                    dl_va = DataLoader(va_subset, batch_size=int(va_bs_v), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                    dl_te = DataLoader(ds_test, batch_size=int(va_bs_v), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)

                    # モデル
                    model_trial = FinetunedBertMlp(
                        model_name=bert_cfg['model_name'],
                        dropout=float(do_v),
                        max_length=int(ml_v),
                        device=str(device),
                        num_classes=num_classes,
                    )

                    # スケジューラ
                    sched_conf = None
                    if sch_v and isinstance(sch_v, dict) and sch_v.get('type', '').lower() == 'steplr':
                        sched_conf = {
                            'type': 'StepLR',
                            'step_size': int(sch_v.get('step_size', 5)),
                            'gamma': float(sch_v.get('gamma', 0.7)),
                        }

                    _ = train_bert_model(
                        model_trial,
                        dl_tr,
                        dl_va,
                        num_epochs=int(ne_v),
                        lr=float(lr_v),
                        device=str(device),
                        model_name=f"FinetunedBertMlp (sweep Fold {fold_idx+1})",
                        early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                        verbose=True,
                        scheduler_config=sched_conf,
                        num_classes=num_classes,
                    )
                    met, _ = evaluate_bert_model(model_trial, dl_te, device=str(device), num_classes=num_classes)
                    variant_key = f"FinetunedBertMlp[lr={lr_v},do={do_v},ml={ml_v},bs={tr_bs_v}]"
                    ensure_results_key(variant_key)
                    for k, v in met.items():
                        results[variant_key][k].append(v)
                    print(f"結果[{variant_key}]: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")
            else:
                print("\n🤖 FinetunedBertMlp を学習中...")
                # DataLoader（設定のバッチサイズ）
                dl_tr = _make_bert_loader(tr_subset, batch_size=int(bert_cfg['batch_size']), num_workers=dl_workers, pin_memory=pin_mem, num_classes=num_classes)
                dl_va = DataLoader(va_subset, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                dl_te = DataLoader(ds_test, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)

                model_finetuned_mlp = FinetunedBertMlp(
                    model_name=bert_cfg['model_name'],
                    dropout=float(bert_cfg['dropout']),
                    max_length=int(bert_cfg['max_length']),
                    device=str(device),
                    num_classes=num_classes,
                )

                # 既定スケジューラ
                sched = None
                if bert_cfg.get('scheduler'):
                    sched = {
                        'type': bert_cfg['scheduler']['type'],
                        'step_size': int(bert_cfg['scheduler']['step_size']),
                        'gamma': float(bert_cfg['scheduler']['gamma']),
                    }

                _ = train_bert_model(
                    model_finetuned_mlp,
                    dl_tr,
                    dl_va,
                    num_epochs=int(bert_cfg['num_epochs']),
                    lr=float(bert_cfg['learning_rate']),
                    device=str(device),
                    model_name=f"FinetunedBertMlp (Fold {fold_idx+1})",
                    early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                    verbose=True,
                    scheduler_config=sched,
                    num_classes=num_classes,
                )
                met, _ = evaluate_bert_model(model_finetuned_mlp, dl_te, device=str(device), num_classes=num_classes)
                for k, v in met.items():
                    results['FinetunedBertMlp'][k].append(v)
                print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 4) FinetunedBertCosSim（シアミーズ + コサインロジット）
        if 'FinetunedBertCosSim' in model_names:
            # このモデル用のデータセット/分割を準備（独立に準備して未定義変数を排除）
            ds_train = ABADataset(train_edges, all_nodes)
            ds_test = ABADataset(test_edges, all_nodes)
            tr_size = int((1 - val_split_ratio) * len(ds_train))
            va_size = len(ds_train) - tr_size
            if tr_size >= 1 and va_size >= 1:
                tr_subset, va_subset = torch.utils.data.random_split(ds_train, [tr_size, va_size])
            else:
                tr_subset, va_subset = ds_train, ds_test

            # スイープ設定がある場合はグリッド/ランダム探索を実行
            if args and hasattr(args, 'sweep_config') and args.sweep_config and (not args.only_model or args.only_model == 'FinetunedBertCosSim'):
                with open(args.sweep_config, 'r', encoding='utf-8') as f:
                    sweep = json.load(f)

                lrs = sweep.get('learning_rate', [float(bert_cfg['learning_rate'])])
                max_lengths = sweep.get('max_length', [int(bert_cfg['max_length'])])
                train_bsz = sweep.get('batch_size', [int(bert_cfg['batch_size'])])
                val_bsz = sweep.get('val_batch_size', [int(bert_cfg['val_batch_size'])])
                num_epochs_list = sweep.get('num_epochs', [int(bert_cfg['num_epochs'])])
                schedulers = sweep.get('scheduler', [bert_cfg.get('scheduler', None)])

                combos = list(itertools.product(lrs, max_lengths, train_bsz, val_bsz, num_epochs_list, schedulers))
                strategy = getattr(args, 'search_strategy', 'grid')
                max_trials = getattr(args, 'max_trials', None)
                if strategy == 'random':
                    _random.shuffle(combos)
                if max_trials is not None:
                    try:
                        max_trials = int(max_trials)
                        combos = combos[:max_trials]
                    except Exception:
                        pass

                print(f"\n🤖 FinetunedBertCosSim スイープ開始: 試行数={len(combos)}")
                for (lr_v, ml_v, tr_bs_v, va_bs_v, ne_v, sch_v) in combos:
                    dl_tr = _make_bert_loader(tr_subset, batch_size=int(tr_bs_v), num_workers=dl_workers, pin_memory=pin_mem, num_classes=num_classes)
                    dl_va = DataLoader(va_subset, batch_size=int(va_bs_v), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                    dl_te = DataLoader(ds_test, batch_size=int(va_bs_v), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)

                    model_trial = FinetunedBertCosSim(
                        model_name=bert_cfg['model_name'],
                        max_length=int(ml_v),
                        device=str(device),
                        num_classes=num_classes,
                    )

                    sched_conf = None
                    if sch_v and isinstance(sch_v, dict) and sch_v.get('type', '').lower() == 'steplr':
                        sched_conf = {
                            'type': 'StepLR',
                            'step_size': int(sch_v.get('step_size', 2)),
                            'gamma': float(sch_v.get('gamma', 0.7)),
                        }

                    _ = train_bert_model(
                        model_trial,
                        dl_tr,
                        dl_va,
                        num_epochs=int(ne_v),
                        lr=float(lr_v),
                        device=str(device),
                        model_name=f"FinetunedBertCosSim (sweep Fold {fold_idx+1})",
                        early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                        verbose=True,
                        scheduler_config=sched_conf,
                        num_classes=num_classes,
                    )
                    met, _ = evaluate_bert_model(model_trial, dl_te, device=str(device), num_classes=num_classes)
                    variant_key = f"FinetunedBertCosSim[lr={lr_v},ml={ml_v},bs={tr_bs_v}]"
                    ensure_results_key(variant_key)
                    for k, v in met.items():
                        results[variant_key][k].append(v)
                    print(f"結果[{variant_key}]: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")
            else:
                print("\n🤖 FinetunedBertCosSim を学習中...")
                dl_tr = _make_bert_loader(tr_subset, batch_size=int(bert_cfg['batch_size']), num_workers=dl_workers, pin_memory=pin_mem, num_classes=num_classes)
                dl_va = DataLoader(va_subset, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                dl_te = DataLoader(ds_test, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=dl_workers, pin_memory=pin_mem)
                sched = None
                if bert_cfg.get('scheduler'):
                    sched = {
                        'type': bert_cfg['scheduler']['type'],
                        'step_size': int(bert_cfg['scheduler']['step_size']),
                        'gamma': float(bert_cfg['scheduler']['gamma']),
                    }
                model_cos = FinetunedBertCosSim(
                    model_name=bert_cfg['model_name'],
                    max_length=int(bert_cfg['max_length']),
                    device=str(device),
                    num_classes=num_classes,
                )
                _ = train_bert_model(
                    model_cos,
                    dl_tr,
                    dl_va,
                    num_epochs=int(bert_cfg['num_epochs']),
                    lr=float(bert_cfg['learning_rate']),
                    device=str(device),
                    model_name=f"FinetunedBertCosSim (Fold {fold_idx+1})",
                    early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                    verbose=True,
                    scheduler_config=sched,
                    num_classes=num_classes,
                )
                met, _ = evaluate_bert_model(model_cos, dl_te, device=str(device), num_classes=num_classes)
                for k, v in met.items():
                    results['FinetunedBertCosSim'][k].append(v)
                print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 5) TfidfLr
        if 'TfidfLr' in model_names:
            # スイープ設定がある場合はグリッド/ランダム探索を実行
            if args and hasattr(args, 'sweep_config') and args.sweep_config and (not args.only_model or args.only_model == 'TfidfLr'):
                print("\n📝 TfidfLr スイープを開始...")
                with open(args.sweep_config, 'r', encoding='utf-8') as f:
                    sweep = json.load(f)

                ngram_ranges = sweep.get('ngram_range', [[1, 1]])
                max_features_list = sweep.get('max_features', [1000])
                min_dfs = sweep.get('min_df', [1])
                Cs = sweep.get('C', [1.0])
                solvers = sweep.get('solver', ["liblinear"])
                class_weights = sweep.get('class_weight', [None])
                max_iters = sweep.get('max_iter', [200])

                combos = list(itertools.product(ngram_ranges, max_features_list, min_dfs, Cs, solvers, class_weights, max_iters))
                strategy = getattr(args, 'search_strategy', 'grid')
                max_trials = getattr(args, 'max_trials', None)
                if strategy == 'random':
                    _random.shuffle(combos)
                if max_trials is not None:
                    try:
                        max_trials = int(max_trials)
                        combos = combos[:max_trials]
                    except Exception:
                        pass

                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression

                for (ng_v, mf_v, md_v, c_v, solver_v, cw_v, mi_v) in combos:
                    tfidf = TfidfLr()
                    # ベクトライザと分類器を上書き構成
                    tfidf.vectorizer = TfidfVectorizer(
                        max_features=int(mf_v),
                        ngram_range=tuple(ng_v),
                        min_df=int(md_v)
                    )
                    tfidf.classifier = LogisticRegression(
                        C=float(c_v),
                        solver=str(solver_v),
                        class_weight=None if cw_v in [None, "null"] else str(cw_v),
                        max_iter=int(mi_v)
                    )
                    tfidf.fit(train_edges, all_nodes)
                    met, _ = evaluate_baseline(tfidf, test_edges, num_classes=num_classes)
                    key = f"TfidfLr[ng={tuple(ng_v)},mf={mf_v},min_df={md_v},C={c_v},solver={solver_v},cw={cw_v},mi={mi_v}]"
                    ensure_results_key(key)
                    for k, v in met.items():
                        results[key][k].append(v)
                    print(f"結果[{key}]: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")
            else:
                print("\n📝 TfidfLr を学習・評価中...")
                tfidf = TfidfLr(class_weight='balanced')
                tfidf.fit(train_edges, all_nodes)
                met, _ = evaluate_baseline(tfidf, test_edges, num_classes=num_classes)
                for k, v in met.items():
                    results['TfidfLr'][k].append(v)
                print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 6) Random
        if 'Random' in model_names:
            print("\n🎲 Random を評価中...")
            rnd = Random(num_classes=num_classes)
            met, _ = evaluate_baseline(rnd, test_edges, num_classes=num_classes)
            for k, v in met.items():
                results['Random'][k].append(v)
            print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

    # 統計・保存・表示・可視化（保存を最優先で実行）
    stats = calculate_statistics(results)
    tests = perform_statistical_tests(results)

    # 先に保存しておく（可視化で例外が出ても結果は残す）
    save_results_to_file(results, stats, tests, output_dir, config)

    # 表示（コンソール）
    display_results_table(stats, tests)

    # 可視化
    vis = config.get('visualization', {"enabled": True, "plots": ["box_plots", "bar_charts", "comprehensive_analysis"], "show_plots": False})
    if vis.get('enabled', True):
        if 'box_plots' in vis.get('plots', []):
            plot_box_plots(results, save_path=os.path.join(output_dir, 'box_plots.png'), show_plot=vis.get('show_plots', False))
        if 'bar_charts' in vis.get('plots', []):
            plot_bar_charts(stats, save_path=os.path.join(output_dir, 'bar_charts.png'), show_plot=vis.get('show_plots', False))
        if 'comprehensive_analysis' in vis.get('plots', []):
            # 空系列があると描画で落ちうるためフィルタし、念のためtryで保護
            filtered_results = {
                name: met for name, met in results.items()
                if len(met.get('accuracy', [])) > 0 and len(met.get('f1', [])) > 0 and len(met.get('auc', [])) > 0
            }
            if len(filtered_results) == 0:
                print("⚠️ comprehensive_analysis をスキップ（有効な系列がありません）")
            else:
                try:
                    plot_comprehensive_analysis(
                        filtered_results,
                        save_path=os.path.join(output_dir, 'comprehensive_analysis.png'),
                        show_plot=vis.get('show_plots', False)
                    )
                except Exception as e:
                    print(f"⚠️ comprehensive_analysis をスキップ（{e}）")
    print(f"\n✅ 完了: 出力は {output_dir} に保存しました")


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed set of named models (6 models) with 5-fold CV.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/robust_experiment.yaml',
        help='Path to configuration file (default: config/robust_experiment.yaml)'
    )
    parser.add_argument(
        '--sweep-config',
        type=str,
        default=None,
        help='Path to JSON file specifying hyperparameter search space for FinetunedBertMlp'
    )
    parser.add_argument(
        '--search-strategy',
        type=str,
        default='grid',
        choices=['grid', 'random'],
        help='Search strategy for sweep-config (grid or random)'
    )
    parser.add_argument(
        '--max-trials',
        type=int,
        default=None,
        help='Max trials to run for sweep (only with sweep-config)'
    )
    parser.add_argument(
        '--best-from-results',
        type=str,
        default=None,
        help='Use best FinetunedBertMlp (by F1 mean) from a previous experiment_results.json'
    )
    parser.add_argument(
        '--best-rgcn-from-results',
        type=str,
        default=None,
        help='Use best FreezedBertRgcnMlp (by F1 mean) from a previous experiment_results.json'
    )
    parser.add_argument(
        '--only-model',
        type=str,
        default=None,
        help='Run only one model from {FreezedBertRgcnMlp, FinetunedBertRgcnMlp, FreezedBertMlp, FinetunedBertMlp, FinetunedBertCosSim, TfidfLr, Random}'
    )
    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        dest='experiment_id',
        help='Base experiment ID (will be suffixed with _named_models)'
    )
    parser.add_argument(
        '--use-contrastive',
        action='store_true',
        dest='use_contrastive',
        help='Enable contrastive learning for embedding optimization'
    )
    args = parser.parse_args()
    run_named_models_experiment(args.config, args)


if __name__ == '__main__':
    main()



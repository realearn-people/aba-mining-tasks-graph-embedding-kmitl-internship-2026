"""
データサイズ別・損失収束分析 実行スクリプト
"""

import os
import sys
import json
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

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
from src.experiments.cross_validation import create_cross_validation_splits
from src.experiments.utils.subsample import (
    split_train_val,
    sample_node_induced,
    qc_minimum_counts,
)
from src.model_defs.models import (
    FreezedBertRgcnMlp,
    FreezedBertMlp,
    FinetunedBertMlp,
    FinetunedBertCosSim,
    TfidfLr,
)
from src.model_training.train import train_model
from src.model_training.train_bert import (
    ABADataset,
    train_bert_model,
    evaluate_bert_model,
)
from src.model_training.evaluate import evaluate_model, evaluate_baseline


def _ensure_graph_path(path_candidate: str) -> str:
    """指定パスが無ければ data/output/... にフォールバックする。"""
    if os.path.exists(path_candidate):
        return path_candidate
    base = os.path.basename(path_candidate)
    alt = os.path.join("data", "output", base)
    return alt


def _save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_history_csv(history: Dict[str, Any], path: str):
    """
    history: {'train_losses': [...], 'val_losses': [...], ...}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_losses = history.get("train_losses", []) or []
    val_losses = history.get("val_losses", []) or []
    rows = []
    max_len = max(len(train_losses), len(val_losses))
    for i in range(max_len):
        rows.append({
            "epoch": i + 1,
            "train_loss": train_losses[i] if i < len(train_losses) else "",
            "val_loss": val_losses[i] if i < len(val_losses) else "",
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)


def run_datasize_sweep(config_path: str):
    # 設定読み込み
    config = load_config(config_path)
    # グラフパスのフォールバック対応
    config['data']['input_graph'] = _ensure_graph_path(config['data']['input_graph'])

    # 実験ID/出力先
    experiment_id = determine_experiment_id(config, args=None) or "datasize_sweep"
    output_dir = setup_output_directory(config, experiment_id)
    config['data']['output_dir'] = output_dir
    config['data']['experiment_id'] = experiment_id

    # デバイス
    if config['compute']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['compute']['device'])
    print(f"\n💻 使用デバイス: {device}")

    # データ準備
    (
        original_graph,
        inference_graph,
        attack_edges,
        all_nodes,
        node_embeddings,
        node_to_idx,
        embedding_matrix,
        data,
    ) = prepare_data(config)

    # ネガティブサンプリング
    all_negatives = generate_negatives(
        original_graph, all_nodes, attack_edges,
        embedding_matrix, node_to_idx, inference_graph, config
    )

    # CV分割（固定の1フォールドを使用）
    cv_splits = create_cross_validation_splits(
        attack_edges, all_negatives, n_splits=5, seed=int(config['data']['seed'])
    )
    base_train_edges, test_edges = cv_splits[0]  # Fold-1 を固定使用

    # 検証データを固定分割（以後のサイズ変更では不変）
    val_ratio = float(config['experiment']['val_split_ratio'])
    train_base, val_fixed = split_train_val(
        base_train_edges, val_ratio, seed=int(config['data']['seed'])
    )
    print(f"\n📌 固定検証セット: {len(val_fixed)}件, 学習ベース: {len(train_base)}件")

    sizes: List[float] = list(config['experiment']['sizes'])
    seeds: List[int] = list(config['experiment']['seeds'])

    # まとめ出力の入れ物
    summary_rows: List[Dict[str, Any]] = []
    losses_rows: List[Dict[str, Any]] = []

    # モデル共通設定
    rgcn_cfg = config['models']['rgcn']
    bert_cfg = config['models']['improved_bert']

    # ループ: サイズ × シード
    for size in sizes:
        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"▶ データサイズ: {int(size*100)}%, seed={seed}")
            print(f"{'-'*70}")
            reduced_train, sampled_nodes, stats = sample_node_induced(train_base, size, seed)
            print(f"  抽出ノード数={stats['nodes']}, エッジ数={stats['edges']} (pos={stats['pos']}, neg={stats['neg']})")

            # QC
            min_pos = int(config['experiment']['min_positive'])
            min_neg = int(config['experiment']['min_negative'])
            if not qc_minimum_counts(reduced_train, min_pos, min_neg):
                print(f"⚠️ QC未達: pos>={min_pos}, neg>={min_neg} を満たさずスキップ")
                continue

            # 出力ディレクトリ（このサイズ・シードの共通ルート）
            run_root = os.path.join(output_dir, "runs", f"size_{int(size*100)}", f"seed_{seed}")
            os.makedirs(run_root, exist_ok=True)
            # メタ情報保存
            _save_json(
                {
                    "size": size,
                    "seed": seed,
                    "stats": stats,
                    "val_fixed": len(val_fixed),
                    "test_edges": len(test_edges),
                },
                os.path.join(run_root, "run_meta.json")
            )

            # 1) FreezedBertRgcnMlp
            if config['models']['rgcn']['enabled']:
                print("\n🔥 FreezedBertRgcnMlp を学習中...")
                rgcn_model = FreezedBertRgcnMlp(
                    input_dim=data.x.shape[1],
                    hidden_dim=int(rgcn_cfg['hidden_dim']),
                    num_layers=int(rgcn_cfg['num_layers']),
                    num_relations=1,
                    dropout_link=0.5
                )
                hist = train_model(
                    rgcn_model,
                    data,
                    reduced_train,
                    node_to_idx,
                    num_epochs=int(rgcn_cfg['num_epochs']),
                    lr=float(rgcn_cfg['learning_rate']),
                    model_name="FreezedBertRgcnMlp",
                    verbose=bool(rgcn_cfg.get('verbose', True)),
                    validation_edges=val_fixed,
                    device=str(device)
                )
                met, _ = evaluate_model(rgcn_model, data, test_edges, node_to_idx, device=str(device))
                model_dir = os.path.join(run_root, "FreezedBertRgcnMlp")
                _save_history_csv(hist, os.path.join(model_dir, "history.csv"))
                _save_json(met, os.path.join(model_dir, "metrics.json"))
                summary_rows.append({
                    "model": "FreezedBertRgcnMlp", "size": size, "seed": seed,
                    **met
                })
                for i, tr in enumerate(hist.get("train_losses", []) or []):
                    losses_rows.append({
                        "model": "FreezedBertRgcnMlp", "size": size, "seed": seed, "epoch": i+1,
                        "train_loss": tr,
                        "val_loss": (hist.get("val_losses", []) or [None]*len(hist.get("train_losses", [])))[i]
                        if i < len(hist.get("val_losses", []) or []) else None
                    })

            # BERT系のローダ準備（固定検証・固定テスト）
            ds_tr = ABADataset(reduced_train, all_nodes)
            ds_va = ABADataset(val_fixed, all_nodes)
            ds_te = ABADataset(test_edges, all_nodes)
            dl_tr = DataLoader(ds_tr, batch_size=int(bert_cfg['batch_size']), shuffle=True, num_workers=int(config['compute']['num_workers']), pin_memory=(str(device) == 'cuda'))
            dl_va = DataLoader(ds_va, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=int(config['compute']['num_workers']), pin_memory=(str(device) == 'cuda'))
            dl_te = DataLoader(ds_te, batch_size=int(bert_cfg['val_batch_size']), shuffle=False, num_workers=int(config['compute']['num_workers']), pin_memory=(str(device) == 'cuda'))

            # スケジューラ
            sched = None
            if bert_cfg.get('scheduler'):
                sched = {
                    'type': bert_cfg['scheduler']['type'],
                    'step_size': int(bert_cfg['scheduler']['step_size']),
                    'gamma': float(bert_cfg['scheduler']['gamma']),
                }

            # 2) FreezedBertMlp
            if config['models']['improved_bert']['enabled']:
                print("\n🤖 FreezedBertMlp を学習中...")
                model_freezed = FreezedBertMlp(
                    model_name=bert_cfg['model_name'],
                    dropout=float(bert_cfg['dropout']),
                    max_length=int(bert_cfg['max_length']),
                    device=str(device),
                )
                hist_b = train_bert_model(
                    model_freezed,
                    dl_tr,
                    dl_va,
                    num_epochs=int(bert_cfg['num_epochs']),
                    lr=float(bert_cfg['learning_rate']),
                    device=str(device),
                    model_name="FreezedBertMlp",
                    early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                    verbose=True,
                    scheduler_config=sched,
                )
                met_b, _ = evaluate_bert_model(model_freezed, dl_te, device=str(device))
                model_dir = os.path.join(run_root, "FreezedBertMlp")
                _save_history_csv(hist_b, os.path.join(model_dir, "history.csv"))
                _save_json(met_b, os.path.join(model_dir, "metrics.json"))
                summary_rows.append({
                    "model": "FreezedBertMlp", "size": size, "seed": seed, **met_b
                })
                for i, tr in enumerate(hist_b.get("train_losses", []) or []):
                    losses_rows.append({
                        "model": "FreezedBertMlp", "size": size, "seed": seed, "epoch": i+1,
                        "train_loss": tr,
                        "val_loss": (hist_b.get("val_losses", []) or [None]*len(hist_b.get("train_losses", [])))[i]
                        if i < len(hist_b.get("val_losses", []) or []) else None
                    })

            # 3) FinetunedBertMlp
            print("\n🤖 FinetunedBertMlp を学習中...")
            model_ft = FinetunedBertMlp(
                model_name=bert_cfg['model_name'],
                dropout=float(bert_cfg['dropout']),
                max_length=int(bert_cfg['max_length']),
                device=str(device),
            )
            hist_ft = train_bert_model(
                model_ft,
                dl_tr,
                dl_va,
                num_epochs=int(bert_cfg['num_epochs']),
                lr=float(bert_cfg['learning_rate']),
                device=str(device),
                model_name="FinetunedBertMlp",
                early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                verbose=True,
                scheduler_config=sched,
            )
            met_ft, _ = evaluate_bert_model(model_ft, dl_te, device=str(device))
            model_dir = os.path.join(run_root, "FinetunedBertMlp")
            _save_history_csv(hist_ft, os.path.join(model_dir, "history.csv"))
            _save_json(met_ft, os.path.join(model_dir, "metrics.json"))
            summary_rows.append({
                "model": "FinetunedBertMlp", "size": size, "seed": seed, **met_ft
            })
            for i, tr in enumerate(hist_ft.get("train_losses", []) or []):
                losses_rows.append({
                    "model": "FinetunedBertMlp", "size": size, "seed": seed, "epoch": i+1,
                    "train_loss": tr,
                    "val_loss": (hist_ft.get("val_losses", []) or [None]*len(hist_ft.get("train_losses", [])))[i]
                    if i < len(hist_ft.get("val_losses", []) or []) else None
                })

            # 4) FinetunedBertCosSim
            print("\n🤖 FinetunedBertCosSim を学習中...")
            model_cos = FinetunedBertCosSim(
                model_name=bert_cfg['model_name'],
                max_length=int(bert_cfg['max_length']),
                device=str(device),
            )
            hist_cos = train_bert_model(
                model_cos,
                dl_tr,
                dl_va,
                num_epochs=int(bert_cfg['num_epochs']),
                lr=float(bert_cfg['learning_rate']),
                device=str(device),
                model_name="FinetunedBertCosSim",
                early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
                verbose=True,
                scheduler_config=sched,
            )
            met_cos, _ = evaluate_bert_model(model_cos, dl_te, device=str(device))
            model_dir = os.path.join(run_root, "FinetunedBertCosSim")
            _save_history_csv(hist_cos, os.path.join(model_dir, "history.csv"))
            _save_json(met_cos, os.path.join(model_dir, "metrics.json"))
            summary_rows.append({
                "model": "FinetunedBertCosSim", "size": size, "seed": seed, **met_cos
            })
            for i, tr in enumerate(hist_cos.get("train_losses", []) or []):
                losses_rows.append({
                    "model": "FinetunedBertCosSim", "size": size, "seed": seed, "epoch": i+1,
                    "train_loss": tr,
                    "val_loss": (hist_cos.get("val_losses", []) or [None]*len(hist_cos.get("train_losses", [])))[i]
                    if i < len(hist_cos.get("val_losses", []) or []) else None
                })

            # 5) TfidfLr
            if config['models']['tfidf']['enabled']:
                print("\n📝 TfidfLr を学習・評価中...")
                tfidf_cfg = config['models']['tfidf']
                tfidf = TfidfLr(
                    max_features=int(tfidf_cfg.get('max_features', 1000)),
                    C=float(tfidf_cfg.get('C', 1.0)),
                    ngram_range=tuple(tfidf_cfg.get('ngram_range', [1, 1])),
                    solver=str(tfidf_cfg.get('solver', 'lbfgs')),
                    class_weight=tfidf_cfg.get('class_weight', None),
                    min_df=int(tfidf_cfg.get('min_df', 1)),
                    max_iter=int(tfidf_cfg.get('max_iter', 200)),
                )
                tfidf.fit(reduced_train, all_nodes)
                met_t, _ = evaluate_baseline(tfidf, test_edges)
                model_dir = os.path.join(run_root, "TfidfLr")
                os.makedirs(model_dir, exist_ok=True)
                # TF-IDFは履歴なしのためメトリクスのみ保存
                _save_json(met_t, os.path.join(model_dir, "metrics.json"))
                summary_rows.append({
                    "model": "TfidfLr", "size": size, "seed": seed, **met_t
                })

    # 集計出力
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 指標集計を保存: {summary_path}")

    losses_df = pd.DataFrame(losses_rows)
    losses_csv = os.path.join(output_dir, "summary_losses.csv")
    losses_df.to_csv(losses_csv, index=False)
    print(f"💾 損失履歴集計を保存: {losses_csv}")

    # 可能ならparquetでも保存（任意）
    try:
        losses_parquet = os.path.join(output_dir, "summary_losses.parquet")
        losses_df.to_parquet(losses_parquet, index=False)
        print(f"💾 損失履歴(parquet)を保存: {losses_parquet}")
    except Exception as e:
        print(f"⚠️ parquet保存をスキップ: {e}")

    print(f"\n✅ データサイズ別実験 完了: 出力は {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run dataset size sweep experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/datasize_sweep.yaml",
        help="Path to configuration file (default: config/datasize_sweep.yaml)",
    )
    args = parser.parse_args()
    run_datasize_sweep(args.config)


if __name__ == "__main__":
    main()



"""
データサイズ別・学習最適化（Grid Search）実行スクリプト
"""

import os
import sys
import json
import csv
import itertools
import copy
from typing import Dict, List, Any

import torch
import numpy as np
import pandas as pd
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


def _save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _ensure_graph_path(path_candidate: str) -> str:
    if os.path.exists(path_candidate):
        return path_candidate
    base = os.path.basename(path_candidate)
    alt = os.path.join("data", "output", base)
    return alt

def run_gridsearch_datasize(config_path: str):
    # 設定読み込み
    config = load_config(config_path)
    config['data']['input_graph'] = _ensure_graph_path(config['data']['input_graph'])

    # 実験ID/出力先
    experiment_id = determine_experiment_id(config, args=None) or "gridsearch_datasize"
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

    # CV分割（固定の1フォールドを使用 -> Hold-out）
    cv_splits = create_cross_validation_splits(
        attack_edges, all_negatives, n_splits=5, seed=int(config['data']['seed'])
    )
    base_train_edges, test_edges = cv_splits[0]

    # 検証データを固定分割
    val_ratio = float(config['experiment']['val_split_ratio'])
    train_base, val_fixed = split_train_val(
        base_train_edges, val_ratio, seed=int(config['data']['seed'])
    )
    print(f"\n📌 固定検証セット: {len(val_fixed)}件, 学習ベース: {len(train_base)}件")

    sizes = list(config['experiment']['sizes'])
    seed = int(config['data']['seed'])

    summary_rows = []
    best_params_log = []

    for size in sizes:
        print(f"\n{'='*70}")
        print(f"▶ データサイズ: {int(size*100)}%")
        print(f"{'-'*70}")
        
        reduced_train, sampled_nodes, stats = sample_node_induced(train_base, size, seed)
        print(f"  抽出ノード数={stats['nodes']}, エッジ数={stats['edges']} (pos={stats['pos']}, neg={stats['neg']})")

        min_pos = int(config.get('experiment', {}).get('min_positive', 10))
        min_neg = int(config.get('experiment', {}).get('min_negative', 10))
        if not qc_minimum_counts(reduced_train, min_pos, min_neg):
            print(f"⚠️ QC未達: pos>={min_pos}, neg>={min_neg} を満たさずスキップ")
            continue

        run_root = os.path.join(output_dir, "runs", f"size_{int(size*100)}")
        os.makedirs(run_root, exist_ok=True)
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

        # === Model Loop ===
        
        # 1. TfidfLr
        if config['models']['TfidfLr']['enabled']:
            print("\n🔎 TfidfLr Grid Search...")
            grid = config['models']['TfidfLr']['grid']
            keys = list(grid.keys())
            values = list(grid.values())
            
            best_score = -1.0
            best_params = {}
            
            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                # FIX 1: ngram_range conversion
                if 'ngram_range' in params and isinstance(params['ngram_range'], list):
                    params['ngram_range'] = tuple(params['ngram_range'])

                try:
                    model = TfidfLr(**params)
                    model.fit(reduced_train, all_nodes)
                    met_val, _ = evaluate_baseline(model, val_fixed)
                    score = met_val['auc']
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as e:
                    print(f"  Failed with params {params}: {e}")
            
            if best_score < 0:
                print("  ⚠️ No valid trial for TfidfLr. Skipping.")
            else:
                print(f"  Best TfidfLr Val AUC: {best_score:.4f} with {best_params}")
                final_model = TfidfLr(**best_params)
                final_model.fit(reduced_train, all_nodes)
                met_test, _ = evaluate_baseline(final_model, test_edges)
                
                model_dir = os.path.join(run_root, "TfidfLr")
                _save_json(best_params, os.path.join(model_dir, "best_params.json"))
                _save_json(met_test, os.path.join(model_dir, "metrics.json"))
                
                summary_rows.append({"model": "TfidfLr", "size": size, **met_test})
                best_params_log.append({"model": "TfidfLr", "size": size, **best_params})

        # 2. FreezedBertRgcnMlp
        if config['models']['FreezedBertRgcnMlp']['enabled']:
            print("\n🔎 FreezedBertRgcnMlp Grid Search...")
            cfg_fixed = config['models']['FreezedBertRgcnMlp']['fixed_params']
            grid = config['models']['FreezedBertRgcnMlp']['grid']
            keys = list(grid.keys())
            values = list(grid.values())
            
            best_score = -1.0
            best_params = {}
            
            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                print(f"  Trial: {params}")
                
                model = FreezedBertRgcnMlp(
                    input_dim=data.x.shape[1],
                    hidden_dim=int(params['hidden_dim']),
                    num_layers=2,
                    dropout_link=float(params['dropout_link'])
                )
                try:
                    # FIX 2: Remove patience arg (not supported by train_model)
                    train_model(
                        model, data, reduced_train, node_to_idx,
                        num_epochs=int(cfg_fixed['num_epochs']),
                        lr=float(params['learning_rate']),
                        device=str(device),
                        # patience=... Removed
                        validation_edges=val_fixed,
                        verbose=False 
                    )
                    met_val, _ = evaluate_model(model, data, val_fixed, node_to_idx, device=str(device))
                    score = met_val['auc']
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as e:
                    print(f"  Failed: {e}")

            # FIX 3: Check success
            if best_score < 0:
                print("  ⚠️ No valid trial for FreezedBertRgcnMlp. Skipping.")
            else:
                print(f"  Best RGCN Val AUC: {best_score:.4f} with {best_params}")
                
                final_model = FreezedBertRgcnMlp(
                    input_dim=data.x.shape[1],
                    hidden_dim=int(best_params['hidden_dim']),
                    num_layers=2,
                    dropout_link=float(best_params['dropout_link'])
                )
                train_model(
                    final_model, data, reduced_train, node_to_idx,
                    num_epochs=int(cfg_fixed['num_epochs']),
                    lr=float(best_params['learning_rate']),
                    device=str(device),
                    # patience=... Removed
                    validation_edges=val_fixed,
                    verbose=True
                )
                met_test, _ = evaluate_model(final_model, data, test_edges, node_to_idx, device=str(device))
                
                model_dir = os.path.join(run_root, "FreezedBertRgcnMlp")
                os.makedirs(model_dir, exist_ok=True)
                _save_json(best_params, os.path.join(model_dir, "best_params.json"))
                _save_json(met_test, os.path.join(model_dir, "metrics.json"))
                summary_rows.append({"model": "FreezedBertRgcnMlp", "size": size, **met_test})
                best_params_log.append({"model": "FreezedBertRgcnMlp", "size": size, **best_params})

        # BERT系共通データセット
        ds_tr = ABADataset(reduced_train, all_nodes)
        ds_va = ABADataset(val_fixed, all_nodes)
        ds_te = ABADataset(test_edges, all_nodes)

        # 3. FreezedBertMlp
        if config['models']['FreezedBertMlp']['enabled']:
            print("\n🔎 FreezedBertMlp Grid Search...")
            cfg_fixed = config['models']['FreezedBertMlp']['fixed_params']
            grid = config['models']['FreezedBertMlp']['grid']
            keys = list(grid.keys())
            values = list(grid.values())
            
            best_score = -1.0
            best_params = {}

            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                print(f"  Trial: {params}")
                
                dl_tr = DataLoader(ds_tr, batch_size=int(params.get('batch_size', 32)), shuffle=True, num_workers=0, pin_memory=(str(device)=='cuda'))
                dl_va = DataLoader(ds_va, batch_size=int(params.get('batch_size', 32)), shuffle=False, num_workers=0)
                
                model = FreezedBertMlp(
                    model_name=cfg_fixed['model_name'],
                    dropout=float(params['dropout']),
                    max_length=128,
                    device=str(device)
                )
                
                try:
                    train_bert_model(
                        model, dl_tr, dl_va,
                        num_epochs=int(cfg_fixed['num_epochs']),
                        lr=float(params['learning_rate']),
                        device=str(device),
                        early_stopping_patience=int(cfg_fixed['early_stopping_patience']),
                        verbose=False
                    )
                    met_val, _ = evaluate_bert_model(model, dl_va, device=str(device))
                    score = met_val['auc']
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as e:
                    print(f"  Failed: {e}")

            if best_score < 0:
                print("  ⚠️ No valid trial for FreezedBertMlp. Skipping.")
            else:
                print(f"  Best FreezedBertMlp Val AUC: {best_score:.4f} with {best_params}")
                
                dl_tr = DataLoader(ds_tr, batch_size=int(best_params.get('batch_size', 32)), shuffle=True, num_workers=0, pin_memory=(str(device)=='cuda'))
                dl_va = DataLoader(ds_va, batch_size=int(best_params.get('batch_size', 32)), shuffle=False, num_workers=0)
                dl_te = DataLoader(ds_te, batch_size=int(best_params.get('batch_size', 32)), shuffle=False, num_workers=0)
                
                final_model = FreezedBertMlp(
                    model_name=cfg_fixed['model_name'],
                    dropout=float(best_params['dropout']),
                    max_length=128,
                    device=str(device)
                )
                train_bert_model(
                    final_model, dl_tr, dl_va,
                    num_epochs=int(cfg_fixed['num_epochs']),
                    lr=float(best_params['learning_rate']),
                    device=str(device),
                    early_stopping_patience=int(cfg_fixed['early_stopping_patience']),
                    verbose=True
                )
                met_test, _ = evaluate_bert_model(final_model, dl_te, device=str(device))
                
                model_dir = os.path.join(run_root, "FreezedBertMlp")
                os.makedirs(model_dir, exist_ok=True)
                _save_json(best_params, os.path.join(model_dir, "best_params.json"))
                _save_json(met_test, os.path.join(model_dir, "metrics.json"))
                summary_rows.append({"model": "FreezedBertMlp", "size": size, **met_test})
                best_params_log.append({"model": "FreezedBertMlp", "size": size, **best_params})

        # 4. FinetunedBertMlp
        if config['models']['FinetunedBertMlp']['enabled']:
            print("\n🔎 FinetunedBertMlp Grid Search...")
            cfg_fixed = config['models']['FinetunedBertMlp']['fixed_params']
            grid = config['models']['FinetunedBertMlp']['grid']
            keys = list(grid.keys())
            values = list(grid.values())
            best_score = -1.0
            best_params = {}

            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                print(f"  Trial: {params}")
                dl_tr = DataLoader(ds_tr, batch_size=int(params['batch_size']), shuffle=True, num_workers=0, pin_memory=(str(device)=='cuda'))
                dl_va = DataLoader(ds_va, batch_size=int(params['batch_size']), shuffle=False, num_workers=0)
                
                model = FinetunedBertMlp(
                    model_name=cfg_fixed['model_name'],
                    dropout=0.3,
                    max_length=128,
                    device=str(device)
                )
                try:
                    train_bert_model(
                        model, dl_tr, dl_va,
                        num_epochs=int(cfg_fixed['num_epochs']),
                        lr=float(params['learning_rate']),
                        device=str(device),
                        early_stopping_patience=int(cfg_fixed['early_stopping_patience']),
                        verbose=False
                    )
                    met_val, _ = evaluate_bert_model(model, dl_va, device=str(device))
                    score = met_val['auc']
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as e:
                    print(f"  Failed: {e}")
            
            if best_score < 0:
                print("  ⚠️ No valid trial for FinetunedBertMlp. Skipping.")
            else:
                print(f"  Best FinetunedBertMlp Val AUC: {best_score:.4f} with {best_params}")
                
                dl_tr = DataLoader(ds_tr, batch_size=int(best_params['batch_size']), shuffle=True, num_workers=0, pin_memory=(str(device)=='cuda'))
                dl_va = DataLoader(ds_va, batch_size=int(best_params['batch_size']), shuffle=False, num_workers=0)
                dl_te = DataLoader(ds_te, batch_size=int(best_params['batch_size']), shuffle=False, num_workers=0)
                final_model = FinetunedBertMlp(
                    model_name=cfg_fixed['model_name'],
                    dropout=0.3,
                    max_length=128,
                    device=str(device)
                )
                train_bert_model(
                    final_model, dl_tr, dl_va,
                    num_epochs=int(cfg_fixed['num_epochs']),
                    lr=float(best_params['learning_rate']),
                    device=str(device),
                    early_stopping_patience=int(cfg_fixed['early_stopping_patience']),
                    verbose=True
                )
                met_test, _ = evaluate_bert_model(final_model, dl_te, device=str(device))
                model_dir = os.path.join(run_root, "FinetunedBertMlp")
                os.makedirs(model_dir, exist_ok=True)
                _save_json(best_params, os.path.join(model_dir, "best_params.json"))
                _save_json(met_test, os.path.join(model_dir, "metrics.json"))
                summary_rows.append({"model": "FinetunedBertMlp", "size": size, **met_test})
                best_params_log.append({"model": "FinetunedBertMlp", "size": size, **best_params})

        # 5. FinetunedBertCosSim
        if config['models']['FinetunedBertCosSim']['enabled']:
            print("\n🔎 FinetunedBertCosSim Grid Search...")
            cfg_fixed = config['models']['FinetunedBertCosSim']['fixed_params']
            grid = config['models']['FinetunedBertCosSim']['grid']
            keys = list(grid.keys())
            values = list(grid.values())
            best_score = -1.0
            best_params = {}

            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                print(f"  Trial: {params}")
                dl_tr = DataLoader(ds_tr, batch_size=int(params['batch_size']), shuffle=True, num_workers=0, pin_memory=(str(device)=='cuda'))
                dl_va = DataLoader(ds_va, batch_size=int(params['batch_size']), shuffle=False, num_workers=0)
                
                model = FinetunedBertCosSim(
                    model_name=cfg_fixed['model_name'],
                    max_length=128,
                    device=str(device)
                )
                try:
                    train_bert_model(
                        model, dl_tr, dl_va,
                        num_epochs=int(cfg_fixed['num_epochs']),
                        lr=float(params['learning_rate']),
                        device=str(device),
                        early_stopping_patience=int(cfg_fixed['early_stopping_patience']),
                        verbose=False
                    )
                    met_val, _ = evaluate_bert_model(model, dl_va, device=str(device))
                    score = met_val['auc']
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as e:
                    print(f"  Failed: {e}")
            
            if best_score < 0:
                 print("  ⚠️ No valid trial for FinetunedBertCosSim. Skipping.")
            else:
                print(f"  Best CosSim Val AUC: {best_score:.4f} with {best_params}")

                dl_tr = DataLoader(ds_tr, batch_size=int(best_params['batch_size']), shuffle=True, num_workers=0, pin_memory=(str(device)=='cuda'))
                dl_va = DataLoader(ds_va, batch_size=int(best_params['batch_size']), shuffle=False, num_workers=0)
                dl_te = DataLoader(ds_te, batch_size=int(best_params['batch_size']), shuffle=False, num_workers=0)
                final_model = FinetunedBertCosSim(
                    model_name=cfg_fixed['model_name'],
                    max_length=128,
                    device=str(device)
                )
                train_bert_model(
                    final_model, dl_tr, dl_va,
                    num_epochs=int(cfg_fixed['num_epochs']),
                    lr=float(best_params['learning_rate']),
                    device=str(device),
                    early_stopping_patience=int(cfg_fixed['early_stopping_patience']),
                    verbose=True
                )
                met_test, _ = evaluate_bert_model(final_model, dl_te, device=str(device))
                model_dir = os.path.join(run_root, "FinetunedBertCosSim")
                os.makedirs(model_dir, exist_ok=True)
                _save_json(best_params, os.path.join(model_dir, "best_params.json"))
                _save_json(met_test, os.path.join(model_dir, "metrics.json"))
                summary_rows.append({"model": "FinetunedBertCosSim", "size": size, **met_test})
                best_params_log.append({"model": "FinetunedBertCosSim", "size": size, **best_params})

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    
    params_df = pd.DataFrame(best_params_log)
    if not params_df.empty:
        params_df.to_csv(os.path.join(output_dir, "summary_best_params.csv"), index=False)
    
    print(f"\n✅ Experiment Completed. Results saved to {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gridsearch_datasize.yaml")
    args = parser.parse_args()
    run_gridsearch_datasize(args.config)

if __name__ == "__main__":
    main()

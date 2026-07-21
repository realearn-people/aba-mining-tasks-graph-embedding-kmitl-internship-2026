#!/usr/bin/env python
import os
import json
import argparse
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401


def load_best_from_results(results_path: str) -> Tuple[str, Dict[str, float]]:
    """
    experiment_results.json から、Accuracy(mean) が最大のバリアントを抽出し、
    その Accuracy/AUC/F1 の mean を返す。
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stats = data.get('statistics', {})
    best_key = None
    best_acc = -1.0
    for name, metrics in stats.items():
        acc_mean = None
        try:
            acc_mean = float(metrics.get('accuracy', {}).get('mean', None))
        except Exception:
            acc_mean = None
        if acc_mean is not None and acc_mean > best_acc:
            best_acc = acc_mean
            best_key = name
    if best_key is None:
        raise ValueError(f"No valid entries found in statistics at {results_path}")
    best = stats[best_key]
    out = {
        'accuracy': float(best.get('accuracy', {}).get('mean', np.nan)),
        'auc': float(best.get('auc', {}).get('mean', np.nan)),
        'f1': float(best.get('f1', {}).get('mean', np.nan)),
    }
    return best_key, out


def plot_bars(values: Dict[str, float], title: str, save_path: str, ylabel: str):
    models = list(values.keys())
    scores = [values[m] for m in models]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, scores, color='#4C72B0')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # annotate
    for b, s in zip(bars, scores):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{s:.3f}", ha='center', va='bottom')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Root directory under data/training_results')
    args = ap.parse_args()

    # 探索するサブディレクトリ（モデル名で固定）
    model_dirs = [
        'FreezedBertRgcnMlp',
        'FreezedBertMlp',
        'FinetunedBertMlp',
        'FinetunedBertCosSim',
        'TfidfLr',
    ]
    best_acc: Dict[str, float] = {}
    best_auc: Dict[str, float] = {}
    best_f1: Dict[str, float] = {}

    for m in model_dirs:
        res_path = os.path.join(args.root, m, 'experiment_results.json')
        if not os.path.isfile(res_path):
            print(f"Skip {m}: results not found -> {res_path}")
            continue
        _, metrics = load_best_from_results(res_path)
        best_acc[m] = metrics['accuracy']
        best_auc[m] = metrics['auc']
        best_f1[m] = metrics['f1']

    # 可視化
    plot_bars(best_acc, '各モデルのベストAccuracy（mean, 5-fold）', os.path.join(args.root, 'best_accuracy_bars.png'), 'Accuracy')
    plot_bars(best_auc, '各モデルのベストAUC（mean, 5-fold）', os.path.join(args.root, 'best_auc_bars.png'), 'AUC')
    plot_bars(best_f1, '各モデルのベストF1（mean, 5-fold）', os.path.join(args.root, 'best_f1_bars.png'), 'F1')


if __name__ == '__main__':
    main()



"""
データサイズ別実験の結果を可視化するスクリプト。
summary_metrics.csv と summary_losses.csv を読み込み、
性能推移グラフと学習曲線を描画します。
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # 日本語対応

def plot_size_vs_metric(df: pd.DataFrame, metric: str, output_dir: str):
    """データサイズ vs 指標の折れ線グラフを作成"""
    plt.figure(figsize=(10, 6))
    
    # 平均と信頼区間（または標準偏差）を計算
    # seedごとのばらつきを帯で表現
    sns.lineplot(
        data=df,
        x="size",
        y=metric,
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="tab10",
        err_style="band",  # 信頼区間の帯を表示
        ci=95  # 95%信頼区間
    )
    
    plt.title(f"データセットサイズと {metric.capitalize()} の関係", fontsize=14)
    plt.xlabel("データセットサイズ (割合)", fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"size_vs_{metric}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_loss_curves_by_size(df_loss: pd.DataFrame, output_dir: str):
    """サイズごとの学習曲線（Train/Val Loss）を作成"""
    # サイズごとにプロット
    sizes = sorted(df_loss['size'].unique())
    
    for size in sizes:
        # そのサイズのデータのみ抽出
        subset = df_loss[df_loss['size'] == size]
        if subset.empty:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # モデルごとにサブプロットを作成するか、重ねて描画するか
        # ここではモデルごとに色を変え、Trainは実線、Valは点線で描画
        
        models = subset['model'].unique()
        palette = sns.color_palette("tab10", n_colors=len(models))
        
        for i, model in enumerate(models):
            model_data = subset[subset['model'] == model]
            # シード平均をとる
            mean_data = model_data.groupby('epoch')[['train_loss', 'val_loss']].mean().reset_index()
            
            color = palette[i]
            plt.plot(mean_data['epoch'], mean_data['train_loss'], label=f"{model} (Train)", color=color, linestyle='-')
            # Val Lossがあれば描画（欠損値NaNは無視される）
            if mean_data['val_loss'].notna().any():
                plt.plot(mean_data['epoch'], mean_data['val_loss'], label=f"{model} (Val)", color=color, linestyle='--')

        plt.title(f"学習曲線 (データサイズ: {int(size*100)}%)", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.yscale('log')  # Lossは見やすさのため対数軸にすることも検討（今回は線形で様子見、必要なら切替）
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"loss_curves_size_{int(size*100)}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Directory containing summary_metrics.csv and summary_losses.csv")
    args = parser.parse_args()
    
    metrics_path = os.path.join(args.input, "summary_metrics.csv")
    losses_path = os.path.join(args.input, "summary_losses.csv")
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    # Metrics Plotting
    df_metrics = pd.read_csv(metrics_path)
    
    # 主要指標のプロット
    for metric in ["accuracy", "auc", "f1"]:
        if metric in df_metrics.columns:
            plot_size_vs_metric(df_metrics, metric, args.input)
            
    # Loss Plotting
    if os.path.exists(losses_path):
        df_losses = pd.read_csv(losses_path)
        plot_loss_curves_by_size(df_losses, args.input)
    else:
        print(f"Warning: {losses_path} not found. Skipping loss curves.")

if __name__ == "__main__":
    main()


"""
BERT モデル学習用のモジュール
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np


class ABADataset(Dataset):
    """
    ABA Link Prediction用データセット
    """
    def __init__(self, edges, all_nodes):
        """
        Args:
            edges: List of ((assumption, proposition), label)
            all_nodes: List of all node texts
        """
        self.edges = edges
        self.all_nodes = all_nodes
        
    def __len__(self):
        return len(self.edges)
    
    def __getitem__(self, idx):
        (assumption, proposition), label = self.edges[idx]
        return {
            'assumption': assumption,
            'proposition': proposition,
            'label': float(label)
        }


def train_bert_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3,
                     device='cpu', model_name="BERT", early_stopping_patience=5,
                     verbose=True, scheduler_config=None, num_classes=2):
    """
    BERTモデルの学習（詳細な学習過程記録付き）
    
    Args:
        model: BERTモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        num_epochs: エポック数
        lr: 学習率
        device: 計算デバイス
        model_name: モデル名（表示用）
        early_stopping_patience: 早期終了のための待機エポック数
        verbose: 詳細表示フラグ
        scheduler_config: スケジューラー設定 (dict or None)
    
    Returns:
        dict: 学習過程の情報
    """
    model = model.to(device)
    
    if verbose:
        # モデルパラメータ情報表示
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 {model_name} パラメータ情報:")
        print(f"  総パラメータ数: {total_params:,}")
        print(f"  学習可能パラメータ数: {trainable_params:,}")
        print(f"  固定パラメータ数: {total_params - trainable_params:,}")
        print(f"  学習対象比率: {trainable_params/total_params*100:.2f}%")
        
        # 学習設定表示
        print(f"\n⚙️  {model_name} 学習設定:")
        print(f"  エポック数: {num_epochs}")
        print(f"  学習率: {lr}")
        print(f"  バッチサイズ: {train_loader.batch_size if hasattr(train_loader, 'batch_size') else '可変'}")
        print(f"  最適化手法: Adam")
        print(f"  早期終了: patience={early_stopping_patience}")
        print("-" * 50)
    
    if num_classes == 2:
        # compute pos_weight from training labels to counter class imbalance
        all_labels = torch.cat([batch['label'] for batch in train_loader])
        n_pos = all_labels.sum().item()
        n_neg = len(all_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if verbose:
            print(f"  クラス重み (binary): neg/pos = {pos_weight.item():.2f}  "
                  f"(pos={int(n_pos)}, neg={int(n_neg)})")
    else:
        # compute class weights from training labels to counter multi-class imbalance
        all_labels = torch.cat([batch['label'] for batch in train_loader]).long()
        from collections import Counter as _Counter
        counts = _Counter(all_labels.tolist())
        total = len(all_labels)
        raw_w = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
        min_w = min(raw_w)
        capped_w = [min(w, min_w * 5) for w in raw_w]
        class_weights = torch.tensor(capped_w, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if verbose:
            print(f"  クラス重み ({num_classes}-class): "
                  + ", ".join(f"cls{c}={class_weights[c].item():.2f}" for c in range(num_classes)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    
    # スケジューラーの設定
    scheduler = None
    if scheduler_config:
        if scheduler_config['type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=scheduler_config.get('step_size', 5),
                gamma=scheduler_config.get('gamma', 0.7)
            )
            if verbose:
                print(f"  スケジューラー: StepLR (step_size={scheduler_config['step_size']}, gamma={scheduler_config['gamma']})")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    if verbose:
        print(f"\n🚀 {model_name} 学習開始...")
        print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                
                assumptions = batch['assumption']
                propositions = batch['proposition']
                labels = batch['label'].to(device)
                
                # バッチサイズをチェック
                if len(assumptions) == 0:
                    continue
                
                outputs = model(assumptions, propositions)
                lbl = labels.long() if num_classes > 2 else labels.float()
                loss = criterion(outputs, lbl)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # 進捗表示（10バッチごと）
                if verbose and batch_idx % 10 == 0 and batch_idx > 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"    Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss = {loss.item():.4f}, LR = {current_lr:.2e}")
                
            except Exception as batch_error:
                if verbose:
                    print(f"    ⚠️ Epoch {epoch+1}, Batch {batch_idx} 処理エラー: {batch_error}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    assumptions = batch['assumption']
                    propositions = batch['proposition']
                    labels = batch['label'].to(device)
                    
                    if len(assumptions) == 0:
                        continue
                    
                    outputs = model(assumptions, propositions)
                    lbl = labels.long() if num_classes > 2 else labels.float()
                    loss = criterion(outputs, lbl)
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                except Exception as val_error:
                    if verbose:
                        print(f"    ⚠️ バリデーションエラー: {val_error}")
                    continue
        
        # 損失計算
        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 早期終了判定
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if verbose:
                print(f"    ✅ 新しい最良モデル (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if verbose:
                print(f"    ⏳ 改善なし {patience_counter}/{early_stopping_patience} エポック")
        
        # スケジューラーステップ
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if verbose:
                print(f"    📉 学習率更新: {current_lr:.2e}")
        
        epoch_time = time.time() - epoch_start_time
        
        # エポック結果表示
        if verbose:
            print(f"    📊 Epoch {epoch+1}/{num_epochs} 完了: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, "
                  f"Time = {epoch_time:.2f}s")
            print("-" * 60)
        
        # 早期終了チェック
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"    🛑 早期終了: {early_stopping_patience} エポック連続で改善なし")
                print(f"    📈 最良検証損失: {best_val_loss:.4f} で学習終了")
            break
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n✅ {model_name} 学習完了!")
        print(f"   総学習時間: {total_time:.2f}秒")
        print(f"   最終訓練損失: {train_losses[-1]:.4f}")
        print(f"   最終検証損失: {val_losses[-1]:.4f}")
        print(f"   最良検証損失: {best_val_loss:.4f}")
    
    # 学習結果をまとめて返す
    training_info = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_time,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'model_name': model_name,
        'num_epochs': len(train_losses),
        'learning_rate': lr
    }
    
    return training_info


def evaluate_bert_model(model, test_loader, device='cpu', num_classes=2):
    """
    BERTモデルの評価
    
    Args:
        model: BERTモデル
        test_loader: テストデータローダー
        device: 計算デバイス
    
    Returns:
        dict: 評価メトリクス
        np.array: 予測確率
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                assumptions = batch['assumption']
                propositions = batch['proposition']
                labels = batch['label'].to(device)
                
                if len(assumptions) == 0:
                    continue
                
                outputs = model(assumptions, propositions)
                if num_classes == 2:
                    probs = torch.sigmoid(outputs)
                    predictions = (probs > 0.5).float()
                    all_probs.extend(probs.cpu().numpy())
                else:
                    probs = torch.softmax(outputs, dim=-1)
                    predictions = torch.argmax(probs, dim=-1).float()
                    all_probs.extend(probs.cpu().numpy())

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"⚠️ 評価中のエラー: {e}")
                continue
    
    if len(all_labels) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0,
                'mean_rank': 0.0, 'mrr': 0.0, 'hits@1_micro': 0.0, 'hits@1_macro': 0.0, 'hits@3': 0.0, 'hits@10': 0.0}, np.array([])

    from sklearn.metrics import accuracy_score as _acc, precision_recall_fscore_support as _prf
    from src.model_training.evaluate import compute_relation_ranking_metrics

    all_labels_np = np.array(all_labels)
    all_preds_np  = np.array(all_predictions)
    all_probs_np  = np.array(all_probs)

    accuracy = _acc(all_labels_np, all_preds_np)

    if num_classes == 2:
        precision, recall, f1, _ = _prf(all_labels_np, all_preds_np, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np) if len(set(all_labels)) > 1 else 0.0
        except Exception:
            auc = 0.0
        ranking = compute_relation_ranking_metrics(all_probs_np, all_labels_np)
    else:
        precision, recall, f1, _ = _prf(all_labels_np, all_preds_np, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro') \
                  if len(set(all_labels)) > 1 else 0.0
        except Exception:
            auc = 0.0
        ranking = compute_relation_ranking_metrics(all_probs_np, all_labels_np)

    metrics = {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auc':       auc,
        **ranking,
    }

    return metrics, all_probs_np


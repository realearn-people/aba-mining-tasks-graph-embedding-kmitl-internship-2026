import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# BERT / LoRA 関連のインポート
try:
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers or peft not available. BERT-based models will not work.")

class AttackLinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_relations=1, num_classes=2):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes

        # R-GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(input_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        # Link prediction head
        out_dim = 1 if num_classes == 2 else num_classes
        head = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, out_dim),
        ]
        if num_classes == 2:
            head.append(nn.Sigmoid())
        self.link_predictor = nn.Sequential(*head)
    
    def forward(self, x, edge_index, edge_type, edge_pairs):
        # R-GCN forward pass
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_type))
        
        # Link prediction
        edge_embeddings = []
        for u, v in edge_pairs:
            u_emb = h[u]
            v_emb = h[v]
            edge_emb = torch.cat([u_emb, v_emb], dim=0)
            edge_embeddings.append(edge_emb)

        edge_embeddings = torch.stack(edge_embeddings)
        predictions = self.link_predictor(edge_embeddings)

        # 2-class: squeeze to (N,); 3-class: keep (N, num_classes)
        return predictions.squeeze(-1) if self.num_classes == 2 else predictions

class RandomBaseline:
    """ランダムベースライン"""
    def __init__(self):
        pass
    
    def predict(self, edge_pairs):
        return np.random.random(len(edge_pairs))

class BERTCosineSimilarityBaseline:
    """BERTコサイン類似度ベースライン"""
    def __init__(self, node_embeddings, node_to_idx):
        self.node_embeddings = node_embeddings
        self.node_to_idx = node_to_idx
        self.embedding_matrix = np.array([node_embeddings[node] for node in sorted(node_to_idx.keys())])
    
    def predict(self, edge_pairs):
        similarities = []
        for u, v in edge_pairs:
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            u_emb = self.embedding_matrix[u_idx]
            v_emb = self.embedding_matrix[v_idx]
            sim = cosine_similarity([u_emb], [v_emb])[0][0]
            similarities.append(sim)
        return np.array(similarities)

class TFIDFLogisticRegressionBaseline:
    """TF-IDF + ロジスティック回帰ベースライン"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = LogisticRegression()
    
    def fit(self, train_edges, all_nodes):
        # TF-IDF特徴量を作成
        node_texts = [node for node in all_nodes] # node_texts は "close_to_airport" などのテキストブロック
        tfidf_matrix = self.vectorizer.fit_transform(node_texts) # tfidf_matrix は "close_to_airport" などのノード名そのものの TF-IDF ベクトル
        
        # エッジペア特徴量を作成
        node_to_tfidf_idx = {node: i for i, node in enumerate(all_nodes)} # node_to_tfidf_idx は "close_to_airport" などのノード名をインデックスに変換した辞書
        
        X_train = [] # X_train は エッジペア特徴量
        y_train = [] # y_train は エッジペアのラベル
        
        for (u, v), label in train_edges:
            u_idx = node_to_tfidf_idx[u] # u_idx は u の TF-IDF ベクトルのインデックス
            v_idx = node_to_tfidf_idx[v] # v_idx は v の TF-IDF ベクトルのインデックス
            u_vec = tfidf_matrix[u_idx].toarray()[0]
            v_vec = tfidf_matrix[v_idx].toarray()[0]
            
            # 特徴量を結合
            features = np.concatenate([u_vec, v_vec, u_vec * v_vec])  # concat, element-wise product
            X_train.append(features)
            y_train.append(label)
        
        self.classifier.fit(X_train, y_train)
        self.tfidf_matrix = tfidf_matrix
        self.node_to_tfidf_idx = node_to_tfidf_idx
    
    def predict(self, edge_pairs):
        X_test = []
        for u, v in edge_pairs:
            u_idx = self.node_to_tfidf_idx[u]
            v_idx = self.node_to_tfidf_idx[v]
            u_vec = self.tfidf_matrix[u_idx].toarray()[0]
            v_vec = self.tfidf_matrix[v_idx].toarray()[0]
            
            features = np.concatenate([u_vec, v_vec, u_vec * v_vec])
            X_test.append(features)
        
        proba = self.classifier.predict_proba(X_test)
        # Binary: return P(class=1); Multi-class: return full proba matrix
        return proba[:, 1] if proba.shape[1] == 2 else proba


# =============================================================================
# BERT-based Models
# =============================================================================

class ImprovedBERTLinkPredictor(nn.Module):
    """
    改良版BERT Link Predictor（固定BERT + 学習可能線形層）
    
    小データセットに適した設計:
    - BERTパラメータを固定し、分類層のみ学習
    - ノードの埋め込みを別々に取得してから結合
    """
    def __init__(self, model_name='google-bert/bert-base-uncased',
                 dropout=0.3, max_length=128, freeze_bert=True, device='cpu', num_classes=2):
        super(ImprovedBERTLinkPredictor, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for BERT models")

        self.model_name = model_name
        self.max_length = max_length
        self.freeze_bert = freeze_bert
        self.device = device
        self.num_classes = num_classes

        # BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768
        out_dim = 1 if num_classes == 2 else num_classes

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 4, out_dim),
        )
        
    def forward(self, assumption_texts, proposition_texts):
        """
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
        
        Returns:
            logits: Tensor of shape (batch_size,)
        """
        # Encode assumptions
        assumption_inputs = self.tokenizer(
            assumption_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        assumption_inputs = {k: v.to(self.device) for k, v in assumption_inputs.items()}
        
        # Encode propositions
        proposition_inputs = self.tokenizer(
            proposition_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        proposition_inputs = {k: v.to(self.device) for k, v in proposition_inputs.items()}
        
        # Get BERT outputs
        assumption_outputs = self.bert(**assumption_inputs)
        proposition_outputs = self.bert(**proposition_inputs)
        
        # Use CLS token embeddings
        assumption_emb = assumption_outputs.last_hidden_state[:, 0]  # [CLS] token
        proposition_emb = proposition_outputs.last_hidden_state[:, 0]  # [CLS] token
        
        combined = torch.cat([assumption_emb, proposition_emb], dim=-1)
        logits = self.classifier(combined)
        return logits.squeeze(-1) if self.num_classes == 2 else logits


class CrossEncoderBERTLinkPredictor(nn.Module):
    """
    Cross-Encoder方式でノードペア関係を直接学習するBERTモデル
    
    ノードペアを単一のシーケンスとして処理し、
    BERT内部のアテンション機構でペア関係を学習
    """
    def __init__(self, model_name='google-bert/bert-base-uncased',
                 dropout=0.3, max_length=256, freeze_bert=True, device='cpu', num_classes=2):
        super(CrossEncoderBERTLinkPredictor, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for BERT models")

        self.model_name = model_name
        self.max_length = max_length
        self.freeze_bert = freeze_bert
        self.device = device
        self.num_classes = num_classes

        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768
        out_dim = 1 if num_classes == 2 else num_classes

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 4, out_dim),
        )
        
    def forward(self, assumption_texts, proposition_texts):
        """
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
        
        Returns:
            logits: Tensor of shape (batch_size,)
        """
        # Cross-encoder: ノードペアを単一シーケンスとして処理
        paired_texts = []
        for assumption, proposition in zip(assumption_texts, proposition_texts):
            # [CLS] assumption [SEP] proposition [SEP] 形式
            paired_text = f"{assumption} [SEP] {proposition}"
            paired_texts.append(paired_text)
        
        # トークン化
        inputs = self.tokenizer(
            paired_texts, 
            padding=True, 
            truncation=True,
            max_length=self.max_length, 
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # BERT処理
        outputs = self.bert(**inputs)
        
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits.squeeze(-1) if self.num_classes == 2 else logits


# =============================================================================
# New Named Model Classes (CamelCase as requested)
# =============================================================================

class FreezedBertRgcnMlp(AttackLinkPredictor):
    """
    Freezed-BERT + R-GCN + MLP に相当。
    - BERT初期埋め込みはパイプライン側で生成し、入力特徴として利用します。
    - link predictor の dropout をハイパラから指定可能にします。
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_relations=1,
                 dropout_link: float = 0.5, num_classes: int = 2):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                         num_relations=num_relations, num_classes=num_classes)
        out_dim = 1 if num_classes == 2 else num_classes
        head = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_link),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_link),
            nn.Linear(64, out_dim),
        ]
        if num_classes == 2:
            head.append(nn.Sigmoid())
        self.link_predictor = nn.Sequential(*head)


class FreezedBertMlp(ImprovedBERTLinkPredictor):
    """固定BERT + MLP。"""
    def __init__(self, model_name='google-bert/bert-base-uncased', dropout=0.3, max_length=128,
                 device='cpu', num_classes=2):
        super(FreezedBertMlp, self).__init__(
            model_name=model_name, dropout=dropout, max_length=max_length,
            freeze_bert=True, device=device, num_classes=num_classes,
        )


class FinetunedBertMlp(ImprovedBERTLinkPredictor):
    """BERT微調整あり + MLP。"""
    def __init__(self, model_name='google-bert/bert-base-uncased', dropout=0.3, max_length=128,
                 device='cpu', num_classes=2):
        super(FinetunedBertMlp, self).__init__(
            model_name=model_name, dropout=dropout, max_length=max_length,
            freeze_bert=False, device=device, num_classes=num_classes,
        )


class FinetunedBertCosSim(nn.Module):
    """
    BERTを微調整し、ノードペアの埋め込みのコサイン類似度をスコア化するモデル。
    - 出力はロジット（BCEWithLogitsLossに入力可能）として返すため、
      正規化コサインに学習可能スケールを掛けた値をロジットとみなす。
    """
    def __init__(self, model_name='google-bert/bert-base-uncased', max_length=128, device='cpu', num_classes=2):
        super(FinetunedBertCosSim, self).__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for BERT models")
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.num_classes = num_classes

        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 微調整を有効にする（freezeしない）
        for p in self.bert.parameters():
            p.requires_grad = True

        hidden_size = self.bert.config.hidden_size
        if num_classes == 2:
            # learnable scale and bias; bias initialised negative so model starts near "not contrary"
            self.logit_scale = nn.Parameter(torch.tensor(10.0))
            self.logit_bias = nn.Parameter(torch.tensor(-5.0))
            self.classifier = None
        else:
            self.logit_scale = None
            self.logit_bias = None
            self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        cls = outputs.last_hidden_state[:, 0]
        # 単位ベクトル化
        cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
        return cls

    def forward(self, assumption_texts, proposition_texts):
        a = self.encode(assumption_texts)
        b = self.encode(proposition_texts)
        if self.num_classes == 2:
            cos = (a * b).sum(dim=-1)
            return self.logit_scale * cos + self.logit_bias
        else:
            return self.classifier(torch.cat([a, b], dim=-1))


class FinetunedBertRgcnMlp(AttackLinkPredictor):
    """
    Finetuned-BERT + R-GCN + MLP モデル。

    - ノードテキストを BERT でエンコードし、その CLS ベクトルをノード特徴とする
    - BERT パラメータは微調整（freeze しない）
    - R-GCN によりグラフ構造を学習し、ノードペア埋め込みを MLP で分類

    AttackLinkPredictor 互換のインターフェイス:
        forward(x, edge_index, edge_type, edge_pairs)
    ただし x は無視され、内部で BERT 埋め込みを計算する。
    """
    def __init__(
        self,
        all_nodes,
        model_name: str = "google-bert/bert-base-uncased",
        max_length: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_relations: int = 1,
        dropout_link: float = 0.5,
        num_classes: int = 2,
        # LoRA hyperparameters (overridable via runner/sweep)
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules=None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and peft libraries are required for FinetunedBertRgcnMlp")

        # BERT 本体とトークナイザ（まだ self に登録しない）
        model_name_local = model_name
        max_length_local = max_length
        all_nodes_list = list(all_nodes)

        bert_model = AutoModel.from_pretrained(model_name_local)
        tokenizer = AutoTokenizer.from_pretrained(model_name_local)

        # まず BERT の既存パラメータを固定し、LoRA だけを学習対象とする
        for p in bert_model.parameters():
            p.requires_grad = False

        # LoRA 設定（BERT の attention / FFN 層に低ランク適応を追加）
        if lora_target_modules is None:
            lora_target_modules = ["query", "value"]
        lora_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            # 特徴抽出用途（CLS 埋め込み取得）なので FEATURE_EXTRACTION を指定
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=lora_target_modules,
        )
        bert_model = get_peft_model(bert_model, lora_config)
        # LoRA 部分のみ学習対象
        assert any(p.requires_grad for p in bert_model.parameters()), "LoRA parameters are not trainable."

        # ノード列を一括トークナイズ
        encoded = tokenizer(
            all_nodes_list,
            padding=True,
            truncation=True,
            max_length=max_length_local,
            return_tensors="pt",
        )

        # AttackLinkPredictor を BERT hidden size を入力次元として初期化
        hidden_size = bert_model.config.hidden_size
        super().__init__(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_relations=num_relations,
            num_classes=num_classes,
        )

        # BERT 関連をモジュール属性として登録
        self.model_name = model_name_local
        self.max_length = max_length_local
        self.all_nodes = all_nodes_list
        self.bert = bert_model
        self.tokenizer = tokenizer
        # BERT に対して gradient checkpointing を有効化（メモリ削減）
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()

        # BERT エンコード時のミニバッチサイズ（GPU メモリとスループットのバランス用）
        self.bert_batch_size = 64

        # link_predictor を FreezedBertRgcnMlp と同様に dropout_link で上書き
        out_dim = 1 if num_classes == 2 else num_classes
        head = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_link),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_link),
            nn.Linear(64, out_dim),
        ]
        if num_classes == 2:
            head.append(nn.Sigmoid())
        self.link_predictor = nn.Sequential(*head)

        # トークナイズ済み入力をバッファとして登録（.to(device) に追従させる）
        self.register_buffer("input_ids", encoded["input_ids"])
        self.register_buffer("attention_mask", encoded["attention_mask"])

    def _encode_nodes_with_bert(self) -> torch.Tensor:
        """
        全ノードを BERT でエンコードし、CLS 埋め込み行列 (num_nodes, hidden_size) を返す。

        メモリ使用量を抑えるため、ノードをミニバッチに分割して順次 BERT に通す。
        BERT 自体は R-GCN と同じデバイス（通常は GPU）上に載せる想定。
        """
        bert_device = next(self.bert.parameters()).device
        input_ids = self.input_ids
        attention_mask = self.attention_mask
        num_nodes = input_ids.size(0)

        cls_list = []
        bs = self.bert_batch_size
        for start in range(0, num_nodes, bs):
            end = start + bs
            batch_inputs = {
                "input_ids": input_ids[start:end].to(bert_device),
                "attention_mask": attention_mask[start:end].to(bert_device),
            }
            outputs = self.bert(**batch_inputs)
            cls_batch = outputs.last_hidden_state[:, 0]  # (batch, hidden)
            cls_list.append(cls_batch)

        cls = torch.cat(cls_list, dim=0)  # (num_nodes, hidden_size)
        return cls

    def forward(self, x, edge_index, edge_type, edge_pairs):
        """
        Args:
            x: 既存パイプラインとの互換性のためのダミー（内部では使用しない）
            edge_index: グラフのエッジインデックス
            edge_type: エッジ種別（R-GCN 用）
            edge_pairs: 学習・評価対象のノードペア（ノードインデックスのタプル列）
        """
        # BERT でノード埋め込みを計算（R-GCN と同じデバイス想定）
        h = self._encode_nodes_with_bert()

        # R-GCN 伝播
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_type))

        # Link prediction（AttackLinkPredictor と同じ形式）
        edge_embeddings = []
        for u, v in edge_pairs:
            u_emb = h[u]
            v_emb = h[v]
            edge_emb = torch.cat([u_emb, v_emb], dim=0)
            edge_embeddings.append(edge_emb)

        edge_embeddings = torch.stack(edge_embeddings)
        predictions = self.link_predictor(edge_embeddings)
        return predictions.squeeze(-1) if self.num_classes == 2 else predictions


class TfidfLr(TFIDFLogisticRegressionBaseline):
    """TF-IDF + Logistic Regression（名称整備）。"""
    def __init__(self, max_features=1000, C=1.0, ngram_range=(1, 1), solver='lbfgs', class_weight=None, min_df=1, max_iter=200, **kwargs):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
        self.classifier = LogisticRegression(C=C, solver=solver, class_weight=class_weight, max_iter=max_iter)


class Random(nn.Module):
    """ランダムベースライン（名称整備）。"""
    def __init__(self, num_classes=2):
        super().__init__()
        self._impl = RandomBaseline()
        self.num_classes = num_classes

    def predict(self, edge_pairs):
        n = len(edge_pairs)
        if self.num_classes == 2:
            return self._impl.predict(edge_pairs)
        # Dirichlet-sampled random probabilities for each class
        return np.random.dirichlet(np.ones(self.num_classes), size=n)
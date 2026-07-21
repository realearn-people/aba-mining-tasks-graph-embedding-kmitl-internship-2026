"""BERT-based classifier for ABA link prediction."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional, Dict


class BERTLinkPredictor(nn.Module):
    """BERT-based model for ABA link prediction."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        hidden_dim: int = 768,
        num_classes: int = 1,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        pooling_strategy: str = 'cls'
    ):
        """
        Initialize BERT link predictor.
        
        Args:
            model_name: Pretrained BERT model name
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
            pooling_strategy: 'cls', 'mean', or 'max'
        """
        super(BERTLinkPredictor, self).__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get BERT output dimension
        bert_hidden_size = self.bert.config.hidden_size
        
        # Classification head for link prediction
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool BERT embeddings based on strategy.
        
        Args:
            hidden_states: BERT hidden states
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        if self.pooling_strategy == 'cls':
            return hidden_states[:, 0]
        elif self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            # Max pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            return torch.max(hidden_states, 1)[0]
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")
    
    def encode_text(
        self,
        text: List[str],
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Encode text using BERT.
        
        Args:
            text: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Text embeddings
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        encoded = {k: v.to(next(self.bert.parameters()).device) for k, v in encoded.items()}
        
        # Get BERT outputs
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert(**encoded)
        
        # Pool embeddings
        embeddings = self.pool_embeddings(outputs.last_hidden_state, encoded['attention_mask'])
        
        return embeddings
    
    def forward(
        self,
        assumption_texts: List[str],
        proposition_texts: List[str],
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
            max_length: Maximum sequence length
            
        Returns:
            Link predictions
        """
        # Encode assumptions and propositions
        assumption_emb = self.encode_text(assumption_texts, max_length)
        proposition_emb = self.encode_text(proposition_texts, max_length)
        
        # Concatenate embeddings
        combined = torch.cat([assumption_emb, proposition_emb], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits
    
    def predict(
        self,
        assumption_texts: List[str],
        proposition_texts: List[str],
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Make predictions with probabilities.
        
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
            max_length: Maximum sequence length
            
        Returns:
            Predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(assumption_texts, proposition_texts, max_length)
            return torch.sigmoid(logits)


class CrossEncoderBERT(nn.Module):
    """Cross-encoder BERT for ABA link prediction."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 1,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        """
        Initialize cross-encoder BERT.
        
        Args:
            model_name: Pretrained BERT model name
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT
        """
        super(CrossEncoderBERT, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        bert_hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_hidden_size, num_classes)
        
    def forward(
        self,
        assumption_texts: List[str],
        proposition_texts: List[str],
        max_length: int = 256
    ) -> torch.Tensor:
        """
        Forward pass using cross-encoding.
        
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
            max_length: Maximum sequence length
            
        Returns:
            Link predictions
        """
        # Create paired inputs with [SEP] token
        paired_texts = []
        for assumption, proposition in zip(assumption_texts, proposition_texts):
            paired_texts.append(f"{assumption} [SEP] {proposition}")
        
        # Tokenize
        encoded = self.tokenizer(
            paired_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        encoded = {k: v.to(next(self.bert.parameters()).device) for k, v in encoded.items()}
        
        # Get BERT outputs
        outputs = self.bert(**encoded)
        
        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits
    
    def predict(
        self,
        assumption_texts: List[str],
        proposition_texts: List[str],
        max_length: int = 256
    ) -> torch.Tensor:
        """
        Make predictions with probabilities.
        
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
            max_length: Maximum sequence length
            
        Returns:
            Predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(assumption_texts, proposition_texts, max_length)
            return torch.sigmoid(logits)
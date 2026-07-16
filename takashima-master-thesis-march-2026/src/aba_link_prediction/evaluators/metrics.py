"""Evaluation metrics for ABA link prediction."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class LinkPredictionMetrics:
    """Metrics for link prediction evaluation."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Decision threshold for binary classification
        """
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """Reset accumulated predictions and labels."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            probabilities: Prediction probabilities
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
        
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }
        
        # Add AUC and AP if probabilities are available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            metrics['roc_auc'] = roc_auc_score(labels, probabilities)
            metrics['average_precision'] = average_precision_score(labels, probabilities)
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.labels, self.predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(
            self.labels,
            self.predictions,
            target_names=['Non-Contrary', 'Contrary']
        )


class GraphMetrics:
    """Metrics specific to graph-based models."""
    
    @staticmethod
    def hits_at_k(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        k: int = 10
    ) -> float:
        """
        Compute Hits@K metric.
        
        Args:
            pos_scores: Scores for positive edges
            neg_scores: Scores for negative edges
            k: Top-k value
            
        Returns:
            Hits@K score
        """
        num_pos = pos_scores.shape[0]
        num_neg = neg_scores.shape[0]
        
        # Combine scores
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(num_pos),
            torch.zeros(num_neg)
        ])
        
        # Get top-k indices
        _, top_indices = torch.topk(scores, min(k, len(scores)))
        top_labels = labels[top_indices]
        
        # Calculate hits
        hits = top_labels.sum().item()
        return hits / min(k, num_pos)
    
    @staticmethod
    def mean_reciprocal_rank(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            pos_scores: Scores for positive edges
            neg_scores: Scores for negative edges
            
        Returns:
            MRR score
        """
        num_pos = pos_scores.shape[0]
        
        reciprocal_ranks = []
        
        for i in range(num_pos):
            pos_score = pos_scores[i]
            
            # Count how many negative scores are higher
            higher_scores = (neg_scores > pos_score).sum().item()
            
            # Rank is 1-indexed
            rank = higher_scores + 1
            reciprocal_ranks.append(1.0 / rank)
        
        return np.mean(reciprocal_ranks)


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = device
        self.metrics = LinkPredictionMetrics()
        
    def evaluate_dataset(
        self,
        data_loader: torch.utils.data.DataLoader,
        model_type: str = 'bert'
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: Data loader
            model_type: Type of model ('bert' or 'rgcn')
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        all_pos_scores = []
        all_neg_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Get predictions based on model type
                if model_type == 'bert':
                    outputs = self.model(
                        batch['assumption_text'],
                        batch['proposition_text']
                    )
                elif model_type == 'rgcn':
                    outputs = self.model(
                        batch.get('x'),
                        batch['edge_index'],
                        batch.get('edge_type'),
                        batch.get('target_edges')
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Convert to probabilities
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                # Update metrics
                self.metrics.update(
                    predictions,
                    batch['label'],
                    probabilities
                )
                
                # Separate positive and negative scores for graph metrics
                labels = batch['label']
                pos_mask = labels == 1
                neg_mask = labels == 0
                
                if pos_mask.any():
                    all_pos_scores.append(outputs[pos_mask])
                if neg_mask.any():
                    all_neg_scores.append(outputs[neg_mask])
        
        # Compute standard metrics
        metrics = self.metrics.compute()
        
        # Add graph-specific metrics if applicable
        if all_pos_scores and all_neg_scores:
            pos_scores = torch.cat(all_pos_scores)
            neg_scores = torch.cat(all_neg_scores)
            
            metrics['hits@1'] = GraphMetrics.hits_at_k(pos_scores, neg_scores, k=1)
            metrics['hits@3'] = GraphMetrics.hits_at_k(pos_scores, neg_scores, k=3)
            metrics['hits@10'] = GraphMetrics.hits_at_k(pos_scores, neg_scores, k=10)
            metrics['mrr'] = GraphMetrics.mean_reciprocal_rank(pos_scores, neg_scores)
        
        return metrics
    
    def print_evaluation_report(
        self,
        metrics: Dict[str, float],
        dataset_name: str = 'Test'
    ):
        """
        Print formatted evaluation report.
        
        Args:
            metrics: Dictionary of metrics
            dataset_name: Name of dataset
        """
        print(f"\n{'='*50}")
        print(f"{dataset_name} Set Evaluation Results")
        print(f"{'='*50}")
        
        # Standard metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1-Score:  {metrics.get('f1', 0):.4f}")
        
        # Ranking metrics
        if 'roc_auc' in metrics:
            print("\nRanking Metrics:")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Avg Prec:  {metrics.get('average_precision', 0):.4f}")
        
        # Graph metrics
        if 'hits@1' in metrics:
            print("\nGraph-specific Metrics:")
            print(f"  Hits@1:    {metrics['hits@1']:.4f}")
            print(f"  Hits@3:    {metrics['hits@3']:.4f}")
            print(f"  Hits@10:   {metrics['hits@10']:.4f}")
            print(f"  MRR:       {metrics.get('mrr', 0):.4f}")
        
        print(f"{'='*50}\n")
        
        # Print confusion matrix
        print("Confusion Matrix:")
        cm = self.metrics.get_confusion_matrix()
        print(cm)
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(self.metrics.get_classification_report())


def compare_models(
    models: Dict[str, torch.nn.Module],
    test_loader: torch.utils.data.DataLoader,
    model_types: Dict[str, str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of model name to model
        test_loader: Test data loader
        model_types: Dictionary of model name to type
        device: Device to use
        
    Returns:
        Dictionary of model name to metrics
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        evaluator = ModelEvaluator(model, device)
        
        metrics = evaluator.evaluate_dataset(
            test_loader,
            model_type=model_types[name]
        )
        
        evaluator.print_evaluation_report(metrics, dataset_name=name)
        results[name] = metrics
    
    # Print comparison table
    print("\n" + "="*80)
    print("Model Comparison Summary")
    print("="*80)
    
    # Header
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    header = "Model".ljust(20) + " | " + " | ".join(m.ljust(10) for m in metric_names)
    print(header)
    print("-" * len(header))
    
    # Rows
    for name, metrics in results.items():
        row = name.ljust(20) + " | "
        row += " | ".join(f"{metrics.get(m, 0):.4f}".ljust(10) for m in metric_names)
        print(row)
    
    print("="*80)
    
    return results
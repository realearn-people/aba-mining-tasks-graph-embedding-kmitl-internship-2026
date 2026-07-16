"""Training utilities for ABA link prediction models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer class for ABA models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_type: Optional[str] = 'cosine',
        warmup_steps: int = 0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay
            scheduler_type: Learning rate scheduler type
            warmup_steps: Number of warmup steps
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = None
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        if self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps
            )
        elif self.scheduler_type == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.warmup_steps
            )
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self._forward_batch(batch)
            labels = batch['label']
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler and epoch > self.warmup_steps:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self._forward_batch(batch)
                labels = batch['label']
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """
        Forward pass for a batch.
        Should be overridden by subclasses.
        
        Args:
            batch: Batch data
            
        Returns:
            Model outputs
        """
        raise NotImplementedError
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        early_stopping_patience: int = 5
    ) -> Dict:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save model
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        self._setup_scheduler(total_steps)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f'Epoch {epoch}/{num_epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_dir:
                    save_path = save_dir / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }, save_path)
                    logger.info(f'  Saved best model to {save_path}')
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    break
        
        return self.history


class RGCNTrainer(BaseTrainer):
    """Trainer for RGCN models."""
    
    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """Forward pass for RGCN."""
        # For graph data, we need edge_index and labels
        # This assumes the batch contains the necessary graph structure
        edge_index = batch['edge_index']
        x = batch.get('x', None)
        edge_type = batch.get('edge_type', None)
        target_edges = batch.get('target_edges', edge_index)
        
        return self.model(x, edge_index, edge_type, target_edges)


class BERTTrainer(BaseTrainer):
    """Trainer for BERT models."""
    
    def _forward_batch(self, batch: Dict) -> torch.Tensor:
        """Forward pass for BERT."""
        assumption_texts = batch['assumption_text']
        proposition_texts = batch['proposition_text']
        
        return self.model(assumption_texts, proposition_texts)


class ContrastiveTrainer:
    """Trainer for contrastive learning."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        temperature: float = 0.07
    ):
        """
        Initialize contrastive trainer.
        
        Args:
            model: Contrastive model
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay
            temperature: Temperature for contrastive loss
        """
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # History
        self.history = {'loss': []}
        
    def train_epoch(
        self,
        data_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train one epoch.
        
        Args:
            data_loader: Data loader
            epoch: Current epoch
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(data_loader, desc=f'Contrastive Epoch {epoch}')
        
        for batch in pbar:
            # Move batch to device
            x = batch['x'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_type = batch.get('edge_type', None)
            
            if edge_type is not None:
                edge_type = edge_type.to(self.device)
            
            # Generate two views
            if hasattr(self.model, 'forward'):
                z1, z2 = self.model(x, edge_index, edge_type)
            else:
                # For GraphContrastiveLearning
                z1, z2 = self.model(x, edge_index, edge_type)
            
            # Compute loss
            loss = self.model.compute_loss(z1, z2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def train(
        self,
        data_loader: DataLoader,
        num_epochs: int,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """
        Train contrastive model.
        
        Args:
            data_loader: Data loader
            num_epochs: Number of epochs
            save_dir: Directory to save model
            
        Returns:
            Training history
        """
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(data_loader, epoch)
            self.history['loss'].append(loss)
            
            logger.info(f'Contrastive Epoch {epoch}/{num_epochs}: Loss = {loss:.4f}')
            
            # Save checkpoint
            if save_dir and epoch % 10 == 0:
                save_path = save_dir / f'contrastive_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss
                }, save_path)
        
        return self.history
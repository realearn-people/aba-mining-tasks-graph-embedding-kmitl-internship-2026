"""Balanced dataset with negative sampling for ABA link prediction."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class BalancedABADataset(Dataset):
    """Balanced dataset with negative sampling for ABA link prediction."""
    
    def __init__(
        self, 
        data_path: str,
        mode: str = 'train',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        balance_ratio: float = 1.0,  # ratio of negative to positive samples
        sampling_strategy: str = 'undersample'  # 'undersample', 'oversample', or 'hybrid'
    ):
        """
        Initialize balanced ABA Dataset.
        
        Args:
            data_path: Path to CSV file
            mode: 'train', 'val', or 'test'
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
            balance_ratio: Ratio of negative to positive samples (1.0 = equal)
            sampling_strategy: How to balance the dataset
        """
        self.data_path = data_path
        self.mode = mode
        self.random_state = random_state
        self.balance_ratio = balance_ratio
        self.sampling_strategy = sampling_strategy
        
        # Load and process data
        self.df = pd.read_csv(data_path)
        
        # Log original class distribution
        original_pos = self.df['isContrary'].sum()
        original_neg = len(self.df) - original_pos
        logger.info(f"Original distribution - Positive: {original_pos}, Negative: {original_neg}")
        logger.info(f"Original ratio - Positive: {original_pos/len(self.df):.2%}, Negative: {original_neg/len(self.df):.2%}")
        
        # Encode assumptions and propositions
        self.assumption_encoder = LabelEncoder()
        self.proposition_encoder = LabelEncoder()
        
        self.df['assumption_id'] = self.assumption_encoder.fit_transform(self.df['Assumption'])
        self.df['proposition_id'] = self.proposition_encoder.fit_transform(self.df['Proposition'])
        
        # Split data BEFORE balancing (to ensure test set remains unbiased)
        train_val_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state, stratify=self.df['isContrary']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=random_state, 
            stratify=train_val_df['isContrary']
        )
        
        # Apply balancing only to training data
        if mode == 'train':
            self.data = self._balance_dataset(train_df)
            logger.info(f"Training set after balancing: {len(self.data)} samples")
        elif mode == 'val':
            self.data = val_df
        elif mode == 'test':
            self.data = test_df
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'")
        
        self.data = self.data.reset_index(drop=True)
        
        # Log final class distribution
        final_pos = self.data['isContrary'].sum()
        final_neg = len(self.data) - final_pos
        logger.info(f"Final {mode} distribution - Positive: {final_pos}, Negative: {final_neg}")
        logger.info(f"Final {mode} ratio - Positive: {final_pos/len(self.data):.2%}, Negative: {final_neg/len(self.data):.2%}")
        
    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset using the specified strategy.
        
        Args:
            df: Input dataframe
            
        Returns:
            Balanced dataframe
        """
        positive_samples = df[df['isContrary'] == True]
        negative_samples = df[df['isContrary'] == False]
        
        n_positive = len(positive_samples)
        n_negative = len(negative_samples)
        
        if self.sampling_strategy == 'undersample':
            # Undersample negative class
            n_negative_balanced = int(n_positive * self.balance_ratio)
            if n_negative_balanced < n_negative:
                negative_samples = negative_samples.sample(
                    n=n_negative_balanced, 
                    random_state=self.random_state
                )
            
            balanced_df = pd.concat([positive_samples, negative_samples])
            
        elif self.sampling_strategy == 'oversample':
            # Oversample positive class
            n_positive_balanced = int(n_negative / self.balance_ratio)
            if n_positive_balanced > n_positive:
                # Oversample with replacement
                positive_samples = positive_samples.sample(
                    n=n_positive_balanced, 
                    replace=True,
                    random_state=self.random_state
                )
            
            balanced_df = pd.concat([positive_samples, negative_samples])
            
        elif self.sampling_strategy == 'hybrid':
            # Combination of oversampling and undersampling
            # Target: make both classes closer to their geometric mean
            target_size = int(np.sqrt(n_positive * n_negative))
            
            # Oversample positive if needed
            if n_positive < target_size:
                positive_samples = positive_samples.sample(
                    n=target_size,
                    replace=True,
                    random_state=self.random_state
                )
            else:
                positive_samples = positive_samples.sample(
                    n=target_size,
                    random_state=self.random_state
                )
            
            # Undersample negative
            negative_target = int(target_size * self.balance_ratio)
            if negative_target < n_negative:
                negative_samples = negative_samples.sample(
                    n=negative_target,
                    random_state=self.random_state
                )
            
            balanced_df = pd.concat([positive_samples, negative_samples])
            
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return balanced_df
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at index.
        
        Returns:
            Dictionary with:
                - assumption_id: Encoded assumption ID
                - proposition_id: Encoded proposition ID
                - assumption_text: Original assumption text
                - proposition_text: Original proposition text
                - label: Binary label (0 or 1)
        """
        row = self.data.iloc[idx]
        
        return {
            'assumption_id': torch.tensor(row['assumption_id'], dtype=torch.long),
            'proposition_id': torch.tensor(row['proposition_id'], dtype=torch.long),
            'assumption_text': row['Assumption'],
            'proposition_text': row['Proposition'],
            'label': torch.tensor(int(row['isContrary']), dtype=torch.float)
        }
    
    def get_num_nodes(self) -> Tuple[int, int]:
        """Return number of unique assumptions and propositions."""
        return (
            len(self.assumption_encoder.classes_),
            len(self.proposition_encoder.classes_)
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Get class weights for weighted loss functions.
        
        Returns:
            Tensor of class weights [weight_for_0, weight_for_1]
        """
        n_samples = len(self.data)
        n_positive = self.data['isContrary'].sum()
        n_negative = n_samples - n_positive
        
        # Inverse frequency weighting
        weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
        weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0
        
        return torch.tensor([weight_negative, weight_positive], dtype=torch.float)


class BalancedGraphDataset:
    """Graph-based dataset with balanced sampling for ABA link prediction."""
    
    def __init__(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        balance_ratio: float = 1.0,
        sampling_strategy: str = 'undersample'
    ):
        """
        Initialize balanced graph dataset.
        
        Args:
            data_path: Path to CSV file
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
            balance_ratio: Ratio of negative to positive samples
            sampling_strategy: How to balance the dataset
        """
        self.data_path = data_path
        self.random_state = random_state
        self.balance_ratio = balance_ratio
        self.sampling_strategy = sampling_strategy
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Log original distribution
        original_pos = df['isContrary'].sum()
        original_neg = len(df) - original_pos
        logger.info(f"Graph data - Original Positive: {original_pos}, Negative: {original_neg}")
        
        # Create node mappings
        assumptions = df['Assumption'].unique()
        propositions = df['Proposition'].unique()
        
        # Create unified node list
        self.node_list = list(assumptions) + list(propositions)
        self.node_to_id = {node: i for i, node in enumerate(self.node_list)}
        
        # Node type mask
        self.node_types = torch.zeros(len(self.node_list), dtype=torch.long)
        self.node_types[len(assumptions):] = 1
        
        # Create edges and labels
        edges = []
        edge_labels = []
        
        for _, row in df.iterrows():
            src = self.node_to_id[row['Assumption']]
            dst = self.node_to_id[row['Proposition']]
            edges.append([src, dst])
            edge_labels.append(int(row['isContrary']))
        
        edges = np.array(edges)
        edge_labels = np.array(edge_labels)
        
        # Split edges for train/val/test
        num_edges = len(edges)
        indices = np.arange(num_edges)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=edge_labels
        )
        
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=edge_labels[train_idx]
        )
        
        # Balance training edges
        train_edges, train_labels = self._balance_edges(
            edges[train_idx], 
            edge_labels[train_idx]
        )
        
        # Combine all edges (balanced train + original val/test)
        all_edges = np.vstack([
            train_edges,
            edges[val_idx],
            edges[test_idx]
        ])
        
        all_labels = np.concatenate([
            train_labels,
            edge_labels[val_idx],
            edge_labels[test_idx]
        ])
        
        # Convert to tensors
        self.edge_index = torch.tensor(all_edges.T, dtype=torch.long)
        self.edge_labels = torch.tensor(all_labels, dtype=torch.float)
        
        # Create masks
        n_train = len(train_edges)
        n_val = len(val_idx)
        n_test = len(test_idx)
        n_total = n_train + n_val + n_test
        
        self.train_mask = torch.zeros(n_total, dtype=torch.bool)
        self.val_mask = torch.zeros(n_total, dtype=torch.bool)
        self.test_mask = torch.zeros(n_total, dtype=torch.bool)
        
        self.train_mask[:n_train] = True
        self.val_mask[n_train:n_train+n_val] = True
        self.test_mask[n_train+n_val:] = True
        
        # Log final distribution
        final_train_pos = train_labels.sum()
        final_train_neg = len(train_labels) - final_train_pos
        logger.info(f"Balanced train edges - Positive: {final_train_pos}, Negative: {final_train_neg}")
        
    def _balance_edges(self, edges: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance edges using the specified strategy.
        
        Args:
            edges: Edge array
            labels: Label array
            
        Returns:
            Balanced edges and labels
        """
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        positive_edges = edges[positive_mask]
        negative_edges = edges[negative_mask]
        
        positive_labels = labels[positive_mask]
        negative_labels = labels[negative_mask]
        
        n_positive = len(positive_edges)
        n_negative = len(negative_edges)
        
        if self.sampling_strategy == 'undersample':
            # Undersample negative edges
            n_negative_balanced = int(n_positive * self.balance_ratio)
            if n_negative_balanced < n_negative:
                idx = np.random.RandomState(self.random_state).choice(
                    n_negative, n_negative_balanced, replace=False
                )
                negative_edges = negative_edges[idx]
                negative_labels = negative_labels[idx]
                
        elif self.sampling_strategy == 'oversample':
            # Oversample positive edges
            n_positive_balanced = int(n_negative / self.balance_ratio)
            if n_positive_balanced > n_positive:
                idx = np.random.RandomState(self.random_state).choice(
                    n_positive, n_positive_balanced, replace=True
                )
                positive_edges = positive_edges[idx]
                positive_labels = positive_labels[idx]
                
        # Combine and shuffle
        balanced_edges = np.vstack([positive_edges, negative_edges])
        balanced_labels = np.concatenate([positive_labels, negative_labels])
        
        # Shuffle
        shuffle_idx = np.random.RandomState(self.random_state).permutation(len(balanced_edges))
        balanced_edges = balanced_edges[shuffle_idx]
        balanced_labels = balanced_labels[shuffle_idx]
        
        return balanced_edges, balanced_labels
    
    def get_num_nodes(self) -> int:
        """Return total number of nodes."""
        return len(self.node_list)
    
    def get_node_features(self) -> torch.Tensor:
        """Get initial node features (one-hot encoding)."""
        return torch.eye(len(self.node_list))
    
    def get_data(self) -> Dict[str, torch.Tensor]:
        """Get all graph data."""
        return {
            'edge_index': self.edge_index,
            'edge_labels': self.edge_labels,
            'node_types': self.node_types,
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
            'test_mask': self.test_mask,
            'num_nodes': self.get_num_nodes()
        }
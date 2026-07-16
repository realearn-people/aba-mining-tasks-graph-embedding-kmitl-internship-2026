"""ABA Dataset for Link Prediction."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ABADataset(Dataset):
    """Dataset class for ABA link prediction."""
    
    def __init__(
        self, 
        data_path: str,
        mode: str = 'train',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize ABA Dataset.
        
        Args:
            data_path: Path to CSV file
            mode: 'train', 'val', or 'test'
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
        """
        self.data_path = data_path
        self.mode = mode
        self.random_state = random_state
        
        # Load and process data
        self.df = pd.read_csv(data_path)
        
        # Encode assumptions and propositions
        self.assumption_encoder = LabelEncoder()
        self.proposition_encoder = LabelEncoder()
        
        self.df['assumption_id'] = self.assumption_encoder.fit_transform(self.df['Assumption'])
        self.df['proposition_id'] = self.proposition_encoder.fit_transform(self.df['Proposition'])
        
        # Split data
        train_val_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state, stratify=self.df['isContrary']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=random_state, 
            stratify=train_val_df['isContrary']
        )
        
        # Set data based on mode
        if mode == 'train':
            self.data = train_df
        elif mode == 'val':
            self.data = val_df
        elif mode == 'test':
            self.data = test_df
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'")
        
        self.data = self.data.reset_index(drop=True)
        
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
    
    def get_edge_index(self) -> torch.Tensor:
        """
        Get edge index for graph construction.
        
        Returns:
            Edge index tensor of shape [2, num_edges]
        """
        edges = []
        for _, row in self.data.iterrows():
            edges.append([row['assumption_id'], row['proposition_id']])
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def get_edge_labels(self) -> torch.Tensor:
        """Get edge labels."""
        return torch.tensor(self.data['isContrary'].values, dtype=torch.float)
    
    def get_all_texts(self) -> Tuple[List[str], List[str]]:
        """Get all unique assumption and proposition texts."""
        assumptions = self.assumption_encoder.classes_.tolist()
        propositions = self.proposition_encoder.classes_.tolist()
        return assumptions, propositions


class ABAGraphDataset:
    """Graph-based dataset for ABA link prediction."""
    
    def __init__(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize ABA Graph Dataset.
        
        Args:
            data_path: Path to CSV file
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
        """
        self.data_path = data_path
        self.random_state = random_state
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Create node mappings
        assumptions = df['Assumption'].unique()
        propositions = df['Proposition'].unique()
        
        # Create unified node list (assumptions first, then propositions)
        self.node_list = list(assumptions) + list(propositions)
        self.node_to_id = {node: i for i, node in enumerate(self.node_list)}
        
        # Node type mask (0 for assumption, 1 for proposition)
        self.node_types = torch.zeros(len(self.node_list), dtype=torch.long)
        self.node_types[len(assumptions):] = 1
        
        # Create edges
        edges = []
        edge_labels = []
        
        for _, row in df.iterrows():
            src = self.node_to_id[row['Assumption']]
            dst = self.node_to_id[row['Proposition']]
            edges.append([src, dst])
            edge_labels.append(int(row['isContrary']))
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).t()
        self.edge_labels = torch.tensor(edge_labels, dtype=torch.float)
        edge_labels = np.array(edge_labels)  # Keep numpy array for stratification
        
        # Split edges for train/val/test
        num_edges = self.edge_index.shape[1]
        indices = np.arange(num_edges)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=edge_labels
        )
        
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=edge_labels[train_idx]
        )
        
        self.train_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.val_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.test_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True
        
    def get_num_nodes(self) -> int:
        """Return total number of nodes."""
        return len(self.node_list)
    
    def get_node_features(self) -> torch.Tensor:
        """
        Get initial node features (one-hot encoding).
        
        Returns:
            Node feature tensor of shape [num_nodes, num_nodes]
        """
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
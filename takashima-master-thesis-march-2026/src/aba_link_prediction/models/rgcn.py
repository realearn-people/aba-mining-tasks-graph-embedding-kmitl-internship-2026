"""Relational Graph Convolutional Network (RGCN) for ABA link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from typing import Optional, Tuple


class RGCN(nn.Module):
    """RGCN model for link prediction in ABA graphs."""
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 128,
        num_relations: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_node_features: bool = True,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize RGCN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            hidden_dim: Hidden dimension size
            num_relations: Number of relation types
            num_layers: Number of RGCN layers
            dropout: Dropout rate
            use_node_features: Whether to use node features
            embedding_dim: Embedding dimension if not using node features
        """
        super(RGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_node_features = use_node_features
        
        # Node embeddings if not using features
        if not use_node_features:
            self.embedding_dim = embedding_dim or hidden_dim
            self.node_embedding = nn.Embedding(num_nodes, self.embedding_dim)
            input_dim = self.embedding_dim
        else:
            input_dim = num_nodes  # One-hot encoding
        
        # RGCN layers
        self.rgcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            
            self.rgcn_layers.append(
                RGCNConv(
                    in_dim, 
                    out_dim, 
                    num_relations,
                    num_bases=None,
                    bias=True
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        # Edge prediction layers
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def encode(
        self, 
        x: Optional[torch.Tensor], 
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode nodes using RGCN layers.
        
        Args:
            x: Node features (optional)
            edge_index: Edge connectivity
            edge_type: Edge types
            
        Returns:
            Node embeddings
        """
        # Get initial node representations
        if not self.use_node_features:
            h = self.node_embedding.weight
        else:
            h = x if x is not None else torch.eye(self.num_nodes, device=edge_index.device)
        
        # Apply RGCN layers
        for i in range(self.num_layers):
            h = self.rgcn_layers[i](h, edge_index, edge_type)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode edge probabilities from node embeddings.
        
        Args:
            z: Node embeddings
            edge_index: Edges to predict
            
        Returns:
            Edge predictions
        """
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        return self.edge_predictor(edge_features).squeeze(-1)
    
    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        target_edges: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_type: Edge types
            target_edges: Edges to predict (if None, use edge_index)
            
        Returns:
            Edge predictions
        """
        # Encode nodes
        z = self.encode(x, edge_index, edge_type)
        
        # Decode edges
        if target_edges is None:
            target_edges = edge_index
            
        return self.decode(z, target_edges)
    
    def predict_link(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        src_node: int,
        dst_node: int,
        edge_type: Optional[torch.Tensor] = None
    ) -> float:
        """
        Predict link probability between two nodes.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            src_node: Source node index
            dst_node: Destination node index
            edge_type: Edge types
            
        Returns:
            Link probability
        """
        with torch.no_grad():
            z = self.encode(x, edge_index, edge_type)
            test_edge = torch.tensor([[src_node], [dst_node]], device=edge_index.device)
            score = self.decode(z, test_edge)
            return torch.sigmoid(score).item()


class RGCNWithAttention(RGCN):
    """RGCN with attention mechanism for enhanced link prediction."""
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 128,
        num_relations: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5,
        use_node_features: bool = True,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize RGCN with attention.
        
        Args:
            num_nodes: Number of nodes
            hidden_dim: Hidden dimension
            num_relations: Number of relations
            num_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_node_features: Whether to use node features
            embedding_dim: Embedding dimension
        """
        super().__init__(
            num_nodes, hidden_dim, num_relations,
            num_layers, dropout, use_node_features, embedding_dim
        )
        
        self.num_heads = num_heads
        
        # Multi-head attention for edge features
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Update edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode with attention mechanism.
        
        Args:
            z: Node embeddings
            edge_index: Edges to predict
            
        Returns:
            Edge predictions
        """
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        
        # Apply self-attention
        edge_features = edge_features.unsqueeze(0)  # Add batch dimension
        attended_features, _ = self.attention(
            edge_features, edge_features, edge_features
        )
        attended_features = attended_features.squeeze(0)  # Remove batch dimension
        
        return self.edge_predictor(attended_features).squeeze(-1)
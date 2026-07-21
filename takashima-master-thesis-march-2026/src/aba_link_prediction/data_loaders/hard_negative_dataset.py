"""Dataset with hard negative sampling as negative examples for robust training."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, List, Set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HardNegativeDataset(Dataset):
    """Dataset with hard negative sampling for robust ABA link prediction."""
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        hard_negative_ratio: float = 0.4,  # 40% hard negatives
        structural_negative_ratio: float = 0.3,  # 30% structural negatives  
        random_negative_ratio: float = 0.3,  # 30% random negatives
        negative_to_positive_ratio: float = 1.0,  # 1:1 ratio
        similarity_threshold: float = 0.5,  # Threshold for hard negatives
        use_tfidf: bool = True  # Use TF-IDF for speed
    ):
        """
        Initialize dataset with hard negative sampling.
        
        Args:
            data_path: Path to CSV file
            mode: 'train', 'val', or 'test'
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
            hard_negative_ratio: Ratio of hard negatives in negative samples
            structural_negative_ratio: Ratio of structural negatives
            random_negative_ratio: Ratio of random negatives
            negative_to_positive_ratio: Ratio of negative to positive samples
            similarity_threshold: Threshold for semantic similarity
            use_tfidf: Whether to use TF-IDF (faster) or sentence embeddings
        """
        self.data_path = data_path
        self.mode = mode
        self.random_state = random_state
        self.hard_negative_ratio = hard_negative_ratio
        self.structural_negative_ratio = structural_negative_ratio
        self.random_negative_ratio = random_negative_ratio
        self.negative_to_positive_ratio = negative_to_positive_ratio
        self.similarity_threshold = similarity_threshold
        self.use_tfidf = use_tfidf
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Load original data
        self.df = pd.read_csv(data_path)
        
        # Log original distribution
        original_pos = self.df['isContrary'].sum()
        original_neg = len(self.df) - original_pos
        logger.info(f"Original distribution - Positive: {original_pos}, Negative: {original_neg}")
        
        # Generate hard negatives
        if mode in ['train', 'val']:  # Add hard negatives to train and val sets
            logger.info(f"Generating hard negatives for {mode} set...")
            self.df = self._add_hard_negatives(self.df)
        
        # Encode assumptions and propositions
        self.assumption_encoder = LabelEncoder()
        self.proposition_encoder = LabelEncoder()
        
        self.df['assumption_id'] = self.assumption_encoder.fit_transform(self.df['Assumption'])
        self.df['proposition_id'] = self.proposition_encoder.fit_transform(self.df['Proposition'])
        
        # Split data AFTER adding hard negatives
        train_val_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state,
            stratify=self.df['isContrary']
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
            raise ValueError(f"Invalid mode: {mode}")
        
        self.data = self.data.reset_index(drop=True)
        
        # Log final distribution
        final_pos = self.data['isContrary'].sum()
        final_neg = len(self.data) - final_pos
        logger.info(f"Final {mode} distribution - Positive: {final_pos}, Negative: {final_neg}")
        logger.info(f"Final {mode} ratio - Positive: {final_pos/len(self.data):.2%}")
    
    def _add_hard_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hard negative samples to the dataset."""
        
        # Get positive and original negative samples
        positive_df = df[df['isContrary'] == True]
        original_negative_df = df[df['isContrary'] == False]
        
        n_positives = len(positive_df)
        n_target_negatives = int(n_positives * self.negative_to_positive_ratio)
        
        # Calculate number of each type of negative
        n_hard_negatives = int(n_target_negatives * self.hard_negative_ratio)
        n_structural_negatives = int(n_target_negatives * self.structural_negative_ratio)
        n_random_negatives = n_target_negatives - n_hard_negatives - n_structural_negatives
        
        logger.info(f"Generating {n_hard_negatives} hard negatives...")
        logger.info(f"Generating {n_structural_negatives} structural negatives...")
        logger.info(f"Generating {n_random_negatives} random negatives...")
        
        # Get unique assumptions and propositions
        unique_assumptions = df['Assumption'].unique()
        unique_propositions = df['Proposition'].unique()
        
        # Create existing pairs set for checking
        existing_pairs = set()
        for _, row in df.iterrows():
            existing_pairs.add((row['Assumption'], row['Proposition']))
        
        # 1. Generate hard negatives (semantically similar but not contrary)
        hard_negatives = self._generate_hard_negatives(
            unique_assumptions, 
            unique_propositions,
            existing_pairs,
            n_hard_negatives
        )
        
        # 2. Generate structural negatives (connected in inference graph)
        structural_negatives = self._generate_structural_negatives(
            df,
            existing_pairs,
            n_structural_negatives
        )
        
        # 3. Generate random negatives
        random_negatives = self._generate_random_negatives(
            unique_assumptions,
            unique_propositions,
            existing_pairs,
            n_random_negatives
        )
        
        # Combine all negatives
        all_new_negatives = hard_negatives + structural_negatives + random_negatives
        
        # Create dataframe for new negatives
        new_negative_df = pd.DataFrame(all_new_negatives)
        
        # Combine with positive samples
        balanced_df = pd.concat([positive_df, new_negative_df], ignore_index=True)
        
        # Add some original negatives if we have room
        remaining_slots = max(0, n_target_negatives - len(all_new_negatives))
        if remaining_slots > 0 and len(original_negative_df) > 0:
            sampled_original = original_negative_df.sample(
                n=min(remaining_slots, len(original_negative_df)),
                random_state=self.random_state
            )
            balanced_df = pd.concat([balanced_df, sampled_original], ignore_index=True)
        
        logger.info(f"Added {len(all_new_negatives)} synthetic negative samples")
        
        return balanced_df
    
    def _generate_hard_negatives(
        self,
        assumptions: np.ndarray,
        propositions: np.ndarray,
        existing_pairs: Set[Tuple[str, str]],
        n_samples: int
    ) -> List[Dict]:
        """Generate hard negative samples based on semantic similarity."""
        
        hard_negatives = []
        
        if self.use_tfidf:
            # Use TF-IDF for faster computation
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            
            # Combine all texts
            all_texts = list(assumptions) + list(propositions)
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Split back into assumptions and propositions
            n_assumptions = len(assumptions)
            assumption_vectors = tfidf_matrix[:n_assumptions]
            proposition_vectors = tfidf_matrix[n_assumptions:]
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(assumption_vectors, proposition_vectors)
            
            # Find hard negatives (high similarity but no edge)
            for i, assumption in enumerate(assumptions):
                for j, proposition in enumerate(propositions):
                    if (assumption, proposition) not in existing_pairs:
                        similarity = similarity_matrix[i, j]
                        
                        # Hard negative: high similarity but no attack edge
                        if similarity > self.similarity_threshold:
                            hard_negatives.append({
                                'Assumption': assumption,
                                'Proposition': proposition,
                                'isContrary': False,
                                'negative_type': 'hard',
                                'similarity': similarity
                            })
            
            # Sort by similarity and take top n_samples
            hard_negatives.sort(key=lambda x: x['similarity'], reverse=True)
            hard_negatives = hard_negatives[:n_samples]
        
        else:
            # Use sentence embeddings (more accurate but slower)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            assumption_embeddings = model.encode(assumptions.tolist())
            proposition_embeddings = model.encode(propositions.tolist())
            
            similarity_matrix = cosine_similarity(assumption_embeddings, proposition_embeddings)
            
            for i, assumption in enumerate(assumptions):
                for j, proposition in enumerate(propositions):
                    if (assumption, proposition) not in existing_pairs:
                        similarity = similarity_matrix[i, j]
                        
                        if similarity > self.similarity_threshold:
                            hard_negatives.append({
                                'Assumption': assumption,
                                'Proposition': proposition,
                                'isContrary': False,
                                'negative_type': 'hard',
                                'similarity': similarity
                            })
            
            hard_negatives.sort(key=lambda x: x['similarity'], reverse=True)
            hard_negatives = hard_negatives[:n_samples]
        
        # Remove similarity scores before returning
        for neg in hard_negatives:
            del neg['similarity']
            del neg['negative_type']
        
        return hard_negatives
    
    def _generate_structural_negatives(
        self,
        df: pd.DataFrame,
        existing_pairs: Set[Tuple[str, str]],
        n_samples: int
    ) -> List[Dict]:
        """Generate structural negatives (connected through inference)."""
        
        structural_negatives = []
        
        # Find pairs that are connected through a common node
        # but don't have a direct attack edge
        assumptions = df['Assumption'].unique()
        propositions = df['Proposition'].unique()
        
        # Build a simple graph structure from existing data
        assumption_to_props = {}
        prop_to_assumptions = {}
        
        for _, row in df.iterrows():
            assumption = row['Assumption']
            proposition = row['Proposition']
            
            if assumption not in assumption_to_props:
                assumption_to_props[assumption] = set()
            assumption_to_props[assumption].add(proposition)
            
            if proposition not in prop_to_assumptions:
                prop_to_assumptions[proposition] = set()
            prop_to_assumptions[proposition].add(assumption)
        
        # Find structural negatives
        for assumption in assumptions:
            # Get propositions connected to this assumption
            connected_props = assumption_to_props.get(assumption, set())
            
            for prop in connected_props:
                # Get other assumptions connected to this proposition
                other_assumptions = prop_to_assumptions.get(prop, set())
                
                for other_assumption in other_assumptions:
                    if other_assumption != assumption:
                        # Check if this pair doesn't exist
                        for other_prop in assumption_to_props.get(other_assumption, set()):
                            if (assumption, other_prop) not in existing_pairs:
                                structural_negatives.append({
                                    'Assumption': assumption,
                                    'Proposition': other_prop,
                                    'isContrary': False
                                })
                                
                                if len(structural_negatives) >= n_samples:
                                    return structural_negatives[:n_samples]
        
        return structural_negatives[:n_samples]
    
    def _generate_random_negatives(
        self,
        assumptions: np.ndarray,
        propositions: np.ndarray,
        existing_pairs: Set[Tuple[str, str]],
        n_samples: int
    ) -> List[Dict]:
        """Generate random negative samples."""
        
        random_negatives = []
        max_attempts = n_samples * 10
        attempts = 0
        
        while len(random_negatives) < n_samples and attempts < max_attempts:
            assumption = random.choice(assumptions)
            proposition = random.choice(propositions)
            
            if (assumption, proposition) not in existing_pairs:
                random_negatives.append({
                    'Assumption': assumption,
                    'Proposition': proposition,
                    'isContrary': False
                })
                existing_pairs.add((assumption, proposition))  # Avoid duplicates
            
            attempts += 1
        
        return random_negatives[:n_samples]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        return {
            'assumption_id': torch.tensor(row['assumption_id'], dtype=torch.long),
            'proposition_id': torch.tensor(row['proposition_id'], dtype=torch.long),
            'assumption_text': row['Assumption'],
            'proposition_text': row['Proposition'],
            'label': torch.tensor(int(row['isContrary']), dtype=torch.float)
        }
    
    def get_num_nodes(self) -> Tuple[int, int]:
        return (
            len(self.assumption_encoder.classes_),
            len(self.proposition_encoder.classes_)
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for weighted loss functions."""
        n_samples = len(self.data)
        n_positive = self.data['isContrary'].sum()
        n_negative = n_samples - n_positive
        
        # Inverse frequency weighting
        weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
        weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0
        
        return torch.tensor([weight_negative, weight_positive], dtype=torch.float)
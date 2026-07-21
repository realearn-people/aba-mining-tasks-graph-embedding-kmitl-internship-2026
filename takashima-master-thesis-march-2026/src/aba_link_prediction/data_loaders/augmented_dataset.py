"""Augmented dataset with hard negative sampling for ABA link prediction."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
import random

logger = logging.getLogger(__name__)


class HardNegativeSampler:
    """Generate hard negative samples based on semantic similarity."""
    
    def __init__(
        self,
        similarity_method: str = 'sentence_transformer',
        similarity_threshold: float = 0.7,
        dissimilarity_threshold: float = 0.3,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize hard negative sampler.
        
        Args:
            similarity_method: 'tfidf' or 'sentence_transformer'
            similarity_threshold: Threshold for semantic similarity (for hard negatives)
            dissimilarity_threshold: Threshold for semantic dissimilarity (for potential attacks)
            model_name: Name of sentence transformer model
        """
        self.similarity_method = similarity_method
        self.similarity_threshold = similarity_threshold
        self.dissimilarity_threshold = dissimilarity_threshold
        
        if similarity_method == 'sentence_transformer':
            self.model = SentenceTransformer(model_name)
        elif similarity_method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
    
    def compute_similarities(self, texts: List[str]) -> np.ndarray:
        """
        Compute pairwise similarities between texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Similarity matrix
        """
        if self.similarity_method == 'sentence_transformer':
            embeddings = self.model.encode(texts)
            similarities = cosine_similarity(embeddings)
        else:  # tfidf
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarities = cosine_similarity(tfidf_matrix)
        
        return similarities
    
    def find_hard_negatives(
        self,
        assumptions: List[str],
        propositions: List[str],
        existing_pairs: set
    ) -> List[Tuple[int, int]]:
        """
        Find hard negative pairs (semantically opposite but no attack edge).
        
        Args:
            assumptions: List of assumption texts
            propositions: List of proposition texts
            existing_pairs: Set of existing (assumption_idx, proposition_idx) pairs
            
        Returns:
            List of hard negative pairs
        """
        logger.info("Computing semantic similarities for hard negative mining...")
        
        # Combine all texts
        all_texts = list(set(assumptions + propositions))
        text_to_idx = {text: i for i, text in enumerate(all_texts)}
        
        # Compute similarities
        similarities = self.compute_similarities(all_texts)
        
        # Find assumption and proposition indices in combined list
        assumption_indices = [text_to_idx[a] for a in assumptions]
        proposition_indices = [text_to_idx[p] for p in propositions]
        
        hard_negatives = []
        
        for i, assumption in enumerate(assumptions):
            a_idx = text_to_idx[assumption]
            
            for j, proposition in enumerate(propositions):
                p_idx = text_to_idx[proposition]
                
                # Skip if pair already exists
                if (i, j) in existing_pairs:
                    continue
                
                # Check similarity (low similarity might indicate opposition)
                similarity = similarities[a_idx, p_idx]
                
                # Hard negative: low similarity (potentially opposing) but no edge
                if similarity < self.dissimilarity_threshold:
                    hard_negatives.append((i, j))
        
        logger.info(f"Found {len(hard_negatives)} hard negative candidates")
        return hard_negatives
    
    def find_semantically_opposite_pairs(
        self,
        df: pd.DataFrame,
        max_pairs: int = 1000
    ) -> pd.DataFrame:
        """
        Find semantically opposite pairs as synthetic positive examples.
        
        Args:
            df: Original dataframe with columns [Assumption, Proposition, isContrary]
            max_pairs: Maximum number of synthetic pairs to generate
            
        Returns:
            Dataframe with synthetic positive examples
        """
        # Get unique assumptions and propositions
        unique_assumptions = df['Assumption'].unique()
        unique_propositions = df['Proposition'].unique()
        
        # Create existing pairs set
        existing_pairs = set()
        for _, row in df.iterrows():
            a_idx = np.where(unique_assumptions == row['Assumption'])[0][0]
            p_idx = np.where(unique_propositions == row['Proposition'])[0][0]
            existing_pairs.add((a_idx, p_idx))
        
        # Find hard negatives
        hard_negatives = self.find_hard_negatives(
            unique_assumptions.tolist(),
            unique_propositions.tolist(),
            existing_pairs
        )
        
        # Sample and create synthetic positive examples
        if len(hard_negatives) > max_pairs:
            hard_negatives = random.sample(hard_negatives, max_pairs)
        
        synthetic_data = []
        for a_idx, p_idx in hard_negatives:
            synthetic_data.append({
                'Assumption': unique_assumptions[a_idx],
                'Proposition': unique_propositions[p_idx],
                'isContrary': True  # Mark as positive (attack relationship)
            })
        
        synthetic_df = pd.DataFrame(synthetic_data)
        logger.info(f"Generated {len(synthetic_df)} synthetic positive examples")
        
        return synthetic_df


class AugmentedABADataset(Dataset):
    """Dataset with data augmentation through hard negative sampling."""
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        balance_ratio: float = 1.0,
        use_augmentation: bool = True,
        augmentation_ratio: float = 0.5,  # Ratio of synthetic to real positive examples
        similarity_method: str = 'tfidf'  # Use TF-IDF for faster computation
    ):
        """
        Initialize augmented dataset.
        
        Args:
            data_path: Path to CSV file
            mode: 'train', 'val', or 'test'
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
            balance_ratio: Ratio of negative to positive samples
            use_augmentation: Whether to use data augmentation
            augmentation_ratio: Ratio of synthetic to real positive examples
            similarity_method: Method for computing similarities
        """
        self.data_path = data_path
        self.mode = mode
        self.random_state = random_state
        self.balance_ratio = balance_ratio
        self.use_augmentation = use_augmentation
        self.augmentation_ratio = augmentation_ratio
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Load original data
        self.df = pd.read_csv(data_path)
        
        # Log original distribution
        original_pos = self.df['isContrary'].sum()
        original_neg = len(self.df) - original_pos
        logger.info(f"Original distribution - Positive: {original_pos}, Negative: {original_neg}")
        
        # Generate synthetic positive examples if augmentation is enabled
        if use_augmentation and mode == 'train':
            sampler = HardNegativeSampler(
                similarity_method=similarity_method,
                dissimilarity_threshold=0.3
            )
            
            # Generate synthetic examples
            num_synthetic = int(original_pos * augmentation_ratio)
            synthetic_df = sampler.find_semantically_opposite_pairs(
                self.df,
                max_pairs=num_synthetic
            )
            
            # Combine original and synthetic data
            self.df = pd.concat([self.df, synthetic_df], ignore_index=True)
            
            augmented_pos = self.df['isContrary'].sum()
            augmented_neg = len(self.df) - augmented_pos
            logger.info(f"After augmentation - Positive: {augmented_pos}, Negative: {augmented_neg}")
        
        # Encode assumptions and propositions
        self.assumption_encoder = LabelEncoder()
        self.proposition_encoder = LabelEncoder()
        
        self.df['assumption_id'] = self.assumption_encoder.fit_transform(self.df['Assumption'])
        self.df['proposition_id'] = self.proposition_encoder.fit_transform(self.df['Proposition'])
        
        # Split data
        train_val_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state, 
            stratify=self.df['isContrary']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=random_state,
            stratify=train_val_df['isContrary']
        )
        
        # Balance training data
        if mode == 'train':
            self.data = self._balance_dataset(train_df)
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
    
    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance dataset using undersampling."""
        positive_samples = df[df['isContrary'] == True]
        negative_samples = df[df['isContrary'] == False]
        
        n_positive = len(positive_samples)
        n_negative_balanced = int(n_positive * self.balance_ratio)
        
        if n_negative_balanced < len(negative_samples):
            negative_samples = negative_samples.sample(
                n=n_negative_balanced,
                random_state=self.random_state
            )
        
        balanced_df = pd.concat([positive_samples, negative_samples])
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return balanced_df
    
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


class ContrastiveAugmentationDataset(Dataset):
    """Dataset with contrastive learning-based augmentation."""
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        use_negation: bool = True,
        use_paraphrase: bool = False
    ):
        """
        Initialize dataset with contrastive augmentation.
        
        Args:
            data_path: Path to CSV file
            mode: 'train', 'val', or 'test'
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
            use_negation: Whether to use negation-based augmentation
            use_paraphrase: Whether to use paraphrase-based augmentation
        """
        self.data_path = data_path
        self.mode = mode
        self.random_state = random_state
        self.use_negation = use_negation
        self.use_paraphrase = use_paraphrase
        
        # Load and process data
        self.df = pd.read_csv(data_path)
        
        # Apply augmentation to positive examples
        if mode == 'train' and (use_negation or use_paraphrase):
            self.df = self._augment_data(self.df)
        
        # Rest of initialization similar to base dataset
        self._prepare_data()
    
    def _augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply contrastive augmentation techniques."""
        augmented_rows = []
        
        # Common negation words and patterns
        negation_words = ['not', 'no', 'never', 'neither', 'nor', 'none', 'nothing']
        antonym_pairs = {
            'clean': 'dirty', 'good': 'bad', 'quiet': 'noisy', 'comfortable': 'uncomfortable',
            'spacious': 'cramped', 'modern': 'outdated', 'friendly': 'unfriendly',
            'helpful': 'unhelpful', 'convenient': 'inconvenient', 'pleasant': 'unpleasant'
        }
        
        for _, row in df[df['isContrary'] == True].iterrows():
            if self.use_negation:
                # Create negation-based augmentation
                assumption = row['Assumption']
                proposition = row['Proposition']
                
                # Simple negation augmentation
                for word, antonym in antonym_pairs.items():
                    if word in assumption.lower():
                        aug_assumption = assumption.lower().replace(word, antonym)
                        augmented_rows.append({
                            'Assumption': aug_assumption,
                            'Proposition': proposition,
                            'isContrary': True
                        })
                        break
                    elif word in proposition.lower():
                        aug_proposition = proposition.lower().replace(word, antonym)
                        augmented_rows.append({
                            'Assumption': assumption,
                            'Proposition': aug_proposition,
                            'isContrary': True
                        })
                        break
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            df = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(f"Added {len(augmented_rows)} augmented examples")
        
        return df
    
    def _prepare_data(self):
        """Prepare data splits and encoding."""
        # Encode assumptions and propositions
        self.assumption_encoder = LabelEncoder()
        self.proposition_encoder = LabelEncoder()
        
        self.df['assumption_id'] = self.assumption_encoder.fit_transform(self.df['Assumption'])
        self.df['proposition_id'] = self.proposition_encoder.fit_transform(self.df['Proposition'])
        
        # Split and balance data (similar to base implementation)
        # ... (implementation details)
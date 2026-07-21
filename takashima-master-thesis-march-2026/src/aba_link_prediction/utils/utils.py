"""Utility functions for ABA link prediction."""

import torch
import numpy as np
import random
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(
    log_file: Optional[str] = None,
    level: str = 'INFO',
    format_str: Optional[str] = None
):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        format_str: Log format string
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=format_str,
        handlers=handlers
    )


def create_directories(paths: List[str]):
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_model(
    model: torch.nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save model
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional metrics dictionary
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    logging.info(f"Model saved to {save_path}")


def load_model(
    model: torch.nn.Module,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        load_path: Path to load model from
        optimizer: Optional optimizer to load state
        device: Device to load model to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Model loaded from {load_path}")
    
    return checkpoint


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ['Non-Contrary', 'Contrary'],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def save_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = 'json'
):
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
        format: Format to save ('json' or 'yaml')
    """
    if format == 'json':
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'yaml':
        with open(save_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logging.info(f"Results saved to {save_path}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        Device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: Model to count parameters
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
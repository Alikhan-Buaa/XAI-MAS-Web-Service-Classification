"""
Common utility functions used across the project
Centralized helper functions for data loading, saving, logging, etc.
"""

import json
import yaml
import pickle
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import os
import sys

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config import (
        LOGGING_CONFIG, DATA_CONFIG, RESULTS_CONFIG,
        MODEL_NAME_MAPPING, FEATURE_NAME_MAPPING  # ← Import mappings from config
    )
except ImportError:
    # Fallback if config import fails
    LOGGING_CONFIG = {
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'handlers': {'console': True, 'file': True}
    }
    MODEL_NAME_MAPPING = {}
    FEATURE_NAME_MAPPING = {}


def setup_logging(log_file: Optional[Path] = None, 
                 level: str = "INFO", 
                 format_str: Optional[str] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
        format_str: Log format string
    """
    if format_str is None:
        format_str = LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    
    logging_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if LOGGING_CONFIG.get('handlers', {}).get('console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if LOGGING_CONFIG.get('handlers', {}).get('file', True) and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_data(file_path: Union[str, Path], 
              file_type: Optional[str] = None) -> Any:
    """
    Load data from various file formats
    
    Args:
        file_path: Path to the data file
        file_type: Type of file ('csv', 'json', 'yaml', 'pickle', 'joblib')
    
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type in ['json', 'jsonl']:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_type == 'jsonl':
                    return [json.loads(line) for line in f]
                return json.load(f)
        elif file_type in ['yaml', 'yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif file_type in ['pickle', 'pkl']:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_type == 'joblib':
            return joblib.load(file_path)
        elif file_type == 'npy':
            return np.load(file_path)
        elif file_type == 'npz':
            return np.load(file_path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        raise


def save_data(data: Any, 
              file_path: Union[str, Path], 
              file_type: Optional[str] = None,
              **kwargs) -> None:
    """
    Save data to various file formats
    
    Args:
        data: Data to save
        file_path: Path to save the file
        file_type: Type of file ('csv', 'json', 'yaml', 'pickle', 'joblib')
        **kwargs: Additional arguments for specific save functions
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    try:
        if file_type == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=kwargs.get('index', False))
            else:
                pd.DataFrame(data).to_csv(file_path, index=kwargs.get('index', False))
        elif file_type == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=kwargs.get('indent', 2), 
                         ensure_ascii=kwargs.get('ensure_ascii', False))
        elif file_type in ['yaml', 'yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, 
                         allow_unicode=kwargs.get('allow_unicode', True))
        elif file_type in ['pickle', 'pkl']:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=kwargs.get('protocol', pickle.HIGHEST_PROTOCOL))
        elif file_type == 'joblib':
            joblib.dump(data, file_path, compress=kwargs.get('compress', 3))
        elif file_type == 'npy':
            np.save(file_path, data)
        elif file_type == 'npz':
            if isinstance(data, dict):
                np.savez_compressed(file_path, **data)
            else:
                np.savez_compressed(file_path, data=data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logging.error(f"Error saving file {file_path}: {str(e)}")
        raise


def get_timestamp() -> str:
    """Get current timestamp as formatted string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Ensure reproducibility by setting random seeds
    
    Args:
        seed: Random seed value
    """
    import random
    import os
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set TensorFlow/Keras seeds if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logging.info(f"Reproducibility ensured with seed: {seed}")


def format_metrics(metrics: Dict[str, float], 
                  decimal_places: int = 4) -> Dict[str, str]:
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metric values
        decimal_places: Number of decimal places
    
    Returns:
        Dictionary of formatted metric strings
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[key] = f"{value:.{decimal_places}f}"
        else:
            formatted[key] = str(value)
    return formatted


def print_section_header(title: str, char: str = "=", width: int = 80) -> None:
    """
    Print a formatted section header
    
    Args:
        title: Section title
        char: Character to use for the line
        width: Total width of the header
    """
    title_len = len(title)
    if title_len >= width - 4:
        print(char * width)
        print(f"  {title}")
        print(char * width)
    else:
        padding = (width - title_len - 2) // 2
        line = char * padding + f" {title} " + char * padding
        if len(line) < width:
            line += char
        print(line)


class FileNamingStandard:
    """
    Standardized file naming conventions across all model types
    Uses mappings from config.py for consistency
    Pattern: {ModelName}_{FeatureType}_top_{N}_categories_{FileType}.{Extension}
    """
    
    @staticmethod
    def standardize_model_name(model_name):
        """
        Convert model name to standard format using config mappings
        Falls back to generic cleaning if not in mapping
        """
        # Try to get from config mapping first
        if model_name in MODEL_NAME_MAPPING:
            return MODEL_NAME_MAPPING[model_name]
        
        # Fallback: Generic cleaning
        clean_name = model_name.replace(' ', '_').replace('-', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        return clean_name
    
    @staticmethod
    def standardize_feature_name(feature_type):
        """
        Convert feature type to standard format using config mappings
        Falls back to uppercase if not in mapping
        """
        # Try to get from config mapping first
        if feature_type in FEATURE_NAME_MAPPING:
            return FEATURE_NAME_MAPPING[feature_type]
        
        # Fallback: Convert to uppercase
        return feature_type.upper()
    
    @staticmethod
    def generate_confusion_matrix_filename(model_name, feature_type, n_categories, file_format='png'):
        """
        Generate standardized confusion matrix filename
        Pattern: {ModelName}_{FeatureType}_top_{N}_categories_confusion_matrix.{Extension}
        """
        clean_model = FileNamingStandard.standardize_model_name(model_name)
        clean_feature = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{clean_model}_{clean_feature}_top_{n_categories}_categories_confusion_matrix.{file_format}"
    
    @staticmethod
    def generate_classification_report_filename(model_name, feature_type, n_categories, file_format='csv'):
        """
        Generate standardized classification report filename
        Pattern: {ModelName}_{FeatureType}_top_{N}_categories_classification_report.{Extension}
        """
        clean_model = FileNamingStandard.standardize_model_name(model_name)
        clean_feature = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{clean_model}_{clean_feature}_top_{n_categories}_categories_classification_report.{file_format}"
    
    @staticmethod
    def generate_training_history_filename(model_name, n_categories, file_format='png'):
        """
        Generate standardized training history filename
        Pattern: {ModelName}_training_history_top_{N}_categories.{Extension}
        """
        clean_model = FileNamingStandard.standardize_model_name(model_name)
        return f"{clean_model}_training_history_top_{n_categories}_categories.{file_format}"
    
    @staticmethod
    def generate_model_filename(model_name, feature_type, n_categories, file_format='pth'):
        """
        Generate standardized model filename
        Pattern: {ModelName}_{FeatureType}_top_{N}_categories_model.{Extension}
        """
        clean_model = FileNamingStandard.standardize_model_name(model_name)
        clean_feature = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{clean_model}_{clean_feature}_top_{n_categories}_categories_model.{file_format}"
    
    @staticmethod
    def generate_metrics_filename(model_name, feature_type, n_categories, file_format='json'):
        """
        Generate standardized metrics filename
        Pattern: {ModelName}_{FeatureType}_top_{N}_categories_metrics.{Extension}
        """
        clean_model = FileNamingStandard.standardize_model_name(model_name)
        clean_feature = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{clean_model}_{clean_feature}_top_{n_categories}_categories_metrics.{file_format}"
    
    @staticmethod
    def generate_config_filename(model_name, feature_type, n_categories, file_format='json'):
        """
        Generate standardized config filename
        Pattern: {ModelName}_{FeatureType}_top_{N}_categories_config.{Extension}
        """
        clean_model = FileNamingStandard.standardize_model_name(model_name)
        clean_feature = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{clean_model}_{clean_feature}_top_{n_categories}_categories_config.{file_format}"


# Export commonly used functions
__all__ = [
    'setup_logging', 'load_data', 'save_data', 
    'get_timestamp', 'ensure_reproducibility',
    'format_metrics', 'print_section_header', 'FileNamingStandard'
]
"""
utils.py  —  src/utils/utils.py
================================
General project utilities (file I/O, logging, reproducibility, naming).

Explainability logic lives in the same package:
    src/utils/explainability_utils.py

All explainability symbols are re-exported here so every existing import
continues to work without any change:
    from src.utils.utils import STOPWORDS, compute_metrics, run_beeswarm ...
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
from collections import defaultdict
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))
try:
    from config import (
        LOGGING_CONFIG, DATA_CONFIG, RESULTS_CONFIG,
        DATA_PATH, PREPROCESSING_CONFIG,
        MODEL_NAME_MAPPING, FEATURE_NAME_MAPPING,
    )
except ImportError:
    LOGGING_CONFIG       = {'format': '%(asctime)s - %(levelname)s - %(message)s',
                            'handlers': {'console': True, 'file': True}}
    MODEL_NAME_MAPPING   = {}
    FEATURE_NAME_MAPPING = {}
    DATA_PATH            = Path('data')
    PREPROCESSING_CONFIG = {}


# ==============================================================================
#  SECTION 1 — GENERAL UTILITIES
# ==============================================================================

def setup_logging(log_file: Optional[Path] = None,
                  level: str = "INFO",
                  format_str: Optional[str] = None) -> None:
    """Setup logging configuration."""
    if format_str is None:
        format_str = LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    logging_level = getattr(logging, level.upper())
    formatter     = logging.Formatter(format_str)
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    logger.handlers.clear()
    if LOGGING_CONFIG.get('handlers', {}).get('console', True):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if LOGGING_CONFIG.get('handlers', {}).get('file', True) and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def load_data(file_path: Union[str, Path],
              file_type: Optional[str] = None) -> Any:
    """Load data from csv / json / yaml / pickle / joblib / npy / npz."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type in ['json', 'jsonl']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [json.loads(l) for l in f] if file_type == 'jsonl' else json.load(f)
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
        logging.error(f"Error loading file {file_path}: {e}")
        raise


def save_data(data: Any,
              file_path: Union[str, Path],
              file_type: Optional[str] = None,
              **kwargs) -> None:
    """Save data to csv / json / yaml / pickle / joblib / npy / npz."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    try:
        if file_type == 'csv':
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            df.to_csv(file_path, index=kwargs.get('index', False))
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
        logging.error(f"Error saving file {file_path}: {e}")
        raise


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_reproducibility(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
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
    return {k: f"{v:.{decimal_places}f}" if isinstance(v, (int, float)) else str(v)
            for k, v in metrics.items()}


def print_section_header(title: str, char: str = "=", width: int = 80) -> None:
    if len(title) >= width - 4:
        print(char * width)
        print(f"  {title}")
        print(char * width)
    else:
        padding = (width - len(title) - 2) // 2
        line = char * padding + f" {title} " + char * padding
        if len(line) < width:
            line += char
        print(line)


class FileNamingStandard:
    """Standardized file naming conventions across all model types."""

    @staticmethod
    def standardize_model_name(model_name: str) -> str:
        if model_name in MODEL_NAME_MAPPING:
            return MODEL_NAME_MAPPING[model_name]
        clean = model_name.replace(' ', '_').replace('-', '_')
        return ''.join(c for c in clean if c.isalnum() or c == '_')

    @staticmethod
    def standardize_feature_name(feature_type: str) -> str:
        if feature_type in FEATURE_NAME_MAPPING:
            return FEATURE_NAME_MAPPING[feature_type]
        return feature_type.upper()

    @staticmethod
    def generate_confusion_matrix_filename(model_name, feature_type, n_categories, file_format='png'):
        m = FileNamingStandard.standardize_model_name(model_name)
        f = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{m}_{f}_top_{n_categories}_categories_confusion_matrix.{file_format}"

    @staticmethod
    def generate_classification_report_filename(model_name, feature_type, n_categories, file_format='csv'):
        m = FileNamingStandard.standardize_model_name(model_name)
        f = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{m}_{f}_top_{n_categories}_categories_classification_report.{file_format}"

    @staticmethod
    def generate_training_history_filename(model_name, n_categories, file_format='png'):
        m = FileNamingStandard.standardize_model_name(model_name)
        return f"{m}_training_history_top_{n_categories}_categories.{file_format}"

    @staticmethod
    def generate_model_filename(model_name, feature_type, n_categories, file_format='pth'):
        m = FileNamingStandard.standardize_model_name(model_name)
        f = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{m}_{f}_top_{n_categories}_categories_model.{file_format}"

    @staticmethod
    def generate_metrics_filename(model_name, feature_type, n_categories, file_format='json'):
        m = FileNamingStandard.standardize_model_name(model_name)
        f = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{m}_{f}_top_{n_categories}_categories_metrics.{file_format}"

    @staticmethod
    def generate_config_filename(model_name, feature_type, n_categories, file_format='json'):
        m = FileNamingStandard.standardize_model_name(model_name)
        f = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{m}_{f}_top_{n_categories}_categories_config.{file_format}"


# ==============================================================================
#  SECTION 2 — RE-EXPORTS FROM explainability_utils.py
#
#  All explainability symbols (STOPWORDS, metrics, plotting, SHAP/LIME helpers,
#  run_beeswarm, run_waterfall) now live in:
#      src/utils/explainability_utils.py   ← same package, next to this file
#
#  These re-exports keep EVERY existing caller working with zero changes.
#  Any file that currently does:
#      from src.utils.utils import STOPWORDS, compute_metrics, run_beeswarm
#  continues to work exactly as before.
# ==============================================================================

from src.utils.explainability_utils import (  # noqa: E402 F401
    STOPWORDS,
    TARGET_CATEGORIES,
    FALLBACK_LABELS,
    load_class_labels,
    top15_tokens,
    plot_bar,
    compute_metrics,
    build_shap_background,
    run_global_shap,
    run_global_lime,
    run_beeswarm,
    run_waterfall,
)


# ==============================================================================
#  EXPORTS
# ==============================================================================

__all__ = [
    # Section 1 — General utilities
    'setup_logging', 'load_data', 'save_data', 'get_timestamp',
    'ensure_reproducibility', 'format_metrics', 'print_section_header',
    'FileNamingStandard',
    # Section 2 — Explainability re-exports (from explainability_utils)
    'STOPWORDS', 'TARGET_CATEGORIES', 'FALLBACK_LABELS',
    'load_class_labels', 'top15_tokens', 'plot_bar', 'compute_metrics',
    'build_shap_background', 'run_global_shap', 'run_global_lime',
    'run_beeswarm', 'run_waterfall',
]

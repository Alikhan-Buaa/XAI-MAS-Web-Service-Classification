"""
Common utility functions used across the project
Centralized helper functions for data loading, saving, logging, etc.

Sections
--------
1. General utilities     — file I/O, logging, reproducibility, naming
2. Explainability utils  — STOPWORDS, metrics, plotting, SHAP/LIME helpers
                           (shared by all 5 explainability modules)
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

# Explainability-specific dependencies
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config import (
        LOGGING_CONFIG, DATA_CONFIG, RESULTS_CONFIG,
        DATA_PATH, PREPROCESSING_CONFIG,            # needed by explainability section
        MODEL_NAME_MAPPING, FEATURE_NAME_MAPPING,
    )
except ImportError:
    # Fallback if config import fails
    LOGGING_CONFIG = {
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'handlers': {'console': True, 'file': True}
    }
    MODEL_NAME_MAPPING  = {}
    FEATURE_NAME_MAPPING = {}
    DATA_PATH            = Path('data')
    PREPROCESSING_CONFIG = {}


# ==============================================================================
#  SECTION 1 — GENERAL UTILITIES
# ==============================================================================

def setup_logging(log_file: Optional[Path] = None,
                  level: str = "INFO",
                  format_str: Optional[str] = None) -> None:
    """
    Setup logging configuration.

    Args:
        log_file:   Path to log file
        level:      Logging level
        format_str: Log format string
    """
    if format_str is None:
        format_str = LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(message)s')

    logging_level = getattr(logging, level.upper())
    formatter     = logging.Formatter(format_str)

    logger = logging.getLogger()
    logger.setLevel(logging_level)
    logger.handlers.clear()

    if LOGGING_CONFIG.get('handlers', {}).get('console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if LOGGING_CONFIG.get('handlers', {}).get('file', True) and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_data(file_path: Union[str, Path],
              file_type: Optional[str] = None) -> Any:
    """
    Load data from various file formats.

    Args:
        file_path: Path to the data file
        file_type: Type of file ('csv', 'json', 'yaml', 'pickle', 'joblib',
                   'npy', 'npz')

    Returns:
        Loaded data
    """
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
    Save data to various file formats.

    Args:
        data:      Data to save
        file_path: Path to save the file
        file_type: Type of file ('csv', 'json', 'yaml', 'pickle', 'joblib',
                   'npy', 'npz')
        **kwargs:  Additional arguments for specific save functions
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

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
                json.dump(data, f,
                          indent=kwargs.get('indent', 2),
                          ensure_ascii=kwargs.get('ensure_ascii', False))
        elif file_type in ['yaml', 'yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f,
                          default_flow_style=False,
                          allow_unicode=kwargs.get('allow_unicode', True))
        elif file_type in ['pickle', 'pkl']:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f,
                            protocol=kwargs.get('protocol', pickle.HIGHEST_PROTOCOL))
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
    """Get current timestamp as formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Ensure reproducibility by setting random seeds for Python, NumPy,
    TensorFlow and PyTorch.

    Args:
        seed: Random seed value
    """
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
    """
    Format metrics dictionary for display.

    Args:
        metrics:        Dictionary of metric values
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
    Print a formatted section header.

    Args:
        title: Section title
        char:  Character to use for the line
        width: Total width of the header
    """
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
    """
    Standardized file naming conventions across all model types.

    Uses MODEL_NAME_MAPPING and FEATURE_NAME_MAPPING from config.py.
    Pattern: {ModelName}_{FeatureType}_top_{N}_categories_{FileType}.{Ext}
    """

    @staticmethod
    def standardize_model_name(model_name: str) -> str:
        """Convert model name to standard format using config mappings."""
        if model_name in MODEL_NAME_MAPPING:
            return MODEL_NAME_MAPPING[model_name]
        clean = model_name.replace(' ', '_').replace('-', '_')
        return ''.join(c for c in clean if c.isalnum() or c == '_')

    @staticmethod
    def standardize_feature_name(feature_type: str) -> str:
        """Convert feature type to standard format using config mappings."""
        if feature_type in FEATURE_NAME_MAPPING:
            return FEATURE_NAME_MAPPING[feature_type]
        return feature_type.upper()

    @staticmethod
    def generate_confusion_matrix_filename(model_name, feature_type,
                                           n_categories, file_format='png'):
        cm = FileNamingStandard.standardize_model_name(model_name)
        cf = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{cm}_{cf}_top_{n_categories}_categories_confusion_matrix.{file_format}"

    @staticmethod
    def generate_classification_report_filename(model_name, feature_type,
                                                 n_categories, file_format='csv'):
        cm = FileNamingStandard.standardize_model_name(model_name)
        cf = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{cm}_{cf}_top_{n_categories}_categories_classification_report.{file_format}"

    @staticmethod
    def generate_training_history_filename(model_name, n_categories,
                                           file_format='png'):
        cm = FileNamingStandard.standardize_model_name(model_name)
        return f"{cm}_training_history_top_{n_categories}_categories.{file_format}"

    @staticmethod
    def generate_model_filename(model_name, feature_type, n_categories,
                                file_format='pth'):
        cm = FileNamingStandard.standardize_model_name(model_name)
        cf = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{cm}_{cf}_top_{n_categories}_categories_model.{file_format}"

    @staticmethod
    def generate_metrics_filename(model_name, feature_type, n_categories,
                                  file_format='json'):
        cm = FileNamingStandard.standardize_model_name(model_name)
        cf = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{cm}_{cf}_top_{n_categories}_categories_metrics.{file_format}"

    @staticmethod
    def generate_config_filename(model_name, feature_type, n_categories,
                                 file_format='json'):
        cm = FileNamingStandard.standardize_model_name(model_name)
        cf = FileNamingStandard.standardize_feature_name(feature_type)
        return f"{cm}_{cf}_top_{n_categories}_categories_config.{file_format}"


# ==============================================================================
#  SECTION 2 — EXPLAINABILITY UTILITIES
#  Shared by: ml_explainability, dl_explainability, bert_explainability,
#             deepseek_explainability, fusion_explainability
#
#  Import from any of those files with:
#      from src.utils.utils import (
#          STOPWORDS, TARGET_CATEGORIES, FALLBACK_LABELS,
#          load_class_labels, top15_tokens, plot_bar,
#          compute_metrics, build_shap_background,
#          run_global_shap, run_global_lime,
#      )
# ==============================================================================

_expl_logger = logging.getLogger('explainability')

# ── STOPWORDS ─────────────────────────────────────────────────────────────────
# Built once at import time from PREPROCESSING_CONFIG['custom_stopwords']
# (the 79 function words that clean_text() already removed from cleaned_text)
# PLUS a small set of noise tokens that can still slip through after
# lemmatisation / BPE tokenisation.
#
# NEVER add domain words ('api','service','data','platform','cloud','tool' …).
# They survived preprocessing intentionally and must appear in LIME/SHAP output.

def _build_stopwords() -> frozenset:
    """Combine preprocessing custom_stopwords with post-tokenisation noise."""
    base = set(PREPROCESSING_CONFIG.get('custom_stopwords', []))
    noise = {
        # URL / domain residuals that regex cleaning sometimes misses
        'http', 'https', 'www', 'com', 'org', 'net', 'gov', 'edu', 'io',
        # Dataset-specific junk abbreviations after lemmatisation
        'abn', 'eur', 'ma', 'acus', 'id', 'inc', 'json',
        # BPE subword fragments from RoBERTa / DeepSeek tokenisers
        's', 't', 're', 've', 'm', 'll', 'd',
        '##s', '##ing', '##ed', '##tion', '##ly', '##y',
    }
    return frozenset(base | noise)


STOPWORDS: frozenset = _build_stopwords()

# ── TARGET_CATEGORIES ─────────────────────────────────────────────────────────
# 15 fixed categories used for local explanation and cross-model comparison.
# Same list across all 5 explainability modules — defined once here.

TARGET_CATEGORIES: List[str] = [
    "Advertising", "Analytics", "Application Development", "Backend",
    "Banking", "Bitcoin", "Chat", "Cloud", "Data", "Database",
    "Domains", "Education", "Email", "Enterprise", "Entertainment",
]

# ── FALLBACK_LABELS ───────────────────────────────────────────────────────────
# Hardcoded id → category name mapping for all 50 categories.
# Used when the YAML label file cannot be loaded.

FALLBACK_LABELS: Dict[int, str] = {
    0:  "Advertising",            1:  "Analytics",
    2:  "Application Development", 3:  "Backend",
    4:  "Banking",                5:  "Bitcoin",
    6:  "Chat",                   7:  "Cloud",
    8:  "Data",                   9:  "Database",
    10: "Domains",                11: "Education",
    12: "Email",                  13: "Enterprise",
    14: "Entertainment",          15: "Events",
    16: "File Sharing",           17: "Financial",
    18: "Games",                  19: "Government",
    20: "Images",                 21: "Internet of Things",
    22: "Mapping",                23: "Media",
    24: "Medical",                25: "Messaging",
    26: "Music",                  27: "News Services",
    28: "Office",                 29: "Other",
    30: "Payments",               31: "Photos",
    32: "Project Management",     33: "Real Estate",
    34: "Reference",              35: "Science",
    36: "Search",                 37: "Security",
    38: "Shipping",               39: "Social",
    40: "Sports",                 41: "Stocks",
    42: "Storage",                43: "Telephony",
    44: "Tools",                  45: "Transportation",
    46: "Travel",                 47: "Video",
    48: "Weather",                49: "eCommerce",
}


# ── load_class_labels ─────────────────────────────────────────────────────────

def load_class_labels(n_categories: int) -> List[str]:
    """
    Load the ordered list of category name strings for a given experiment size.

    Resolution order:
      1. YAML  — data/processed/labels_top_{n}_categories.yaml
                 written by data_preprocessing.save_label_mapping()
                 supports both list format and dict-with-id_to_label format
      2. Pickle — data/processed/top_{n}_categories/label_encoder.pkl
                 (sklearn LabelEncoder saved by the BERT pipeline)
      3. FALLBACK_LABELS — hardcoded 50-category safety net, always succeeds

    Uses load_data() defined in this same module for consistent file I/O.
    """
    yaml_path = DATA_PATH / "processed" / f"labels_top_{n_categories}_categories.yaml"
    try:
        d = load_data(yaml_path, file_type='yaml')
        if isinstance(d, list):
            return d
        if isinstance(d, dict) and 'id_to_label' in d:
            m = d['id_to_label']
            return [str(m[k]) for k in sorted(m.keys(), key=int)]
    except Exception as e:
        _expl_logger.warning(f"  load_class_labels: YAML warning ({yaml_path.name}): {e}")

    le_path = DATA_PATH / "processed" / f"top_{n_categories}_categories" / "label_encoder.pkl"
    try:
        le = load_data(le_path, file_type='pickle')
        return list(le.classes_)
    except Exception:
        pass

    _expl_logger.warning(
        f"  load_class_labels: using hardcoded fallback for {n_categories} categories"
    )
    return [FALLBACK_LABELS.get(i, f"Class_{i}") for i in range(n_categories)]


# ── top15_tokens ──────────────────────────────────────────────────────────────

def top15_tokens(
    features,
    weights,
    stopwords: frozenset = STOPWORDS,
    clean_glyph: bool = False,
) -> List[Tuple[str, float]]:
    """
    Return up to 15 (token, weight) pairs sorted by |weight| descending,
    with stopword filtering and deduplication.

    Parameters
    ----------
    features    : iterable of token strings
    weights     : iterable of float weights (same length as features)
    stopwords   : set to filter against (default: module-level STOPWORDS)
    clean_glyph : if True, strip RoBERTa 'Ġ' prefix before filtering —
                  set True in bert / deepseek / fusion callers
    """
    paired = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)
    seen: set = set()
    out: List[Tuple[str, float]] = []

    for f, w in paired:
        fs = str(f).lower().strip()
        if clean_glyph:
            fs = fs.replace('ġ', '').strip()
        if fs in stopwords or len(fs) < 2 or (clean_glyph and fs.isnumeric()):
            continue
        if fs not in seen:
            out.append((fs, float(w)))
            seen.add(fs)
        if len(out) >= 15:
            break

    # Relax stopword constraint if fewer than 15 found
    if len(out) < 15:
        for f, w in paired:
            fs = str(f).lower().strip()
            if clean_glyph:
                fs = fs.replace('ġ', '').strip()
            if fs not in seen:
                out.append((fs, float(w)))
                seen.add(fs)
            if len(out) >= 15:
                break

    return out[:15]


# ── plot_bar ──────────────────────────────────────────────────────────────────

def plot_bar(
    items: List[Tuple],
    title: str,
    output_path: Path,
    plot_dpi: int = 300,
) -> None:
    """
    Uniform horizontal bar chart for cross-model LIME / SHAP comparison.

    Design rules — identical across all 5 models so charts are side-by-side
    comparable without rescaling:
      • Fixed x-axis [-1.0, 1.0] — same scale for every model.
      • Always 15 rows (padded with empty strings) — same height always.
      • No value labels on bars — bar length is the visual comparison signal.
      • Positive → blue (#1f77b4),  negative → orange (#ff7f0e).
      • Subtle x-grid, no top/right spines.
    """
    N = 15
    pairs = list(items)[:N]
    while len(pairs) < N:
        pairs.append(('', 0.0))

    names   = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#1f77b4' if w >= 0 else '#ff7f0e' for w in weights]
    ax.barh(range(N), weights, color=colors, height=0.7)

    ax.set_yticks(range(N))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()

    ax.set_xlim(-1.0, 1.0)
    ax.axvline(x=0, color='#333333', linewidth=0.8, linestyle='-')
    ax.set_xlabel("LIME / SHAP Impact Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()


# ── compute_metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    lime_score: Optional[float],
    shap_top15: List[Tuple],
    lime_top15: List[Tuple],
    category_shap_vectors: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute three honest explainability metrics. No clamping, no random noise.

    Fidelity
        sqrt(|R²|) from LIME's local linear fit score.
        Measures how well LIME's surrogate approximates the black-box locally.

    Jaccard
        |SHAP_tokens ∩ LIME_tokens| / |SHAP_tokens ∪ LIME_tokens|
        Measures agreement on which tokens matter (set overlap of top-15).

    Stability
        Mean pairwise Spearman-r of SHAP vectors across same-category samples.
        Measures how consistently the model explains the same category.
        Falls back to Jaccard when fewer than 2 SHAP vectors are available.
    """
    raw_r2   = abs(lime_score) if lime_score is not None else 0.0
    fidelity = round(float(np.sqrt(min(raw_r2, 1.0))), 4)

    s_set   = {str(x[0]).lower().strip() for x in shap_top15 if x[0]}
    l_set   = {str(x[0]).lower().strip() for x in lime_top15 if x[0]}
    union   = s_set | l_set
    jaccard = round(len(s_set & l_set) / len(union), 4) if union else 0.0

    if category_shap_vectors and len(category_shap_vectors) >= 2:
        corrs: List[float] = []
        ref = category_shap_vectors[0]
        for vec in category_shap_vectors[1:]:
            if len(vec) == len(ref) and np.std(vec) > 1e-9 and np.std(ref) > 1e-9:
                r, _ = spearmanr(ref, vec)
                corrs.append(float(r))
        stability = round(float(np.mean(corrs)), 4) if corrs else jaccard
    else:
        stability = jaccard

    return {'Fidelity': fidelity, 'Jaccard': jaccard, 'Stability': stability}


# ── build_shap_background ─────────────────────────────────────────────────────

def build_shap_background(X_train: np.ndarray, n: int = 100) -> np.ndarray:
    """
    Compress the training set into n kmeans cluster centres for use as the
    KernelExplainer background distribution.

    n=100 is a good balance for 384-dim SBERT embeddings — accurate enough,
    fast enough.
    """
    n = min(n, len(X_train))
    _expl_logger.info(f"  Building KernelExplainer background ({n} kmeans clusters)…")
    return shap.kmeans(X_train, n).data


# ── run_global_shap ───────────────────────────────────────────────────────────

def run_global_shap(
    kernel_explainer,
    X_sample: np.ndarray,
    class_labels: List[str],
    model_name: str,
    output_path: Path,
    plot_dpi: int = 300,
) -> None:
    """
    Run global SHAP over X_sample, compute per-target-category mean |SHAP|,
    and save a category importance bar chart.

    Used by ML and DL models (KernelExplainer on SBERT embeddings).
    BERT / DeepSeek / Fusion use shap.Explainer (text masker) instead and
    build their global aggregation inline.
    """
    _expl_logger.info(f"  Global SHAP for {model_name} ({len(X_sample)} samples)…")
    shap_vals = kernel_explainer.shap_values(X_sample, silent=True)

    category_impact: List[Tuple[str, float]] = []
    if isinstance(shap_vals, list):
        for idx, sv in enumerate(shap_vals):
            if idx < len(class_labels) and class_labels[idx] in TARGET_CATEGORIES:
                category_impact.append(
                    (class_labels[idx], float(np.mean(np.abs(sv))))
                )
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        for idx in range(shap_vals.shape[2]):
            if idx < len(class_labels) and class_labels[idx] in TARGET_CATEGORIES:
                category_impact.append(
                    (class_labels[idx], float(np.mean(np.abs(shap_vals[:, :, idx]))))
                )

    if not category_impact:
        _expl_logger.warning(f"  No category impact extracted for {model_name}.")
        return

    # Normalise if raw margins are very large (e.g. XGBoost)
    vals = [v for _, v in category_impact]
    if max(vals, default=0) > 100:
        total = sum(vals) + 1e-9
        category_impact = [(c, v / total) for c, v in category_impact]

    # Pad any missing target categories with 0
    existing = {c for c, _ in category_impact}
    for cat in TARGET_CATEGORIES:
        if cat not in existing:
            category_impact.append((cat, 0.0))
    category_impact.sort(key=lambda x: x[1], reverse=True)

    plot_bar(
        category_impact,
        f"Global Category Importance (SBERT) — {model_name}",
        output_path,
        plot_dpi=plot_dpi,
    )


# ── run_global_lime ───────────────────────────────────────────────────────────

def run_global_lime(
    lime_explainer,
    predict_fn,
    test_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    sample_limit: int = 15,
    clean_glyph: bool = False,
    plot_dpi: int = 300,
) -> None:
    """
    Aggregate LIME word weights across up to sample_limit samples and save a
    global importance bar chart.

    Parameters
    ----------
    lime_explainer : LimeTextExplainer instance
    predict_fn     : callable  texts → probability array
    test_df        : DataFrame with 'cleaned_text' column
    model_name     : used in chart title and log messages
    output_path    : full path for the saved PNG
    sample_limit   : max distinct prediction classes to sample (default 15)
    clean_glyph    : strip RoBERTa 'Ġ' byte prefix — set True for
                     BERT / DeepSeek / Fusion
    plot_dpi       : output resolution
    """
    _expl_logger.info(f"  Global LIME for {model_name} ({sample_limit} samples)…")
    global_w: Dict[str, float] = defaultdict(float)
    seen: set = set()

    for i in range(len(test_df)):
        if len(seen) >= sample_limit:
            break
        try:
            text    = str(test_df.iloc[i]['cleaned_text'])
            probs   = predict_fn([text])[0]
            top_cls = int(np.argmax(probs))
            exp = lime_explainer.explain_instance(
                text, predict_fn,
                labels=[top_cls], num_features=15, num_samples=300,
            )
            for word, w in exp.as_list(label=top_cls):
                fs = word.lower().strip()
                if clean_glyph:
                    fs = fs.replace('ġ', '').strip()
                if fs not in STOPWORDS and len(fs) >= 2 and not fs.isnumeric():
                    global_w[fs] += abs(w)
            seen.add(top_cls)
        except Exception:
            continue

    if global_w:
        items = sorted(global_w.items(), key=lambda x: x[1], reverse=True)
        top   = top15_tokens(
            [k for k, _ in items],
            [v for _, v in items],
            clean_glyph=clean_glyph,
        )
        plot_bar(
            top,
            f"Global LIME Aggregated — {model_name}",
            output_path,
            plot_dpi=plot_dpi,
        )


# ==============================================================================
#  EXPORTS
# ==============================================================================

__all__ = [
    # Section 1 — General utilities
    'setup_logging',
    'load_data',
    'save_data',
    'get_timestamp',
    'ensure_reproducibility',
    'format_metrics',
    'print_section_header',
    'FileNamingStandard',
    # Section 2 — Explainability utilities
    'STOPWORDS',
    'TARGET_CATEGORIES',
    'FALLBACK_LABELS',
    'load_class_labels',
    'top15_tokens',
    'plot_bar',
    'compute_metrics',
    'build_shap_background',
    'run_global_shap',
    'run_global_lime',
]

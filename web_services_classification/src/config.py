"""
Configuration file for Web Services Classification Project
Enhanced to ensure consistency across all model types
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"
LOGS_PATH = PROJECT_ROOT / "logs"

# Random seed for reproducibility
RANDOM_SEED = 42

# Category sizes to process
CATEGORY_SIZES = [50]

MODEL_TYPES = {
    'ml': 'ml',
    'dl': 'dl',
    'bert': 'bert',
    'roberta': 'bert',  # RoBERTa uses BERT directories
    'deepseek': 'deepseek',
    'fusion': 'fusion'
}
# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'console': True,
        'file': True
    },
    'log_files': {
        'data_analysis': LOGS_PATH / 'data_analysis.log',
        'preprocessing': LOGS_PATH / 'preprocessing.log',
        'feature_extraction': LOGS_PATH / 'feature_extraction.log',
        'training': LOGS_PATH / 'training.log',
        'evaluation': LOGS_PATH / 'evaluation.log'
    }
}

# =============================================================================
# Data Configuration
# =============================================================================
DATA_CONFIG = {
    # Input data
    "raw_data_path": DATA_PATH / "raw" / "web_services_dataset.csv",
    
    # Processed data path (added for DL models)
    "processed_data_path": DATA_PATH / "processed",

    # Column names (adjust if CSV schema changes)
    "text_column": "Service Description",          # service description text
    "target_column": "Service Classification",     # classification labels
}

# =============================================================================
# Step 1: Analysis - Uniform top_{n}_categories naming
# =============================================================================
ANALYSIS_PATH = DATA_PATH / "analysis"
ANALYSIS_CONFIG = {
    "overall": ANALYSIS_PATH / "overall",                    # global dataset stats & plots
    "category_wise": ANALYSIS_PATH / "top_{n}_categories",   # Top-N stats & distributions (uniform naming)
    "comparisons": ANALYSIS_PATH / "comparisons"             # cross-TopN comparisons
}

# =============================================================================
# Step 2: Preprocessing - Uniform top_{n}_categories naming
# =============================================================================
PREPROCESS_PATH = DATA_PATH / "processed"

PREPROCESSING_CONFIG = {
    "processed_data": str(PREPROCESS_PATH / "top_{n}_categories"),        # cleaned datasets (uniform naming)
    "splits": str(PREPROCESS_PATH / "splits" / "top_{n}_categories"),     # train/val/test splits (uniform naming)
    "labels": str(PREPROCESS_PATH / "labels_top_{n}_categories.yaml"),    # label mappings (uniform naming)

    # Basic text cleaning
    "remove_stopwords": False,
    "remove_numbers": True,
    "lemmatization": True,
    "lowercase": True,
    "remove_punctuation": True,

    # Word filtering
    "min_word_length": 2,
    "max_word_length": 50,

    # Custom stopwords
    "custom_stopwords": ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to"],

    # Advanced cleaning
    "remove_urls": True,
    "remove_emails": True,
    "normalize_whitespace": True
}

# =============================================================================
# Step 3: Features - Uniform top_{n}_categories naming
# =============================================================================
FEATURES_PATH = DATA_PATH / "features"

FEATURES_CONFIG = {
    "tfidf_path": str(FEATURES_PATH / "tfidf" / "top_{n}_categories"),   # TF-IDF vectors (consistent naming)
    "sbert_path": str(FEATURES_PATH / "sbert" / "top_{n}_categories"),   # SBERT embeddings (consistent naming)
    "plots": FEATURES_PATH / "feature_plots",                            # tfidf_top_terms, sbert_clusters
    "stats": FEATURES_PATH / "feature_stats",                            # vocab/embedding stats

    # TF-IDF settings
    "tfidf": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True
    },

    # SBERT settings
    "sbert": {
        "model_name": "all-MiniLM-L6-v2",
        "max_length": 512,
        "batch_size": 32,
        "device": "cpu",  # change to "cuda" for GPU
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "convert_to_tensor": False,
        "convert_to_numpy": True
    }
}

# =============================================================================
# Data Split Configuration
# =============================================================================
SPLIT_CONFIG = {
    "train_size": 0.80,       # proportion of data for training
    "val_size": 0.10,         # proportion of data for validation
    "test_size": 0.10,        # proportion of data for testing
    "random_state": RANDOM_SEED,  # reproducibility
    "stratify": True,         # preserve label distribution in splits
    "shuffle": True           # shuffle before splitting
}


# Results configuration - CRITICAL for proper file organization
RESULTS_CONFIG = {
    # ML Results
    'ml_results_path': RESULTS_PATH / "ml",
    'ml_comparisons_path': RESULTS_PATH / "ml" / "comparisons",
    'ml_category_paths': {
        n: RESULTS_PATH / "ml" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # DL Results  
    'dl_results_path': RESULTS_PATH / "dl",
    'dl_comparisons_path': RESULTS_PATH / "dl" / "comparisons",
    'dl_category_paths': {
        n: RESULTS_PATH / "dl" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # BERT Results
    'bert_results_path': RESULTS_PATH / "bert",
    'bert_comparisons_path': RESULTS_PATH / "bert" / "comparisons",
    'bert_category_paths': {
        n: RESULTS_PATH / "bert" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # DeepSeek Results
    'deepseek_results_path': RESULTS_PATH / "deepseek", 
    'deepseek_comparisons_path': RESULTS_PATH / "deepseek" / "comparisons",
    'deepseek_category_paths': {
        n: RESULTS_PATH / "deepseek" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    
    # fusion Results
    'fusion_results_path': RESULTS_PATH / "fusion",
    'fusion_comparisons_path': RESULTS_PATH / "fusion" / "comparisons",
    'fusion_category_paths': {
        n: RESULTS_PATH / "fusion" / f"top_{n}_categories" for n in CATEGORY_SIZES
    },
    # Overall Results
    'overall_results_path': RESULTS_PATH / "overall",
}

# Saved models configuration - CRITICAL for model storage
SAVED_MODELS_CONFIG = {
    'ml_models_path': MODELS_PATH / "saved_models" / "ml_models",
    'dl_models_path': MODELS_PATH / "saved_models" / "dl_models", 
    'bert_models_path': MODELS_PATH / "saved_models" / "bert_models",
    'deepseek_models_path': MODELS_PATH / "saved_models" / "deepseek_models",
    'fusion_models_path': MODELS_PATH / "saved_models" / "fusion_models" 
}

# ML Models configuration
ML_CONFIG = {
    'model_type': 'ml',  # ← Added
    'models': ['LogisticRegression', 'RandomForest', 'XGBoost'],
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'xgboost': {
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
}

# DL Models configuration
DL_CONFIG = {
    'model_type': 'dl',  # ← Added
    'models': ['BiLSTM'],
    'feature_types': ['tfidf', 'sbert'],
    'bilstm': {
        'lstm_units': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']
    },
    'callbacks': {
        'early_stopping': {
            'monitor': 'val_accuracy',
            'patience': 3,
            'restore_best_weights': True
        },
        'model_checkpoint': {
            'monitor': 'val_accuracy',
            'save_best_only': True,
            'save_weights_only': False
        },
        'reduce_lr': {
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 2,
            'min_lr': 1e-6
        }
    }
}

# BERT Models configuration
BERT_CONFIG = {
    'model_type': 'bert',  # ← Added
    'models': ['roberta-base','roberta-large'],
    'available_models': {
        'roberta_base': 'roberta-base',
        'roberta_large': 'roberta-large'
    },
    'max_length': 512,
    'num_train_epochs': 3,
    'eval_strategy': 'epoch',
    'logging_strategy': 'epoch',
    'logging_steps': 100,
    'save_strategy': 'epoch',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_accuracy',
    'greater_is_better': True,
    'seed': RANDOM_SEED,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'per_device_train_batch_size': 16,  # Default batch size
    'per_device_eval_batch_size': 32,   # Default eval batch size
    'batch_sizes': {
        'roberta-base': {
            'train_batch_size': 16,
            'eval_batch_size': 32
        },
        'roberta-large': {
            'train_batch_size': 8,
            'eval_batch_size': 16
        }
    }
}

# DeepSeek Models configuration
DEEPSEEK_CONFIG = {
    'model_type': 'deepseek',  # ← Adde
    'available_models': {"deepseek": "deepseek-ai/deepseek-llm-7b-base"},
    'models': [ 'deepseek-ai/deepseek-llm-7b-base'],
    'trust_remote_code': True,
    'max_length': 512,
    'padding': 'max_length',
    'truncation': True,
    'num_train_epochs': 3,
    'eval_strategy': 'epoch',
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 4,
    'logging_steps': 100,
    'save_strategy': 'epoch',
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_accuracy',
    'greater_is_better': True,
    'random_state': RANDOM_SEED,
    'learning_rate': 1e-4,
    'gradient_checkpointing': True,
    'quantization': {
        'load_in_4bit': True,
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'float16'
    },
    'lora': {
        'task_type': 'SEQ_CLS',
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'bias': 'none'
    },
    'text_preprocessing': {
        'clean_text': True
    },
    'batch_sizes': {
        'deepseek-ai/deepseek-llm-7b-base': {
            'train_batch_size': 4,
            'eval_batch_size': 8
        }
    }
}

# ============================================================================
# FUSION CONFIGURATION (Single Hybrid: RoBERTa + DeepSeek)
# ============================================================================

FUSION_CONFIG = {
    'model_type': 'fusion',
    'models': ['deepseek-ai/deepseek-llm-7b-base','roberta-base'],
    'deepseek_model': 'deepseek-ai/deepseek-llm-7b-base',
    'roberta_model': 'roberta-base',
    'fusion_types': ['concat', 'average', 'weighted', 'gating'],  # Different fusion strategies
    'feature_types': ['Concat', 'Average', 'Weighted', 'Gating'],  # Standardized feature type names
    'common_dim': 768,  # Common embedding dimension
    'max_length': 128,
    'num_train_epochs': 15,
    'batch_size': 8,
    'eval_batch_size': 16,
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'dropout': 0.3,
    'scheduler': {
        'mode': 'max',
        'patience': 2,
        'factor': 0.5
    }
}


# ============================================================================
# FILE NAMING STANDARDS AND MODEL/FEATURE MAPPINGS
# ============================================================================

# Model name standardization mapping
MODEL_NAME_MAPPING = {
    # ML Models
    'logistic_regression': 'LogisticRegression',
    'LogisticRegression': 'LogisticRegression',
    'random_forest': 'RandomForest',
    'RandomForest': 'RandomForest',
    'xgboost': 'XGBoost',
    'XGBoost': 'XGBoost',
    
    # DL Models
    'bilstm': 'BiLSTM',
    'BiLSTM': 'BiLSTM',
    
    # BERT Models
    'roberta_base': 'RoBERTa_Base',
    'RoBERTa_Base': 'RoBERTa_Base',
    'roberta-base': 'RoBERTa_Base',
    'roberta_large': 'RoBERTa_Large',
    'RoBERTa_Large': 'RoBERTa_Large',
    'roberta-large': 'RoBERTa_Large',
    
    # DeepSeek Models
    'deepseek_7b_base': 'DeepSeek_7B_Base',
    'DeepSeek_7B_Base': 'DeepSeek_7B_Base',
    'deepseek-ai/deepseek-llm-7b-base': 'DeepSeek_7B_Base',
    
    # Fusion Models - ALL map to base name, fusion type becomes feature type
    'deepseek_roberta_fusion': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek_RoBERTa_Fusion': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek-RoBERTa-Fusion': 'DeepSeek_RoBERTa_Fusion',
    'deepseek_roberta_fusion_concat': 'DeepSeek_RoBERTa_Fusion',
    'deepseek_roberta_fusion_average': 'DeepSeek_RoBERTa_Fusion',
    'deepseek_roberta_fusion_weighted': 'DeepSeek_RoBERTa_Fusion',
    'deepseek_roberta_fusion_gating': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek-RoBERTa-Fusion-Concat': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek-RoBERTa-Fusion-Average': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek-RoBERTa-Fusion-Weighted': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek-RoBERTa-Fusion-Gating': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek_RoBERTa_Fusion_Concat': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek_RoBERTa_Fusion_Average': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek_RoBERTa_Fusion_Weighted': 'DeepSeek_RoBERTa_Fusion',
    'DeepSeek_RoBERTa_Fusion_Gating': 'DeepSeek_RoBERTa_Fusion',
}

# Feature type standardization mapping
FEATURE_NAME_MAPPING = {
    'tfidf': 'TFIDF',
    'TFIDF': 'TFIDF',
    'sbert': 'SBERT',
    'SBERT': 'SBERT',
    'raw_text': 'RawText',
    'RawText': 'RawText',
    # Fusion feature types - simplified names
    # Fusion feature types
    'concat': 'Concat',
    'Concat': 'Concat',
    'average': 'Average',
    'Average': 'Average',
    'weighted': 'Weighted',
    'Weighted': 'Weighted',
    'gating': 'Gating',
    'Gating': 'Gating',
}

EXPLAINABILITY_CONFIG = {
    # ML Explainability Results
    'ml_explainability_path': RESULTS_PATH / "ml" / "explainability",
    'ml_explainability_category_paths': {
        n: RESULTS_PATH / "ml" / f"top_{n}_categories" / "explainability" 
        for n in CATEGORY_SIZES
    },
    
    # Subdirectories for different explainability methods
    'explainability_subdirs': {
        'shap': 'shap',
        'lime': 'lime',
        'combined': 'combined',
        'feature_importance': 'feature_importance'
    },
    
    # Output formats
    'plot_dpi': 300,
    'plot_format': 'png',
    'max_features_display': 20,
    
    # SHAP specific settings
    'shap_background_samples': 100,
    'shap_explain_samples': 50,
    
    # LIME specific settings
    'lime_num_samples': 5000,
    'lime_num_features': 20,
    'lime_num_instances': 5
}

def create_all_directories():
    """Create all necessary directories"""
    directories = [
        DATA_PATH / "raw",
        DATA_PATH / "processed", 
        DATA_PATH / "splits",
        DATA_PATH / "features" / "tfidf",
        DATA_PATH / "features" / "sbert",
        DATA_PATH / "features" / "plots",
        DATA_PATH / "analysis",
        MODELS_PATH / "saved_models" / "ml_models",
        MODELS_PATH / "saved_models" / "dl_models",
        MODELS_PATH / "saved_models" / "bert_models", 
        MODELS_PATH / "saved_models" / "deepseek_models",
        MODELS_PATH / "saved_models" / "fusion_models",
        RESULTS_PATH / "ml" / "comparisons",
        RESULTS_PATH / "dl" / "comparisons",
        RESULTS_PATH / "bert" / "comparisons",
        RESULTS_PATH / "deepseek" / "comparisons",
        RESULTS_PATH / "fusion" / "comparisons",
        RESULTS_PATH / "overall",
        RESULTS_PATH / "ml" / "explainability",
        LOGS_PATH
    ]
    
    # Create category-specific directories
    for n_categories in CATEGORY_SIZES:
        directories.extend([
            DATA_PATH / "splits" / f"top_{n_categories}_categories",
            DATA_PATH / "processed" / f"top_{n_categories}_categories",
            DATA_PATH / "features" / "tfidf" / f"top_{n_categories}_categories",
            DATA_PATH / "features" / "sbert" / f"top_{n_categories}_categories",
            DATA_PATH / "analysis" / f"top_{n_categories}_categories",
            RESULTS_PATH / "ml" / f"top_{n_categories}_categories",
            RESULTS_PATH / "dl" / f"top_{n_categories}_categories", 
            RESULTS_PATH / "bert" / f"top_{n_categories}_categories",
            RESULTS_PATH / "deepseek" / f"top_{n_categories}_categories",
            RESULTS_PATH / "fusion" / f"top_{n_categories}_categories",
            RESULTS_PATH / "ml" / f"top_{n_categories}_categories" / "explainability",
            RESULTS_PATH / "ml" / f"top_{n_categories}_categories" / "explainability" / "shap",
            RESULTS_PATH / "ml" / f"top_{n_categories}_categories" / "explainability" / "lime",
            RESULTS_PATH / "ml" / f"top_{n_categories}_categories" / "explainability" / "combined",
            RESULTS_PATH / "ml" / f"top_{n_categories}_categories" / "explainability" / "feature_importance"
            

        ])
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Created {len(directories)} directories")

if __name__ == "__main__":
    create_all_directories()
    print("Configuration initialized and directories created.")

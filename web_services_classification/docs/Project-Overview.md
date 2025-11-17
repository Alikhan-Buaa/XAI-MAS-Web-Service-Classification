# Web Service Classification Project Overview

## Executive Summary

This project implements a comprehensive machine learning pipeline for classifying web services into their respective categories. The system evaluates and compares multiple model architectures—from traditional ML to state-of-the-art transformer models—to establish robust performance baselines for web service classification tasks.

## Project Architecture

### Core Components

The project follows a modular, phase-based architecture with clear separation of concerns:

- **Data Pipeline**: Handles raw data ingestion, analysis, and preprocessing for top-50 category classification
- **Feature Engineering**: Implements dual feature extraction strategies using TF-IDF and SBERT embeddings
- **Model Zoo**: Encompasses 40+ trained models across four paradigms:
  - Traditional ML (Logistic Regression, Random Forest, XGBoost)
  - Deep Learning (BiLSTM)
  - Transformers (RoBERTa Small & Large variants)
  - Fusion Models (DeepSeek + RoBERTa hybrid architectures)

### Key Technical Specifications

- **Dataset**: Balanced samples from top-50 web service categories
- **Data Splits**: Fixed 80/10/10 train/validation/test distribution
- **Evaluation Metrics**: Top-K accuracy (K=1,3,5), Macro/Micro F1 scores
- **Reproducibility**: YAML-based configuration management
- **Visualization**: Comprehensive performance dashboards with confusion matrices and ranking curves

## Workflow Pipeline

### Phase-Based Execution

The system executes through eight sequential phases, each building upon the previous:

1. **Analysis Phase**: Statistical profiling of raw web service data
2. **Preprocessing Phase**: Text normalization, cleaning, and category balancing
3. **Feature Extraction**: Generation of TF-IDF vectors and SBERT embeddings
4. **ML Training**: Traditional classifier training with hyperparameter optimization
5. **DL Training**: BiLSTM neural network training with embedding layers
6. **BERT Training**: Fine-tuning RoBERTa models for sequence classification
7. **Fusion Training**: Hybrid model development combining embeddings with classifiers
8. **Evaluation & Visualization**: Cross-model benchmarking and performance analysis

### Model Comparison Framework

The project implements a sophisticated comparison framework that:
- Generates unified leaderboards across all model types
- Produces Top-K performance curves for ranking quality assessment
- Creates confusion matrices for error pattern analysis
- Analyzes category-wise difficulty and model strengths

## Key Innovations

### Multi-Paradigm Approach
Systematic evaluation across traditional ML, deep learning, and transformer architectures provides comprehensive insights into model strengths for web service classification.

### Fusion Architecture
Novel combination of DeepSeek and RoBERTa embeddings with downstream classifiers, exploring four fusion strategies: concatenation, averaging, weighted combination, and gating mechanisms.

### Reproducible Research
Complete configuration management through YAML files ensures all experiments are reproducible, with fixed random seeds and versioned data splits.

## Results Structure

The project generates organized outputs across multiple dimensions:

- **Model-Specific Results**: Individual performance metrics and predictions for each trained model
- **Comparative Analysis**: Cross-model benchmarking charts and statistical comparisons
- **Category Analysis**: Performance breakdown by service category complexity
- **Visualization Suite**: Interactive charts, confusion matrices, and ranking curves

## Technical Stack

- **Core Framework**: Python with scikit-learn, PyTorch, Transformers
- **Feature Engineering**: TF-IDF vectorization, Sentence-BERT embeddings
- **Deep Learning**: BiLSTM architectures with attention mechanisms
- **Transformers**: Hugging Face ecosystem with RoBERTa models
- **Visualization**: Matplotlib, Seaborn for comprehensive result analysis

## Project Impact

This comprehensive benchmarking framework establishes:
- Baseline performance metrics for web service classification tasks
- Comparative insights across different modeling paradigms
- Reproducible experimental pipeline for future research
- Production-ready models for real-world deployment

The modular architecture ensures easy extension for additional models, features, or evaluation metrics, making it a valuable resource for both research and industrial applications in web service categorization.

## Project Directory Structure

```
Phase-04/
├── web_services_classification/      # Main project directory
│   ├── data/                        # Data pipeline
│   │   ├── raw/                     # Original dataset
│   │   ├── processed/               # Cleaned and preprocessed data
│   │   │   ├── splits/              # Train/val/test splits
│   │   │   └── top_50_categories/   # Category-specific data
│   │   ├── features/                # Extracted features
│   │   │   ├── tfidf/               # TF-IDF vectors
│   │   │   └── sbert/               # SBERT embeddings
│   │   └── analysis/                # Data analysis reports
│   │       ├── overall/             # Full dataset statistics
│   │       └── top_50_categories/   # Category-wise analysis
│   │
│   ├── results/                     # Model outputs and evaluations
│   │   ├── ml/                      # Traditional ML results
│   │   │   ├── top_50_categories/   # Model predictions
│   │   │   └── comparisons/charts/  # Performance visualizations
│   │   ├── dl/                      # Deep learning results
│   │   │   ├── top_50_categories/   # BiLSTM outputs
│   │   │   └── comparisons/charts/  # DL performance charts
│   │   ├── bert/                    # Transformer results
│   │   │   ├── top_50_categories/   # RoBERTa predictions
│   │   │   └── comparisons/charts/  # BERT benchmarks
│   │   ├── fusion/                  # Fusion model results
│   │   │   ├── top_50_categories/   # Hybrid model outputs
│   │   │   └── comparisons/charts/  # Fusion performance
│   │   ├── comparison/              # Cross-model analysis
│   │   │   └── all_models_analysis/ # Unified benchmarks
│   │   └── overall/                 # Aggregate metrics
│   │
│   ├── src/                         # Source code
│   │   ├── preprocessing/           # Data cleaning modules
│   │   ├── modeling/                # Model implementations
│   │   ├── evaluation/              # Evaluation metrics
│   │   ├── explainability/          # LIME/SHAP modules
│   │   ├── visualization/           # Plotting utilities
│   │   └── utils/                   # Helper functions
│   │
│   ├── docs/                        # Documentation
│   ├── main.py                      # Main execution script
│   └── requirements.txt             # Python dependencies
│
├── comparison/                       # External comparison tools
│   └── cross_model_analysis/         # Advanced analytics
│
└── Web-services-data-analysis-balancing/  # Data balancing utilities
    └── data/                        # Supplementary datasets
        ├── analysis/                # Category-wise statistics
        └── raw/                     # Original data sources
```

## Installation and Execution Guide

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Git for repository cloning
- CUDA-capable GPU (optional, for faster deep learning training)
- Minimum 16GB RAM for model training
- 50GB free disk space for models and data

### Step-by-Step Setup

#### 1. Environment Setup
```bash
# Clone the repository
git clone git@github.com:Alikhan-Buaa/MAS-Web-Service-Classification.git

# Navigate to project directory
cd MAS-Web-Service-Classification/Phase-04/web_services_classification/

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -m nltk.downloader punkt stopwords wordnet
```

#### 2. Phase-wise Execution

Execute each phase sequentially for complete pipeline:

```bash
# Phase 1: Data Analysis
# Analyzes raw data distribution and statistics
python main.py --phase analysis
# Output: ./data/analysis/overall/ and ./data/analysis/top_50_categories/

# Phase 2: Data Preprocessing
# Cleans text, handles missing values, balances categories
python main.py --phase preprocessing
# Output: ./data/processed/top_50_categories/

# Phase 3: Feature Extraction
# Generates TF-IDF and SBERT embeddings
python main.py --phase features
# Output: ./data/features/tfidf/ and ./data/features/sbert/

# Phase 4: Traditional ML Training
# Trains LogisticRegression, RandomForest, XGBoost
python main.py --phase ml_training
# Output: ./results/ml/top_50_categories/

# Phase 5: Deep Learning Training
# Trains BiLSTM models with embeddings
python main.py --phase dl_training
# Output: ./results/dl/top_50_categories/

# Phase 6: BERT Fine-tuning
# Fine-tunes RoBERTa Small and Large models
python main.py --phase bert_training
# Output: ./results/bert/top_50_categories/

# Phase 7: Fusion Model Training
# Trains hybrid DeepSeek + RoBERTa models
python main.py --phase fusion_training
# Output: ./results/fusion/top_50_categories/

# Phase 8: Evaluation and Benchmarking
# Generates comprehensive performance reports
python main.py --phase evaluation
# Output: ./results/comparison/all_models_analysis/

# Phase 9: Visualization Generation
# Creates charts, confusion matrices, ranking curves
python main.py --phase visualize
# Output: ./results/*/comparisons/charts/
```

### Quick Start Options

#### Run Complete Pipeline
```bash
# Execute all phases automatically
python main.py --phase all 
```

#### Run Specific Model Type Only
```bash
# Traditional ML only
python main.py --phase ml_pipeline

# Deep Learning only
python main.py --phase dl_pipeline

# BERT models only
python main.py --phase bert_pipeline
```

### Output Verification

After execution, verify outputs:

```bash
# Check model performances
cat results/comparison/all_models_analysis/leaderboard.csv

# View confusion matrices
ls results/*/comparisons/charts/*confusion_matrix.png

# Review evaluation metrics
cat results/overall/evaluation_summary.json
```

### Troubleshooting

Common issues and solutions:

- **Memory Error**: Reduce batch size in configuration or use fewer categories
- **CUDA Error**: Set `device: cpu` in config file if GPU unavailable
- **Missing Dependencies**: Run `pip install -r requirements.txt --upgrade`
- **Data Not Found**: Ensure raw data is placed in `./data/raw/` directory

### Performance Tips

- Use GPU for BERT and deep learning phases (10x speedup)
- Enable multiprocessing for ML models: `--n_jobs -1`
- Cache embeddings to avoid recomputation: `--use_cache true`
- Adjust batch sizes based on available memory

The complete pipeline typically takes 6-10 hours on GPU or 24-48 hours on CPU for all 40 models.
# Web Services Classification - Module Architecture Overview

## 1. Project Configuration Module (`config.py`)

### Purpose
Central configuration hub managing all project parameters, paths, and model hyperparameters through a unified interface.

### Core Components

#### **Path Management**
- **Base Paths**: Establishes project structure with `DATA_PATH`, `MODELS_PATH`, `RESULTS_PATH`, and `LOGS_PATH`
- **Dynamic Path Generation**: Creates category-specific paths using template patterns like `top_{n}_categories`
- **Automatic Directory Creation**: `create_all_directories()` ensures all required paths exist before execution

#### **Model Configuration Sections**
- **ML_CONFIG**: Parameters for LogisticRegression, RandomForest, and XGBoost
- **DL_CONFIG**: BiLSTM architecture with embedding specifications
- **BERT_CONFIG**: RoBERTa base/large variants with training parameters
- **DEEPSEEK_CONFIG**: 7B parameter model with LoRA and quantization settings
- **FUSION_CONFIG**: Hybrid model combining DeepSeek and RoBERTa embeddings

#### **Standardization Mappings**
- **MODEL_NAME_MAPPING**: Ensures consistent model naming across all outputs
- **FEATURE_NAME_MAPPING**: Standardizes feature type references (TFIDF, SBERT, Concat, etc.)
- **MODEL_TYPES**: Maps model families to their respective directories

### Storage Strategy
```
Configuration Flow:
config.py → defines paths → creates directories → standardizes names
         ↓                ↓                    ↓
    Parameters      File Structure      Consistent Outputs
```

## 2. Main Execution Pipeline (`main.py`)

### Purpose
Orchestrates the entire classification workflow through phase-based execution with comprehensive logging and error handling.

### Pipeline Manager Architecture

#### **Phase Management**
- **Sequential Execution**: Nine distinct phases from analysis to visualization
- **Timing Tracking**: Records execution time for each phase
- **Error Recovery**: Captures failures without stopping entire pipeline
- **Result Aggregation**: Collects outputs from all phases for final summary

#### **Execution Phases**
1. **analysis**: Statistical profiling of raw data
2. **preprocessing**: Text cleaning and normalization
3. **features**: TF-IDF and SBERT extraction
4. **ml_training**: Traditional classifier training
5. **dl_training**: Neural network training
6. **bert_training**: Transformer fine-tuning
7. **fusion_training**: Hybrid model development
8. **evaluation**: Performance metrics calculation
9. **visualize**: Chart and plot generation

#### **Logging System**
- Phase-specific log files for detailed debugging
- Execution summaries with timing metrics
- JSON-serialized results for programmatic access
- Automatic cleanup of non-serializable objects (sparse matrices, numpy arrays)

### Command Interface
```bash
# Phase execution patterns
python main.py --phase [phase_name]     # Single phase
python main.py --phase all              # Complete pipeline
python main.py --phase ml_pipeline      # Model-specific pipeline
```

## 3. Data Analysis Module (`src/preprocessing/data_analysis.py`)

### Purpose
Comprehensive statistical analysis and visualization of web service datasets before processing.

### Analysis Components

#### **Overall Statistics**
- Dataset size and shape metrics
- Category distribution analysis
- Text length statistics (words, characters)
- Missing value detection
- Class imbalance measurements

#### **Category-wise Analysis**
- Per-category sample counts
- Text complexity metrics by category
- Distribution plots and histograms
- Category co-occurrence patterns

#### **Visualization Outputs**
- Category distribution bar plots
- Text length distribution curves
- Word cloud generation
- Correlation heatmaps

### Storage Locations
```
data/analysis/
├── overall/                 # Global dataset metrics
│   ├── statistics.json
│   └── distribution_plots/
└── top_50_categories/       # Category-specific analysis
    ├── category_stats.csv
    └── visualizations/
```

## 4. Preprocessing Module (`src/preprocessing/data_preprocessing.py`)

### Purpose
Transforms raw text data into clean, normalized format suitable for feature extraction.

### Processing Pipeline

#### **Text Cleaning Steps**
1. **Normalization**: Lowercase conversion, whitespace standardization
2. **Removal**: URLs, emails, special characters, numbers
3. **Linguistic**: Lemmatization, stopword removal
4. **Filtering**: Word length constraints (2-50 characters)

#### **Data Splitting**
- **Stratified Splits**: Maintains category proportions
- **Fixed Ratios**: 80% train, 10% validation, 10% test
- **Reproducible**: Seeded random state for consistency

#### **Category Balancing**
- Identifies and handles imbalanced categories
- Optional oversampling/undersampling strategies
- Preserves original distribution metrics

### Output Structure
```
data/processed/
├── top_50_categories/
│   ├── train_processed.csv
│   ├── val_processed.csv
│   └── test_processed.csv
└── splits/top_50_categories/
    └── split_metadata.json
```

## 5. Feature Extraction Module (`src/preprocessing/feature_extraction.py`)

### Purpose
Converts preprocessed text into numerical representations for model consumption.

### Feature Types

#### **TF-IDF Features**
- **Configuration**: Max 10,000 features, bigrams, sublinear TF
- **Optimization**: Min/max document frequency filtering
- **Storage**: Sparse matrix format for efficiency

#### **SBERT Embeddings**
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Batching**: 32 samples per batch
- **Normalization**: L2 normalized vectors

### Feature Pipeline
```
Text → Vectorizer/Encoder → Features → Storage
     ↓                    ↓           ↓
  TF-IDF/SBERT      Numpy Arrays   .npz/.npy
```

### Storage Format
```
data/features/
├── tfidf/top_50_categories/
│   ├── train_features.npz
│   └── vectorizer.pkl
└── sbert/top_50_categories/
    └── train_embeddings.npy
```

## 6. ML Models Module (`src/modeling/ml_models.py`)

### Purpose
Implements and trains traditional machine learning classifiers with standardized interfaces.

### Model Implementations

#### **Supported Algorithms**
- **LogisticRegression**: L2 regularized linear classifier
- **RandomForest**: 100 estimators ensemble
- **XGBoost**: Gradient boosting with multi-class support

#### **Training Pipeline**
1. Feature loading (TF-IDF or SBERT)
2. Hyperparameter configuration
3. Model fitting with validation monitoring
4. Prediction generation
5. Model serialization

### Output Structure
```
models/saved_models/ml_models/
├── LogisticRegression_TFIDF_top_50.pkl
├── RandomForest_SBERT_top_50.pkl
└── XGBoost_TFIDF_top_50.pkl

results/ml/top_50_categories/
├── predictions/
├── metrics/
└── confusion_matrices/
```

## 7. Deep Learning Module (`src/modeling/dl_models.py`)

### Purpose
Implements BiLSTM neural networks for sequence classification with attention mechanisms.

### Architecture Components

#### **Model Structure**
- Embedding layer (trainable or pre-trained)
- Bidirectional LSTM (256 hidden units)
- Attention mechanism
- Dropout regularization (0.5)
- Dense classification layer

#### **Training Configuration**
- Adam optimizer with learning rate scheduling
- Early stopping (patience=3)
- Batch size: 32
- Maximum epochs: 50

### Storage Outputs
```
models/saved_models/dl_models/
└── BiLSTM_[feature]_top_50.pth

results/dl/top_50_categories/
├── training_history.json
└── predictions/
```

## 8. BERT Models Module (`src/modeling/bert_models.py`)

### Purpose
Fine-tunes RoBERTa transformer models for web service classification.

### Model Variants

#### **RoBERTa Base**
- 125M parameters
- Batch size: 16 (train), 32 (eval)
- 3 training epochs

#### **RoBERTa Large**
- 355M parameters
- Batch size: 8 (train), 16 (eval)
- Gradient accumulation for memory efficiency

### Training Strategy
- Warmup steps: 500
- Weight decay: 0.01
- Learning rate: 2e-5
- Mixed precision training

## 9. Fusion Models Module (`src/modeling/fusion_models.py`)

### Purpose
Combines DeepSeek and RoBERTa embeddings through multiple fusion strategies.

### Fusion Strategies

#### **Concatenation**
Direct joining of embedding vectors

#### **Averaging**
Element-wise mean of embeddings

#### **Weighted**
Learned weights for each embedding source

#### **Gating**
Neural gate mechanism for dynamic weighting

### Architecture
```
DeepSeek → Embeddings → 
                        Fusion Layer → Classifier
RoBERTa → Embeddings →
```

## 10. Evaluation Module (`src/evaluation/evaluate.py`)

### Purpose
Comprehensive performance assessment across all models and metrics.

### Metrics Computed

#### **Classification Metrics**
- Top-1, Top-3, Top-5 accuracy
- Macro/Micro F1 scores
- Precision and recall
- Confusion matrices

#### **Comparative Analysis**
- Model ranking tables
- Performance curves
- Statistical significance tests
- Category difficulty analysis

### Output Formats
```
results/comparison/
├── all_models_leaderboard.csv
├── performance_curves.png
└── statistical_analysis.json
```

## 11. Visualization Module (`src/visualization/`)

### Purpose
Generates publication-ready charts and plots for result interpretation.

### Visualization Types

#### **Performance Charts**
- Bar plots comparing models (grouped and individual)
- Line plots for Top-K accuracy trends
- Radar charts for multi-metric comparison
- Combined Top-K performance curves

#### **Analysis Plots**
- Confusion matrix heatmaps for each model-feature combination
- Training history plots for DL/BERT/Fusion models
- Category-wise performance analysis
- Time-based metrics (training/inference)

### Complete Results Structure

#### **Model-Specific Results**
Each model type (ML, DL, BERT, Fusion) follows identical structure:

##### **ML Results Structure**
```
results/ml/
├── comparisons/
│   ├── ml_final_results.json              # Aggregated metrics in JSON
│   ├── ml_final_results.pkl               # Pickle format for Python
│   ├── ml_radar_accuracy_top_50_categories.png
│   ├── ml_radar_f1_score_top_50_categories.png
│   ├── ml_radar_precision_top_50_categories.png
│   ├── ml_radar_recall_top_50_categories.png
│   │
│   └── charts/
│       ├── ml_bar_grouped_top_50_categories.png
│       ├── ml_bar_individual_accuracy_top_50_categories.png
│       ├── ml_bar_individual_f1_score_top_50_categories.png
│       ├── ml_bar_individual_precision_top_50_categories.png
│       ├── ml_bar_individual_recall_top_50_categories.png
│       ├── ml_bar_individual_top1_accuracy_top_50_categories.png
│       ├── ml_bar_individual_top3_accuracy_top_50_categories.png
│       ├── ml_bar_individual_top5_accuracy_top_50_categories.png
│       ├── ml_line_accuracy_top_50_categories.png
│       ├── ml_line_f1_score_top_50_categories.png
│       ├── ml_line_precision_top_50_categories.png
│       ├── ml_line_recall_top_50_categories.png
│       ├── ml_line_top1_accuracy_top_50_categories.png
│       ├── ml_line_top3_accuracy_top_50_categories.png
│       ├── ml_line_top5_accuracy_top_50_categories.png
│       ├── ml_line_topk_combined_top_50_categories.png
│       └── ml_summary_statistics.csv
│
└── top_50_categories/
    ├── LogisticRegression_SBERT_top_50_categories_classification_report.csv
    ├── LogisticRegression_SBERT_top_50_categories_confusion_matrix.png
    ├── LogisticRegression_TFIDF_top_50_categories_classification_report.csv
    ├── LogisticRegression_TFIDF_top_50_categories_confusion_matrix.png
    ├── RandomForest_SBERT_top_50_categories_classification_report.csv
    ├── RandomForest_SBERT_top_50_categories_confusion_matrix.png
    ├── RandomForest_TFIDF_top_50_categories_classification_report.csv
    ├── RandomForest_TFIDF_top_50_categories_confusion_matrix.png
    ├── XGBoost_SBERT_top_50_categories_classification_report.csv
    ├── XGBoost_SBERT_top_50_categories_confusion_matrix.png
    ├── XGBoost_TFIDF_top_50_categories_classification_report.csv
    └── XGBoost_TFIDF_top_50_categories_confusion_matrix.png
```

##### **Similar Structure for DL, BERT, and Fusion**
Each model type maintains the same organizational pattern:
- `comparisons/` - Aggregate comparisons and radar charts
- `comparisons/charts/` - All visualization outputs
- `top_50_categories/` - Individual model results

#### **Visualization Files Breakdown**

##### **Radar Charts** (4 per model type)
- `[model]_radar_accuracy_top_50_categories.png` - Overall accuracy comparison
- `[model]_radar_f1_score_top_50_categories.png` - F1 score distribution
- `[model]_radar_precision_top_50_categories.png` - Precision metrics
- `[model]_radar_recall_top_50_categories.png` - Recall performance

##### **Bar Charts** (9 per model type)
- `[model]_bar_grouped_top_50_categories.png` - All models grouped comparison
- Individual metric bars for: accuracy, f1_score, precision, recall
- Individual Top-K bars for: top1, top3, top5 accuracy

##### **Line Charts** (8 per model type)
- Performance trends for: accuracy, f1_score, precision, recall
- Top-K progression: top1, top3, top5 accuracy
- `[model]_line_topk_combined_top_50_categories.png` - Combined Top-K curves

##### **Model-Specific Outputs** (2 files per model-feature combination)
- `[ModelName]_[FeatureType]_top_50_categories_classification_report.csv`
- `[ModelName]_[FeatureType]_top_50_categories_confusion_matrix.png`

#### **Overall Comparison Results**
```
results/overall/
├── Overall_BAR_[metric].png              # Cross-model bar comparisons
├── Overall_Comparison_[metric].png       # Combined visualizations
├── Overall_LINE_topk_accuracy.png        # Top-K trends across models
├── Overall_summary_leaderboard.csv       # Ranked model performance
└── pipeline_execution_*.json             # Execution metadata
```

## 12. Complete Data Organization

### Data Pipeline Structure

```
data/
├── raw/
│   └── web_services_dataset.csv          # Original dataset
│
├── analysis/
│   ├── overall/
│   │   ├── dataset_summary.json          # Global statistics
│   │   ├── category_distribution.png     # Category counts
│   │   ├── text_length_distribution.png  # Text statistics
│   │   └── word_count_distribution.png   # Word statistics
│   │
│   └── top_50_categories/
│       ├── category_statistics_top50.csv # Per-category metrics
│       ├── category_distribution_top50.png
│       ├── text_length_boxplot_top50.png # Category-wise text analysis
│       └── word_count_boxplot_top50.png
│
├── processed/
│   ├── labels_top_50_categories.yaml     # Label mappings
│   ├── top_50_categories/
│   │   └── cleaned_dataset.csv           # Preprocessed full data
│   └── splits/top_50_categories/
│       ├── split_info.json               # Split metadata
│       ├── train.csv                     # 80% training data
│       ├── val.csv                       # 10% validation data
│       └── test.csv                      # 10% test data
│
├── features/
│   ├── tfidf/top_50_categories/
│   │   ├── train_features.pkl            # Sparse TF-IDF matrices
│   │   ├── val_features.pkl
│   │   ├── test_features.pkl
│   │   ├── vectorizer.pkl                # Fitted TF-IDF vectorizer
│   │   └── feature_info.json             # Feature metadata
│   │
│   └── sbert/top_50_categories/
│       ├── train_embeddings.npy          # Dense SBERT vectors
│       ├── val_embeddings.npy
│       ├── test_embeddings.npy
│       └── embedding_config.json         # Embedding parameters
│
└── splits/top_50_categories/              # Duplicate split location
    └── [train/val/test].csv               # For backward compatibility
```

## Data Flow Summary

```
Raw Data → Analysis → Preprocessing → Feature Extraction
                                            ↓
                    ┌─────────────────────────┘
                    ↓
        ┌──────────────────────┐
        │   Model Training      │
        ├──────────────────────┤
        │ • ML Models          │
        │ • Deep Learning      │
        │ • BERT Models        │
        │ • Fusion Models      │
        └──────────┬───────────┘
                   ↓
            Evaluation → Visualization → Results
```

## Output File Naming Conventions

### Model Output Files
Pattern: `[ModelName]_[FeatureType]_top_[N]_categories_[FileType].[ext]`

Examples:
- `LogisticRegression_TFIDF_top_50_categories_classification_report.csv`
- `BiLSTM_SBERT_top_50_categories_confusion_matrix.png`
- `RoBERTa_Base_RawText_top_50_categories_classification_report.csv`
- `DeepSeek_RoBERTa_Fusion_Concat_top_50_categories_confusion_matrix.png`

### Visualization Files
Pattern: `[ModelType]_[ChartType]_[Metric]_top_[N]_categories.png`

Examples:
- `ml_bar_grouped_top_50_categories.png`
- `bert_radar_accuracy_top_50_categories.png`
- `fusion_line_topk_combined_top_50_categories.png`

### Metrics Tracked Per Model

#### **Classification Metrics**
- **Accuracy**: Overall correct predictions
- **Top-1 Accuracy**: Exact match with top prediction
- **Top-3 Accuracy**: Correct label in top 3 predictions
- **Top-5 Accuracy**: Correct label in top 5 predictions
- **Precision**: Per-class and macro/micro averaged
- **Recall**: Per-class and macro/micro averaged
- **F1-Score**: Harmonic mean of precision and recall

#### **Operational Metrics**
- **Training Time**: Time to train model (seconds)
- **Inference Time**: Average prediction time per sample (ms)
- **Model Size**: Disk space for saved model (MB)
- **Memory Usage**: Peak RAM during training (GB)

## Model Inventory Summary

### Total Models Trained: 16

#### **ML Models (6)**
- LogisticRegression + TFIDF
- LogisticRegression + SBERT
- RandomForest + TFIDF
- RandomForest + SBERT
- XGBoost + TFIDF
- XGBoost + SBERT

#### **DL Models (2)**
- BiLSTM + TFIDF
- BiLSTM + SBERT

#### **BERT Models (2)**
- RoBERTa Base
- RoBERTa Large

#### **Fusion Models (4)**
- DeepSeek-RoBERTa + Concat
- DeepSeek-RoBERTa + Average
- DeepSeek-RoBERTa + Weighted
- DeepSeek-RoBERTa + Gating

#### **DeepSeek Models (1)**
- DeepSeek 7B Base (if trained separately)

### Output Files Per Model Type

| Model Type | Comparison Files | Chart Files | Model-Specific Files | Total Files |
|------------|-----------------|-------------|---------------------|-------------|
| **ML** | 6 (2 json/pkl + 4 radar) | 17 charts + 1 CSV | 12 (6 models × 2 files) | **36 files** |
| **DL** | 6 (2 json/pkl + 4 radar) | 19 charts + 1 CSV | 8 (2 models × 4 files) | **34 files** |
| **BERT** | 6 (2 json/pkl + 4 radar) | 17 charts + 1 CSV | 6 (2 models × 3 files) | **30 files** |
| **Fusion** | 6 (2 json/pkl + 4 radar) | 17 charts + 1 CSV | 9 (4 models × 2 + history) | **33 files** |
| **Overall** | - | - | 24 comparison PNGs | **24 files** |
| | | | **Total Project Outputs:** | **157 files** |

## Storage Requirements

### Approximate Sizes
- **Raw Data**: ~500 MB
- **Processed Data**: ~300 MB per category size
- **Features**: 
  - TF-IDF: ~200 MB per split
  - SBERT: ~150 MB per split
- **Trained Models**:
  - ML: ~50-200 MB each
  - DL: ~100-300 MB each
  - BERT: ~500 MB (Base), ~1.5 GB (Large)
  - Fusion: ~600 MB each
- **Results & Visualizations**: ~500 MB total

### Total Project Size: ~15-20 GB

## Key Design Principles

1. **Modularity**: Each component is self-contained with clear interfaces
2. **Reproducibility**: Fixed seeds and versioned configurations
3. **Scalability**: Handles multiple category sizes and model types
4. **Standardization**: Consistent naming and output formats
5. **Comprehensive Logging**: Detailed tracking at every stage
6. **Error Resilience**: Graceful failure handling with recovery options

This architecture ensures systematic experimentation, reliable comparisons, and production-ready model deployment for web service classification tasks.
# Evaluation, Comparison & Utilities System
## Functional Documentation

**System Component:** Evaluation Framework & Utility Infrastructure  
**Document Type:** Technical Specification

---
## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Architecture](#2-module-architecture)
3. [ModelEvaluator (evaluate.py)](#3-modelevaluator-evaluatepy)
4. [OverallPerformanceAnalyzer (overall_comparison.py)](#4-overallperformanceanalyzer-overall_comparisonpy)
5. [Utility Functions (utils.py)](#5-utility-functions-utilspy)
6. [File Naming Standards](#6-file-naming-standards)
7. [Visualization Pipeline](#7-visualization-pipeline)
8. [Data Flow & Integration](#8-data-flow--integration)
9. [Usage Examples](#9-usage-examples)

---

## 1. System Overview

### 1.1 Purpose

This documentation covers three critical infrastructure modules that provide:

1. **Model Evaluation** (`evaluate.py`): Standardized metrics calculation, visualization generation, and performance reporting
2. **Cross-Model Comparison** (`overall_comparison.py`): Unified analysis across ML/DL/BERT/DeepSeek/Fusion models
3. **Utility Infrastructure** (`utils.py`): Common helpers for data I/O, logging, reproducibility, and naming standards

### 1.2 Design Philosophy

**Core Principles:**
- **Standardization**: Consistent naming, metrics, and outputs across all model types
- **Modularity**: Reusable components with clear interfaces
- **Extensibility**: Easy to add new models, metrics, or visualizations
- **Reproducibility**: Deterministic results with proper seeding
- **Scalability**: Handles varying category sizes and model complexities

### 1.3 Module Relationships

```
┌──────────────────────────────────────────────────────────┐
│                    Model Training                         │
│        (ML / DL / BERT / DeepSeek / Fusion)              │
└─────────────────┬────────────────────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │   utils.py        │
        │ • File I/O        │
        │ • Logging         │
        │ • Naming Std      │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  evaluate.py      │
        │ • Metrics Calc    │
        │ • Visualizations  │
        │ • Per-Model Eval  │
        └─────────┬─────────┘
                  │
                  │ (Results Files: pkl/json)
                  │
        ┌─────────▼─────────────────┐
        │ overall_comparison.py     │
        │ • Cross-Model Analysis    │
        │ • Unified Comparisons     │
        │ • Summary Reports         │
        └───────────────────────────┘
```

---

## 2. Module Architecture

### 2.1 Component Overview

| Module | Primary Class | Lines of Code | Key Responsibilities |
|--------|---------------|---------------|---------------------|
| `evaluate.py` | `ModelEvaluator` | ~763 | Metrics, visualizations, per-model evaluation |
| `overall_comparison.py` | `OverallPerformanceAnalyzer` | ~482 | Cross-model comparison, unified plots |
| `utils.py` | `FileNamingStandard` + utilities | ~400 | I/O, logging, naming, reproducibility |

### 2.2 Dependency Graph

```
config.py (centralized configuration)
    ↓
utils.py (foundational utilities)
    ↓
evaluate.py (model-specific evaluation)
    ↓
overall_comparison.py (cross-model analysis)
```

### 2.3 Key Design Patterns

**1. Standardized Naming Pattern:**
```
{ModelName}_{FeatureType}_top_{N}_categories_{OutputType}.{Extension}

Examples:
- LogisticRegression_tfidf_top_5_categories_confusion_matrix.png
- RoBERTa_Base_raw_text_top_10_categories_classification_report.csv
- DeepSeek_RoBERTa_Fusion_Concat_top_20_categories_metrics.json
```

**2. Unified Metrics Format:**
```python
{
    "model_name": str,
    "feature_type": str,
    "n_categories": int,
    "accuracy": float,
    "top1_accuracy": float,
    "top3_accuracy": float,
    "top5_accuracy": float,
    "macro_precision": float,
    "macro_recall": float,
    "macro_f1": float,
    "micro_precision": float,
    "micro_recall": float,
    "micro_f1": float,
    "training_time": float,
    "inference_time": float
}
```

**3. Result File Structure:**
```python
# Nested dictionary structure
results = {
    n_categories: {
        "model_feature": {
            # metrics dictionary
        }
    }
}
```

---

## 3. ModelEvaluator (evaluate.py)

### 3.1 Class Overview

```python
class ModelEvaluator:
    """
    Comprehensive evaluation framework for all model types
    Provides standardized metrics calculation and visualization
    """
    
    def __init__(self):
        self.final_results = {}
```

**Responsibilities:**
1. Calculate evaluation metrics (accuracy, precision, recall, F1, top-K)
2. Generate confusion matrices and classification reports
3. Create comparison plots (line, bar, radar)
4. Save evaluation results in standardized format
5. Load and process class labels

---

### 3.2 Core Methods

#### 3.2.1 Metrics Calculation

**calculate_top_k_accuracy()**
```python
def calculate_top_k_accuracy(self, y_true, y_proba, k=1):
    """
    Calculate top-k accuracy for classification
    
    Args:
        y_true: True labels (indices or one-hot)
        y_proba: Predicted probabilities [samples x classes]
        k: Number of top predictions to consider
    
    Returns:
        float: Top-k accuracy score
    
    Algorithm:
        - For k=1: argmax of probabilities
        - For k>1: Check if true label in top-k predictions
    """
```

**Supported Values:** k ∈ {1, 3, 5}

**Use Cases:**
- k=1: Standard accuracy
- k=3: Lenient evaluation for similar categories
- k=5: Very lenient, useful for large category counts

---

#### 3.2.2 Class Label Management

**load_class_labels()**
```python
def load_class_labels(self, n_categories):
    """
    Load human-readable class labels
    
    Process:
        1. Read train.csv for category size
        2. Extract 'Service Classification' column
        3. Map encoded_label → category name
        4. Return sorted list of labels
    
    Fallback:
        If loading fails, generate generic labels:
        ["Category_0", "Category_1", ..., "Category_N-1"]
    """
```

---

#### 3.2.3 Visualization Generation

**A. Confusion Matrix Heatmap**

```python
def generate_confusion_heatmap(
    self, cm, class_labels, model_name, 
    n_categories, feature_type, model_type
):
    """
    Generate confusion matrix visualization
    
    Configuration:
        - Figure Size: 12x10 inches
        - Color Map: 'Blues'
        - Annotations: True (cell counts)
        - Format: 'd' (integer)
    
    Output:
        - DPI: 300
        - Format: PNG
        - Location: results/{model_type}/top_{n}_categories/
        - Filename: {Model}_{Feature}_top_{N}_categories_confusion_matrix.png
    """
```

**Visual Elements:**
- X-axis: Predicted labels
- Y-axis: True labels
- Color intensity: Number of predictions
- Title: Model name, category count, feature type

---

**B. Classification Report CSV**

```python
def generate_classification_report_csv(
    self, y_true, y_pred, class_labels, model_name,
    n_categories, feature_type, model_type
):
    """
    Generate detailed per-class metrics CSV
    
    Columns:
        - category_name: Class label
        - precision: Per-class precision
        - recall: Per-class recall
        - f1-score: Per-class F1 score
        - support: Number of samples
        - accuracy: Per-class accuracy
    
    Additional Rows:
        - macro avg: Unweighted average
        - micro avg: Weighted average
        - weighted avg: Support-weighted average
    """
```

---

**C. Comparison Plots**

```python
def plot_results_comparison(
    self, results_file_path, charts_dir, model_type
):
    """
    Generate line plots comparing models across category sizes
    
    Plots Generated:
        1. Accuracy vs Category Size
        2. F1-Score vs Category Size
        3. Top-1 Accuracy vs Category Size
        4. Top-3 Accuracy vs Category Size
        5. Top-5 Accuracy vs Category Size
        6. Training Time vs Category Size
    
    Visual Configuration:
        - X-axis: Category sizes [5, 10, 20]
        - Y-axis: Metric value
        - Lines: One per model-feature combination
        - Markers: 'o' with line connections
        - Grid: Enabled (alpha=0.3)
        - Legend: Upper right, outside plot area
    """
```

---

**D. Radar Plots**

```python
def generate_radar_plots(self, model_type, show_plots=False):
    """
    Generate radar (spider) plots for multi-metric visualization
    
    Metrics Visualized:
        - Precision (per-category)
        - Recall (per-category)
        - F1-Score (per-category)
        - Accuracy (per-category)
    
    Configuration:
        - Polar plot with N axes (N = number of categories)
        - One line per model-feature combination
        - Filled areas with transparency (alpha=0.1)
        - Angular labels: Category names
        - Radial scale: 0 to 1
    
    Use Case:
        Visual comparison of model performance across all categories
        simultaneously, useful for identifying category-specific strengths
    """
```

**Adaptive Sizing:**
- 5-20 categories: 8x8 figure, font size 10
- 20-40 categories: 14x14 figure, font size 8
- 40+ categories: 14x14 figure, font size 6

---

### 3.3 Model Type Handling

**Supported Model Types:**
```python
model_types = {
    'ml': ['LogisticRegression', 'RandomForest', 'XGBoost'],
    'dl': ['BiLSTM'],
    'bert': ['RoBERTa_Base', 'RoBERTa_Large'],
    'deepseek': ['DeepSeek_7B_Base'],
    'fusion': ['DeepSeek_RoBERTa_Fusion_*']
}
```

**Path Resolution:**
```python
def _get_results_path(self, model_type, n_categories):
    """
    Determine correct results directory based on model type
    
    Mapping:
        ml      → results/ml_models/top_{n}_categories/
        dl      → results/dl_models/top_{n}_categories/
        bert    → results/bert_models/top_{n}_categories/
        deepseek → results/deepseek_models/top_{n}_categories/
        fusion  → results/fusion_models/top_{n}_categories/
    """
```

---

### 3.4 Metric Printing

**print_model_metrics()**
```python
def print_model_metrics(
    self, results, model_name, n_categories, 
    feature_type, training_time, model_type_label
):
    """
    Pretty-print evaluation results to console
    
    Output Format:
        =====================================
        MODEL: LogisticRegression (ML)
        FEATURE: TFIDF
        CATEGORIES: 5
        =====================================
        ✓ Accuracy:        0.8542
        ✓ Top-1 Accuracy:  0.8542
        ✓ Top-3 Accuracy:  0.9234
        ✓ Top-5 Accuracy:  0.9612
        ✓ Macro Precision: 0.8423
        ✓ Macro Recall:    0.8389
        ✓ Macro F1:        0.8401
        ✓ Micro Precision: 0.8542
        ✓ Micro Recall:    0.8542
        ✓ Micro F1:        0.8542
        ───────────────────────────────────
        Training Time:     45.23s
        Inference Time:    0.12s
        =====================================
    """
```

---

### 3.5 Result Persistence

**save_model_performance_data()**
```python
def save_model_performance_data(
    self, results, model_name, n_categories, 
    feature_type, model_type
):
    """
    Save evaluation results to JSON file
    
    Filename Pattern:
        {Model}_{Feature}_top_{N}_categories_metrics.json
    
    Content:
        - All metrics from results dictionary
        - Timestamp
        - Model configuration details
    
    Location:
        results/{model_type}/top_{n}_categories/
    """
```

---

## 4. OverallPerformanceAnalyzer (overall_comparison.py)

### 4.1 Class Overview

```python
class OverallPerformanceAnalyzer:
    """
    Cross-model performance comparison and unified analysis
    Combines ML, DL, BERT, DeepSeek, and Fusion results
    """
    
    def __init__(self):
        self.overall_dir = Path("results/overall")
        self.naming_patterns = {...}  # Standardized model names
```

**Primary Purpose:**
Aggregate and compare results across all model types to identify:
- Best performing models overall
- Best performers by model type
- Feature type effectiveness
- Category size impact
- Training efficiency

---

### 4.2 Data Loading & Normalization

#### 4.2.1 Load All Results

```python
def load_all_results(self):
    """
    Load results from all model types
    
    Result Files:
        - ml_final_results.pkl
        - dl_final_results.pkl
        - bert_final_results.pkl
        - deepseek_final_results.pkl
        - fusion_final_results.pkl
    
    Returns:
        Dictionary mapping model_type → results data
    
    Error Handling:
        - Missing files: Log warning, continue
        - Load errors: Log error, skip model type
    """
```

---

#### 4.2.2 Normalize Data Structure

```python
def normalize_data_structure(self, data, model_type):
    """
    Convert different result formats to unified structure
    
    Input Formats:
        ML/DL:
            {n_categories: [list of result dicts]}
        
        BERT/DeepSeek/Fusion:
            {n_categories: {"model_feature": result_dict}}
    
    Output Format (Unified):
        {
            n_categories: [
                {
                    'model': str,
                    'model_type': str,
                    'feature_type': str,
                    'n_categories': int,
                    'accuracy': float,
                    'precision': float,
                    'recall': float,
                    'f1_score': float,
                    'top1_accuracy': float,
                    'top3_accuracy': float,
                    'top5_accuracy': float,
                    'training_time': float,
                    'inference_time': float
                },
                ...
            ]
        }
    
    Benefits:
        - Consistent access patterns
        - Simplified plotting code
        - Easy to extend with new model types
    """
```

---

### 4.3 Visualization Generation

#### 4.3.1 Combined Line Plots

```python
def generate_combined_line_plots(self, combined_metrics):
    """
    Generate line plots showing metric trends across category sizes
    
    Plots:
        1. Overall_LINE_Accuracy.png
        2. Overall_LINE_Precision.png
        3. Overall_LINE_Recall.png
        4. Overall_LINE_F1_Score.png
        5. Overall_LINE_Top_1.png
        6. Overall_LINE_Top_3.png
        7. Overall_LINE_Top_5.png
        8. Overall_LINE_Training_Time.png
    
    Configuration:
        - X-axis: Category sizes
        - Y-axis: Metric value
        - Lines: One per model configuration
        - Colors: Grouped by model type
            • ML: Blues palette
            • DL: Greens palette
            • BERT: Oranges palette
            • DeepSeek: Purples palette
            • Fusion: Reds palette
        - Markers: Circle with connecting lines
        - Grid: Enabled
    
    Legend:
        - Position: Upper right, outside plot
        - Format: Model (Feature)
        - Font size: 10
    """
```

**Color Scheme:**
```python
model_colors = {
    'ML': ['#1f77b4', '#aec7e8', '#2ca02c'],
    'DL': ['#ff7f0e', '#ffbb78', '#d62728'],
    'BERT': ['#9467bd', '#c5b0d5', '#8c564b'],
    'DEEPSEEK': ['#e377c2', '#f7b6d2', '#7f7f7f'],
    'FUSION': ['#c49c94', '#dbdb8d', '#17becf']
}
```

---

#### 4.3.2 Combined Bar Plots

```python
def generate_combined_bar_plots(self, combined_metrics):
    """
    Generate bar charts showing average performance across all categories
    
    Plots:
        - Overall_BAR_Accuracy.png
        - Overall_BAR_Precision.png
        - Overall_BAR_Recall.png
        - Overall_BAR_F1_Score.png
        - Overall_BAR_Top_1.png
        - Overall_BAR_Top_3.png
        - Overall_BAR_Top_5.png
    
    Chart Configuration:
        - X-axis: Model names (rotated 45°)
        - Y-axis: Average metric value
        - Bars: Colored by model type
        - Width: Automatic based on number of models
        - Grid: Y-axis only (horizontal lines)
    
    Calculation:
        For each model:
            Average metric value across all category sizes
    
    Use Case:
        Quick visual comparison of which models perform best
        on average, ignoring category size effects
    """
```

---

### 4.4 Summary Generation

```python
def generate_summary_comparison(self, all_results):
    """
    Generate comprehensive summary tables and analysis
    
    Outputs:
        1. CSV File: Overall_Performance_Summary.csv
        2. Console Report: Multi-section analysis
    
    CSV Columns:
        - Categories: Number of categories
        - Model_Type: ML/DL/BERT/DeepSeek/Fusion
        - Model: Specific model name
        - Feature: Feature type used
        - Accuracy, Precision, Recall, F1-Score
        - Top-1, Top-3, Top-5
        - Training_Time, Inference_Time
    
    Console Report Sections:
        1. Best Overall Performers (by metric)
        2. Best Performer by Model Type
        3. Feature Type Effectiveness
        4. Model Coverage Summary
    """
```

**Example Console Output:**
```
================================================================================
COMPREHENSIVE MODEL PERFORMANCE ANALYSIS
================================================================================

Best Overall Performers by Metric:
  Accuracy    : DeepSeek_RoBERTa_Fusion_Gating (FUSION , Gating  ) on  5 categories = 0.9234
  F1-Score    : DeepSeek_RoBERTa_Fusion_Concat (FUSION , Concat  ) on  5 categories = 0.9156
  Top-1       : RoBERTa_Large                   (BERT   , raw_text) on  5 categories = 0.9198
  Top-3       : DeepSeek_RoBERTa_Fusion_Weighted(FUSION , Weighted) on  5 categories = 0.9712
  Top-5       : XGBoost                         (ML     , sbert   ) on  5 categories = 0.9834

Best Performer by Model Type:
  ML      : XGBoost                     (sbert   ) on  5 categories
           Top-1: 0.8756, F1: 0.8634, Training: 45.23s
  DL      : BiLSTM                      (glove   ) on  5 categories
           Top-1: 0.8823, F1: 0.8745, Training: 234.56s
  BERT    : RoBERTa_Large               (raw_text) on  5 categories
           Top-1: 0.9198, F1: 0.9124, Training: 1234.78s
  DEEPSEEK: DeepSeek_7B_Base            (raw_text) on  5 categories
           Top-1: 0.9045, F1: 0.8976, Training: 2345.67s
  FUSION  : DeepSeek_RoBERTa_Fusion_Gating (Gating  ) on  5 categories
           Top-1: 0.9234, F1: 0.9189, Training: 567.89s

Feature Type Effectiveness:
  tfidf     : Avg Top-1: 0.7234, Avg F1: 0.7156, Avg Training: 23.45s
  sbert     : Avg Top-1: 0.7645, Avg F1: 0.7567, Avg Training: 34.56s
  glove     : Avg Top-1: 0.7512, Avg F1: 0.7434, Avg Training: 145.67s
  raw_text  : Avg Top-1: 0.8512, Avg F1: 0.8456, Avg Training: 1567.89s
  Concat    : Avg Top-1: 0.8823, Avg F1: 0.8767, Avg Training: 234.56s
  Average   : Avg Top-1: 0.8734, Avg F1: 0.8678, Avg Training: 223.45s
  Weighted  : Avg Top-1: 0.8845, Avg F1: 0.8789, Avg Training: 245.67s
  Gating    : Avg Top-1: 0.8912, Avg F1: 0.8856, Avg Training: 267.89s

Model Coverage Summary:
            5   10  20
ML         6   6   6
DL         2   2   2
BERT       2   2   2
DEEPSEEK   1   1   1
FUSION     4   4   4
================================================================================
```

---

### 4.5 Model Naming Patterns

```python
naming_patterns = {
    "logistic_regression": "LogisticRegression",
    "random_forest": "RandomForest", 
    "xgboost": "XGBoost",
    "bilstm": "BiLSTM",
    "roberta_base": "RoBERTa-Base",
    "roberta_large": "RoBERTa-Large",
    "deepseek_7b_base": "DeepSeek-7B-Base",
    "deepseek_roberta_fusion": "DeepSeek-RoBERTa-Fusion",
    "deepseek_roberta_fusion_concat": "DeepSeek-RoBERTa-Fusion-Concat",
    "deepseek_roberta_fusion_average": "DeepSeek-RoBERTa-Fusion-Average",
    "deepseek_roberta_fusion_weighted": "DeepSeek-RoBERTa-Fusion-Weighted",
    "deepseek_roberta_fusion_gating": "DeepSeek-RoBERTa-Fusion-Gating"
}
```

**Purpose:**
Convert internal model identifiers to human-readable display names

---

## 5. Utility Functions (utils.py)

### 5.1 Logging Infrastructure

```python
def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    format_str: Optional[str] = None
) -> None:
    """
    Configure logging with console and file handlers
    
    Default Format:
        '%(asctime)s - %(levelname)s - %(message)s'
    
    Handlers:
        - Console: stdout, INFO level
        - File: Specified path, INFO level
    
    Usage:
        setup_logging(Path("logs/training.log"), "DEBUG")
    """
```

---

### 5.2 Data I/O

#### 5.2.1 Load Data

```python
def load_data(
    file_path: Union[str, Path],
    file_type: Optional[str] = None
) -> Any:
    """
    Universal data loader supporting multiple formats
    
    Supported Formats:
        - CSV: pandas.read_csv()
        - JSON: json.load()
        - JSONL: Line-delimited JSON
        - YAML: yaml.safe_load()
        - Pickle: pickle.load()
        - Joblib: joblib.load()
        - NumPy: np.load() (.npy, .npz)
    
    Type Inference:
        If file_type not provided, infer from extension
    
    Error Handling:
        - FileNotFoundError: File doesn't exist
        - ValueError: Unsupported file type
        - Logs error and re-raises
    
    Example:
        df = load_data("data/train.csv")
        config = load_data("config.yaml")
        model = load_data("model.pkl")
    """
```

---

#### 5.2.2 Save Data

```python
def save_data(
    data: Any,
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Universal data saver supporting multiple formats
    
    Format-Specific Options:
        CSV:
            index=False (default)
        JSON:
            indent=2 (default)
            ensure_ascii=False (default)
        YAML:
            allow_unicode=True (default)
        Pickle:
            protocol=HIGHEST_PROTOCOL (default)
        Joblib:
            compress=3 (default)
    
    Auto-Create Directories:
        Creates parent directories if they don't exist
    
    Example:
        save_data(df, "output/results.csv")
        save_data(config, "config.yaml")
        save_data(model, "model.joblib", compress=5)
    """
```

---

### 5.3 Reproducibility

```python
def ensure_reproducibility(seed: int = 42) -> None:
    """
    Set random seeds for deterministic results
    
    Seeds Set:
        1. Python random module
        2. NumPy random
        3. Environment variable PYTHONHASHSEED
        4. TensorFlow/Keras (if available)
        5. PyTorch (if available, including CUDA)
    
    Usage:
        Call at the start of training scripts:
        ensure_reproducibility(42)
    
    Note:
        Some operations (e.g., GPU convolutions) may
        still have non-deterministic behavior
    """
```

---

### 5.4 Display Utilities

#### 5.4.1 Format Metrics

```python
def format_metrics(
    metrics: Dict[str, float],
    decimal_places: int = 4
) -> Dict[str, str]:
    """
    Format metrics for clean display
    
    Example:
        Input:  {'accuracy': 0.85423, 'f1': 0.84567}
        Output: {'accuracy': '0.8542', 'f1': '0.8457'}
    
    Handles:
        - Numeric values: Format to N decimal places
        - Non-numeric: Convert to string
    """
```

---

#### 5.4.2 Section Headers

```python
def print_section_header(
    title: str,
    char: str = "=",
    width: int = 80
) -> None:
    """
    Print formatted section dividers
    
    Example Outputs:
        ================================
             Model Training
        ================================
        
        or
        
        ======= Model Training ========
        
    Usage:
        print_section_header("Results", char="-", width=60)
    """
```

---

### 5.5 Timestamp Generation

```python
def get_timestamp() -> str:
    """
    Generate current timestamp string
    
    Format: YYYYMMDD_HHMMSS
    Example: 20250110_143022
    
    Use Case:
        Append to filenames for versioning
        filename = f"model_{get_timestamp()}.pkl"
    """
```

---

## 6. File Naming Standards

### 6.1 FileNamingStandard Class

```python
class FileNamingStandard:
    """
    Centralized naming conventions using config mappings
    Ensures consistency across all model outputs
    """
```

---

### 6.2 Name Standardization

#### 6.2.1 Model Names

```python
@staticmethod
def standardize_model_name(model_name):
    """
    Convert to standard format using config mappings
    
    Mappings (from config.py):
        "logistic_regression" → "LogisticRegression"
        "random_forest" → "RandomForest"
        "xgboost" → "XGBoost"
        "bilstm" → "BiLSTM"
        "roberta_base" → "RoBERTa_Base"
        "roberta_large" → "RoBERTa_Large"
    
    Fallback:
        Generic cleaning (remove spaces, special chars)
    """
```

---

#### 6.2.2 Feature Names

```python
@staticmethod
def standardize_feature_name(feature_type):
    """
    Convert to standard format using config mappings
    
    Mappings:
        "tfidf" → "TFIDF"
        "sbert" → "SBERT"
        "glove" → "GloVe"
        "raw_text" → "RAW_TEXT"
        "concat" → "Concat"
        "average" → "Average"
    
    Fallback:
        Convert to uppercase
    """
```

---

### 6.3 Filename Generators

#### 6.3.1 Confusion Matrix

```python
@staticmethod
def generate_confusion_matrix_filename(
    model_name, feature_type, n_categories, file_format='png'
):
    """
    Pattern: {Model}_{Feature}_top_{N}_categories_confusion_matrix.{ext}
    
    Examples:
        LogisticRegression_TFIDF_top_5_categories_confusion_matrix.png
        RoBERTa_Base_RAW_TEXT_top_10_categories_confusion_matrix.png
    """
```

---

#### 6.3.2 Classification Report

```python
@staticmethod
def generate_classification_report_filename(
    model_name, feature_type, n_categories, file_format='csv'
):
    """
    Pattern: {Model}_{Feature}_top_{N}_categories_classification_report.{ext}
    
    Examples:
        RandomForest_SBERT_top_5_categories_classification_report.csv
        BiLSTM_GloVe_top_20_categories_classification_report.csv
    """
```

---

#### 6.3.3 Model Checkpoint

```python
@staticmethod
def generate_model_filename(
    model_name, feature_type, n_categories, file_format='pth'
):
    """
    Pattern: {Model}_{Feature}_top_{N}_categories_model.{ext}
    
    Extensions by Model Type:
        ML: .pkl (joblib/pickle)
        DL: .pth (PyTorch)
        BERT: .bin (transformers)
        Fusion: .pth (PyTorch)
    
    Examples:
        XGBoost_SBERT_top_5_categories_model.pkl
        BiLSTM_GloVe_top_10_categories_model.pth
    """
```

---

#### 6.3.4 Metrics File

```python
@staticmethod
def generate_metrics_filename(
    model_name, feature_type, n_categories, file_format='json'
):
    """
    Pattern: {Model}_{Feature}_top_{N}_categories_metrics.{ext}
    
    Example:
        LogisticRegression_TFIDF_top_5_categories_metrics.json
    """
```

---

#### 6.3.5 Training History

```python
@staticmethod
def generate_training_history_filename(
    model_name, n_categories, file_format='png'
):
    """
    Pattern: {Model}_training_history_top_{N}_categories.{ext}
    
    Used For:
        Training/validation loss and accuracy curves (DL/BERT)
    
    Example:
        BiLSTM_training_history_top_10_categories.png
    """
```

---

#### 6.3.6 Configuration File

```python
@staticmethod
def generate_config_filename(
    model_name, feature_type, n_categories, file_format='json'
):
    """
    Pattern: {Model}_{Feature}_top_{N}_categories_config.{ext}
    
    Stores:
        Hyperparameters, training configuration, model architecture
    
    Example:
        RoBERTa_Large_RAW_TEXT_top_20_categories_config.json
    """
```

---

## 7. Visualization Pipeline

### 7.1 Visualization Types

| Type | Purpose | Generated By | File Pattern |
|------|---------|-------------|--------------|
| Confusion Matrix | Per-model classification errors | ModelEvaluator | `{Model}_{Feature}_top_{N}_confusion_matrix.png` |
| Line Plot | Cross-model metric trends | ModelEvaluator, OverallPerformanceAnalyzer | `{Type}_LINE_{Metric}.png` |
| Bar Plot | Average performance comparison | OverallPerformanceAnalyzer | `Overall_BAR_{Metric}.png` |
| Radar Plot | Multi-metric category performance | ModelEvaluator | `{Type}_radar_{metric}_top_{N}.png` |
| Training Curve | Epoch-wise training progress | Model trainers (DL/BERT) | `{Model}_training_history_top_{N}.png` |

---

### 7.2 Plotting Configuration

**Standard Settings:**
```python
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 12,
    'title_size': 16,
    'label_size': 14,
    'legend_size': 10,
    'line_width': 2,
    'marker_size': 8,
    'grid_alpha': 0.3,
    'fill_alpha': 0.1
}
```

---

### 7.3 Color Schemes

**Model Type Colors:**
```python
TYPE_COLORS = {
    'ML': ['#1f77b4', '#aec7e8', '#2ca02c'],      # Blues
    'DL': ['#ff7f0e', '#ffbb78', '#d62728'],      # Oranges/Reds
    'BERT': ['#9467bd', '#c5b0d5', '#8c564b'],    # Purples
    'DEEPSEEK': ['#e377c2', '#f7b6d2', '#7f7f7f'], # Pinks/Grays
    'FUSION': ['#c49c94', '#dbdb8d', '#17becf']   # Browns/Cyans
}
```

**Heatmap Colors:**
```python
HEATMAP_CMAPS = {
    'confusion_matrix': 'Blues',
    'correlation': 'coolwarm',
    'attention': 'viridis'
}
```

---

### 7.4 Output Directories

```
results/
├── ml_models/
│   ├── comparisons/
│   │   └── charts/
│   │       ├── ml_LINE_*.png
│   │       └── ml_radar_*.png
│   └── top_X_categories/
│       ├── *_confusion_matrix.png
│       └── *_classification_report.csv
├── dl_models/
│   └── (same structure)
├── bert_models/
│   └── (same structure)
├── deepseek_models/
│   └── (same structure)
├── fusion_models/
│   └── (same structure)
└── overall/
    ├── Overall_LINE_*.png
    ├── Overall_BAR_*.png
    └── Overall_Performance_Summary.csv
```

---

## 8. Data Flow & Integration

### 8.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: MODEL TRAINING                                 │
├─────────────────────────────────────────────────────────┤
│ ml_models.py / dl_models.py / bert_models.py /          │
│ fusion_models.py                                        │
│                                                          │
│ For each model:                                          │
│   1. Load preprocessed data                             │
│   2. Extract features (utils.load_data)                 │
│   3. Train model                                        │
│   4. Calculate metrics (evaluator.calculate_*)          │
│   5. Generate visualizations                            │
│      (evaluator.generate_confusion_heatmap, etc.)       │
│   6. Save results (utils.save_data)                     │
│   7. Save model (standardized naming)                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Result Files    │
        │ *.pkl / *.json  │
        └────────┬────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 2: PER-MODEL-TYPE COMPARISON                      │
├─────────────────────────────────────────────────────────┤
│ Called from each training script:                       │
│   evaluator.plot_results_comparison()                   │
│   evaluator.generate_radar_plots()                      │
│                                                          │
│ Generates:                                               │
│   - ML comparison charts                                │
│   - DL comparison charts                                │
│   - BERT comparison charts                              │
│   - Fusion comparison charts                            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ PHASE 3: OVERALL CROSS-MODEL COMPARISON                 │
├─────────────────────────────────────────────────────────┤
│ overall_comparison.py                                   │
│                                                          │
│ OverallPerformanceAnalyzer:                             │
│   1. Load all result files (load_all_results)           │
│   2. Normalize data structures                          │
│   3. Combine metrics                                    │
│   4. Generate unified visualizations:                   │
│      • Line plots (all models)                          │
│      • Bar plots (average performance)                  │
│   5. Generate summary report                            │
│                                                          │
│ Outputs:                                                 │
│   - Overall_LINE_*.png                                  │
│   - Overall_BAR_*.png                                   │
│   - Overall_Performance_Summary.csv                     │
└─────────────────────────────────────────────────────────┘
```

---

### 8.2 Configuration Integration

**Config File (config.py):**
```python
# Used by all modules
RESULTS_CONFIG = {
    'ml_category_paths': {...},
    'dl_category_paths': {...},
    'bert_category_paths': {...},
    'deepseek_category_paths': {...},
    'fusion_category_paths': {...},
    'ml_comparisons_path': Path(...),
    'dl_comparisons_path': Path(...),
    'bert_comparisons_path': Path(...),
    'deepseek_comparisons_path': Path(...),
    'fusion_comparisons_path': Path(...),
    'overall_results_path': Path(...)
}

MODEL_NAME_MAPPING = {...}  # Used by FileNamingStandard
FEATURE_NAME_MAPPING = {...}  # Used by FileNamingStandard
```

**Import Pattern:**
```python
from src.config import RESULTS_CONFIG, CATEGORY_SIZES
from src.utils.utils import FileNamingStandard, load_data, save_data
from src.evaluation.evaluate import ModelEvaluator
```

---

### 8.3 Error Handling Strategy

**Principle:** Fail gracefully, log errors, continue processing

**Example Implementation:**
```python
try:
    # Attempt operation
    results = train_model(...)
    save_data(results, output_path)
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    # Use defaults or skip
except ValueError as e:
    logger.error(f"Invalid data: {e}")
    # Skip this model/category
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Continue with next item
finally:
    # Cleanup if needed
    pass
```

---

## 9. Usage Examples

### 9.1 Basic Evaluation Workflow

```python
# Initialize evaluator
from src.evaluation.evaluate import ModelEvaluator
from src.utils.utils import ensure_reproducibility

ensure_reproducibility(42)
evaluator = ModelEvaluator()

# Train and evaluate model
model = train_model(...)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Calculate metrics
top1 = evaluator.calculate_top_k_accuracy(y_test, y_proba, k=1)
top3 = evaluator.calculate_top_k_accuracy(y_test, y_proba, k=3)
top5 = evaluator.calculate_top_k_accuracy(y_test, y_proba, k=5)

# Generate visualizations
class_labels = evaluator.load_class_labels(n_categories=5)
cm = confusion_matrix(y_test, y_pred)

evaluator.generate_confusion_heatmap(
    cm, class_labels, "LogisticRegression", 5, "tfidf", "ml"
)
evaluator.generate_classification_report_csv(
    y_test, y_pred, class_labels, "LogisticRegression", 5, "tfidf", "ml"
)

# Save results
results = {
    "accuracy": accuracy_score(y_test, y_pred),
    "top1_accuracy": top1,
    "top3_accuracy": top3,
    "top5_accuracy": top5,
    # ... other metrics
}

evaluator.save_model_performance_data(
    results, "LogisticRegression", 5, "tfidf", "ml"
)
```

---

### 9.2 Generate Model Comparisons

```python
# After training all models of a type
from src.evaluation.evaluate import ModelEvaluator
from src.config import RESULTS_CONFIG

evaluator = ModelEvaluator()

# Generate line plots
results_file = RESULTS_CONFIG['ml_comparisons_path'] / 'ml_final_results.pkl'
charts_dir = RESULTS_CONFIG['ml_comparisons_path'] / 'charts'

evaluator.plot_results_comparison(results_file, charts_dir, "ml")

# Generate radar plots
evaluator.generate_radar_plots("ml", show_plots=False)
```

---

### 9.3 Overall Comparison

```python
# After all model types are trained
from src.evaluation.overall_comparison import OverallPerformanceAnalyzer

analyzer = OverallPerformanceAnalyzer()
analyzer.generate_all_comparisons()

# This will:
# 1. Load all ML/DL/BERT/DeepSeek/Fusion results
# 2. Normalize data structures
# 3. Generate combined visualizations
# 4. Generate summary report
```

---

### 9.4 Utility Usage Examples

```python
from src.utils.utils import (
    setup_logging, load_data, save_data,
    ensure_reproducibility, FileNamingStandard
)

# Setup logging
setup_logging(Path("logs/training.log"), level="INFO")

# Ensure reproducibility
ensure_reproducibility(42)

# Load data
train_df = load_data("data/train.csv")
config = load_data("config.yaml")

# Save results
save_data(results_dict, "results/metrics.json")
save_data(model, "models/model.pkl")

# Generate standardized filenames
cm_filename = FileNamingStandard.generate_confusion_matrix_filename(
    "LogisticRegression", "tfidf", 5
)
# Returns: "LogisticRegression_TFIDF_top_5_categories_confusion_matrix.png"

model_filename = FileNamingStandard.generate_model_filename(
    "BiLSTM", "glove", 10, "pth"
)
# Returns: "BiLSTM_GloVe_top_10_categories_model.pth"
```

---

### 9.5 Custom Metric Calculation

```python
from src.evaluation.evaluate import ModelEvaluator
import numpy as np

evaluator = ModelEvaluator()

# Custom top-K accuracy
y_true = np.array([0, 1, 2, 3, 4])
y_proba = np.array([
    [0.7, 0.2, 0.05, 0.03, 0.02],  # Pred: 0 (correct)
    [0.3, 0.4, 0.2, 0.05, 0.05],    # Pred: 1 (correct)
    [0.1, 0.3, 0.4, 0.1, 0.1],      # Pred: 2 (correct)
    [0.2, 0.2, 0.2, 0.3, 0.1],      # Pred: 3 (correct)
    [0.15, 0.15, 0.15, 0.15, 0.4]   # Pred: 4 (correct)
])

top1 = evaluator.calculate_top_k_accuracy(y_true, y_proba, k=1)
# Result: 1.0 (all predictions correct)

top3 = evaluator.calculate_top_k_accuracy(y_true, y_proba, k=3)
# Result: 1.0 (all true labels in top-3)
```

---

## 10. Best Practices

### 10.1 File Organization

**Always use standardized naming:**
```python
# Good
filename = FileNamingStandard.generate_model_filename(
    model_name, feature_type, n_categories, extension
)

# Bad
filename = f"{model_name}_{feature_type}_model.pkl"
```

---

### 10.2 Error Handling

**Always wrap I/O operations:**
```python
try:
    data = load_data(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    # Handle gracefully
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

---

### 10.3 Logging

**Use appropriate log levels:**
```python
logger.debug("Detailed debug information")
logger.info("Normal operation information")
logger.warning("Warning: potential issue")
logger.error("Error occurred, operation failed")
logger.critical("Critical error, system unstable")
```

---

### 10.4 Reproducibility

**Always set seeds at script start:**
```python
if __name__ == "__main__":
    ensure_reproducibility(42)
    main()
```

---

### 10.5 Configuration Management

**Use centralized configuration:**
```python
# Good
from src.config import RESULTS_CONFIG
output_dir = RESULTS_CONFIG['ml_comparisons_path']

# Bad
output_dir = Path("results/ml_models/comparisons")
```

---

## 11. Troubleshooting

### 11.1 Common Issues

**Issue: Inconsistent file naming**
- **Cause:** Manual filename construction
- **Solution:** Always use `FileNamingStandard` methods

**Issue: Missing result files**
- **Cause:** Training didn't complete or failed
- **Solution:** Check logs, ensure training completed successfully

**Issue: Visualization errors**
- **Cause:** Mismatched data dimensions, missing data
- **Solution:** Validate data before plotting, use try-except blocks

**Issue: Memory errors during visualization**
- **Cause:** Too many models, large category counts
- **Solution:** Generate plots separately by model type, reduce DPI

---

### 11.2 Debugging Tips

**Enable debug logging:**
```python
setup_logging(Path("debug.log"), level="DEBUG")
```

**Validate data structure:**
```python
print(f"Results keys: {results.keys()}")
print(f"Metrics: {results[5].keys()}")
print(f"Values: {results[5]['LogisticRegression_tfidf']}")
```

**Check file existence:**
```python
result_file = Path("results/ml_final_results.pkl")
print(f"File exists: {result_file.exists()}")
if result_file.exists():
    print(f"File size: {result_file.stat().st_size} bytes")
```

---

## 12. Performance Considerations

### 12.1 Optimization Tips

**Data Loading:**
- Cache loaded data when processing multiple category sizes
- Use joblib for large NumPy arrays (faster than pickle)

**Visualization:**
- Generate plots in batch, close figures immediately
- Use lower DPI for preview, high DPI for final outputs
- Consider parallel generation for independent plots

**Memory Management:**
- Delete large objects after use
- Use generators for large datasets
- Monitor memory usage in long-running scripts

---

### 12.2 Benchmarks

**Typical Operation Times:**
```
Load result file (10MB pkl):      ~0.5s
Generate confusion matrix:        ~2s
Generate classification report:   ~1s
Generate line plot (5 models):    ~3s
Generate radar plot (20 cats):    ~4s
Generate bar plot:                ~2s
Overall comparison (full):        ~30-60s
```

---


# Web Service Classification System: Functional Documentation

## Document Overview

**System:** Multi-Model Web Service Classification Pipeline  
**Document Type:** Functional Specification

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Machine Learning Models (ML)](#2-machine-learning-models-ml)
3. [Deep Learning Models (DL)](#3-deep-learning-models-dl)
4. [BERT Models](#4-bert-models)
5. [Fusion Models](#5-fusion-models)
6. [Feature Extraction Pipeline](#6-feature-extraction-pipeline)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Configuration Management](#8-configuration-management)
9. [Workflow & Integration](#9-workflow--integration)

---

## 1. System Architecture Overview

### 1.1 Purpose

The system implements a comprehensive multi-model classification pipeline for web service categorization, supporting:

- **Traditional Machine Learning**: LogisticRegression, RandomForest, XGBoost
- **Deep Learning**: BiLSTM with attention mechanisms
- **Transformer Models**: RoBERTa variants (base, large)
- **Fusion Models**: DeepSeek + RoBERTa embeddings with multiple fusion strategies

### 1.2 Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Data Layer                         │
│              (Web Service Descriptions/Text)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│ Preprocessing  │       │ Feature        │
│ Pipeline       │───────│ Extraction     │
└───────┬────────┘       └───────┬────────┘
        │                        │
        └────────┬───────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐  ┌────▼────┐  ┌───▼────┐
│  ML   │  │   DL    │  │  BERT  │
│Models │  │ Models  │  │ Models │
└───┬───┘  └────┬────┘  └───┬────┘
    │           │           │
    └─────┬─────┴─────┬─────┘
          │           │
     ┌────▼────┐ ┌───▼─────┐
     │ Fusion  │ │Evaluate │
     │ Models  │ │& Report │
     └─────────┘ └─────────┘
```

### 1.3 Category Sizes

The system supports multi-scale classification:

- **5 categories**: Simplified classification
- **10 categories**: Moderate granularity
- **20 categories**: High granularity
- **Full dataset**: Complete classification space

---

## 2. Machine Learning Models (ML)

### 2.1 Model Architecture

#### 2.1.1 Logistic Regression

**Purpose:** Baseline linear classifier for fast inference

**Configuration:**
```python
{
    "max_iter": 1000,
    "solver": "lbfgs",
    "multi_class": "multinomial",
    "random_state": 42
}
```

**Features:**
- Linear decision boundaries
- Probabilistic outputs
- Fast training and inference
- Good interpretability

**Use Cases:**
- Baseline performance benchmarking
- Real-time classification scenarios
- Feature importance analysis

---

#### 2.1.2 Random Forest

**Purpose:** Ensemble method for robust classification

**Configuration:**
```python
{
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1
}
```

**Features:**
- Tree-based ensemble
- Handles non-linear relationships
- Built-in feature importance
- Resistant to overfitting

**Use Cases:**
- Complex pattern recognition
- Feature interaction modeling
- Robust predictions with noisy data

---

#### 2.1.3 XGBoost

**Purpose:** Gradient boosting for high-performance classification

**Configuration:**
```python
{
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "use_label_encoder": False
}
```

**Features:**
- Advanced gradient boosting
- Regularization capabilities
- GPU acceleration support
- State-of-the-art performance

**Use Cases:**
- Maximum accuracy requirements
- Large-scale classification
- Competition-grade performance

---

### 2.2 Feature Types

#### 2.2.1 TF-IDF Features

**Generation Process:**
```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
```

**Characteristics:**
- Sparse vector representation
- N-gram support (unigrams + bigrams)
- Document frequency filtering
- Fast computation

**Output Dimensions:** 10,000 features

---

#### 2.2.2 SBERT Features

**Generation Process:**
```python
SentenceTransformer("all-MiniLM-L6-v2")
```

**Characteristics:**
- Dense semantic embeddings
- Contextual understanding
- Fixed-size vectors
- Pre-trained representations

**Output Dimensions:** 384 features

---

### 2.3 ML Training Pipeline

#### Workflow

```python
def train_model_on_category(n_categories, feature_type):
    # 1. Load preprocessed data
    train_df = load_data(f"top_{n_categories}_categories/train.csv")
    test_df = load_data(f"top_{n_categories}_categories/test.csv")
    
    # 2. Extract features
    if feature_type == "tfidf":
        X_train = tfidf_transform(train_df["cleaned_text"])
        X_test = tfidf_transform(test_df["cleaned_text"])
    else:
        X_train = load_sbert_features(n_categories, "train")
        X_test = load_sbert_features(n_categories, "test")
    
    # 3. Train model
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # 5. Save model and results
    save_model(model, model_path)
    save_results(results, results_path)
    
    return results
```

---

### 2.4 ML Evaluation Metrics

**Core Metrics:**
- **Accuracy**: Overall classification accuracy
- **Top-1/3/5 Accuracy**: Top-k prediction accuracy
- **Precision/Recall/F1**: Per-class and averaged metrics
- **Macro Averaging**: Unweighted class averaging
- **Micro Averaging**: Weighted by support

**Performance Outputs:**
- Confusion matrix heatmaps
- Classification reports (CSV)
- Training/inference time logs
- Model performance JSON

---

## 3. Deep Learning Models (DL)

### 3.1 BiLSTM Architecture

#### 3.1.1 Model Structure

```
Input (Text Sequences)
    ↓
Embedding Layer (trainable/pre-trained)
    ↓
Bidirectional LSTM
    ├── Forward LSTM
    └── Backward LSTM
    ↓
Attention Mechanism (optional)
    ↓
Fully Connected Layers
    ↓
Dropout (regularization)
    ↓
Output Layer (softmax)
```

**Configuration:**
```python
{
    "embedding_dim": 300,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "attention": True
}
```

---

#### 3.1.2 Component Details

**Embedding Layer:**
- **Type**: Word2Vec, GloVe, or trainable
- **Dimension**: 300
- **Vocabulary Size**: Dynamic based on dataset
- **Padding**: Post-padding with zeros
- **Truncation**: Post-truncation at max_length

**BiLSTM Layer:**
- **Hidden Units**: 256 per direction (512 total)
- **Layers**: 2 stacked BiLSTM layers
- **Dropout**: 0.3 between layers
- **Output**: Concatenated forward/backward states

**Attention Mechanism:**
```python
class AttentionLayer(nn.Module):
    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim*2]
        attention_weights = softmax(tanh(W @ lstm_output))
        context_vector = sum(attention_weights * lstm_output)
        return context_vector
```

**Classification Head:**
- Fully connected layer: hidden_dim*2 → num_classes
- Dropout: 0.3
- Activation: Softmax for multi-class

---

### 3.2 Training Configuration

**Hyperparameters:**
```python
{
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "early_stopping": {
        "patience": 5,
        "min_delta": 0.001
    }
}
```

**Data Processing:**
- **Tokenization**: Word-level or subword
- **Sequence Length**: Max 512 tokens
- **Padding Strategy**: Post-padding
- **Batch Processing**: Dynamic batching with DataLoader

---

### 3.3 DL Training Pipeline

```python
def train_dl_model(n_categories):
    # 1. Load and prepare data
    train_loader = create_dataloader(train_data, batch_size=32)
    val_loader = create_dataloader(val_data, batch_size=32)
    
    # 2. Initialize model
    model = BiLSTMClassifier(vocab_size, embedding_dim, 
                             hidden_dim, num_classes)
    
    # 3. Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc = validate(model, val_loader)
        
        # Early stopping check
        if early_stopping(val_loss):
            break
    
    # 4. Evaluation
    test_results = evaluate_model(model, test_loader)
    
    # 5. Save model
    save_checkpoint(model, model_path)
    
    return test_results
```

---

### 3.4 DL Evaluation

**Metrics:**
- Training/validation loss curves
- Epoch-wise accuracy progression
- Confusion matrices
- Per-class precision/recall/F1
- Inference time per sample

**Visualizations:**
- Training curves (loss & accuracy)
- Attention weight heatmaps
- Confusion matrix heatmaps
- ROC curves (one-vs-rest)

---

## 4. BERT Models

### 4.1 RoBERTa Architecture

#### 4.1.1 Model Variants

**RoBERTa-Base:**
```python
{
    "model_name": "roberta-base",
    "hidden_size": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 514
}
```

**RoBERTa-Large:**
```python
{
    "model_name": "roberta-large",
    "hidden_size": 1024,
    "num_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "max_position_embeddings": 514
}
```

---

#### 4.1.2 Fine-tuning Architecture

```
Input Text
    ↓
RoBERTa Tokenizer
    ↓
RoBERTa Encoder (12/24 layers)
    ├── Multi-Head Self-Attention
    ├── Layer Normalization
    └── Feed-Forward Networks
    ↓
[CLS] Token Representation
    ↓
Classification Head
    ├── Linear Layer
    ├── Dropout (0.1)
    └── Softmax
    ↓
Class Predictions
```

---

### 4.2 Training Configuration

**Fine-tuning Parameters:**
```python
{
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 5,
    "max_length": 512,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "gradient_accumulation_steps": 2
}
```

**Training Strategy:**
- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: Linear warmup + decay
- **Gradient Clipping**: Max norm 1.0
- **Mixed Precision**: FP16 training (optional)

---

### 4.3 BERT Training Pipeline

```python
def train_bert_model(n_categories, model_type="base"):
    # 1. Load pretrained model
    tokenizer = AutoTokenizer.from_pretrained(f"roberta-{model_type}")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"roberta-{model_type}",
        num_labels=n_categories
    )
    
    # 2. Prepare datasets
    train_dataset = tokenize_dataset(train_texts, train_labels)
    val_dataset = tokenize_dataset(val_texts, val_labels)
    
    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=f"./models/roberta_{model_type}_{n_categories}",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    # 4. Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # 5. Train
    trainer.train()
    
    # 6. Evaluate
    results = trainer.evaluate(test_dataset)
    
    # 7. Save model
    trainer.save_model(model_path)
    
    return results
```

---

### 4.4 BERT Evaluation

**Metrics:**
- Validation loss per epoch
- Top-1/3/5 accuracy
- Per-class F1 scores
- Macro/Micro averaged metrics
- Inference time analysis

**Model Outputs:**
- Fine-tuned model checkpoint
- Tokenizer vocabulary
- Training logs
- Confusion matrices
- Classification reports

---

## 5. Fusion Models

### 5.1 Architecture Overview

Fusion models combine DeepSeek and RoBERTa embeddings to leverage complementary strengths:

- **DeepSeek**: Domain-specific semantic understanding
- **RoBERTa**: General language representation
- **Fusion**: Combined decision-making

---

### 5.2 Fusion Strategies

#### 5.2.1 Concatenation Fusion

```python
def concatenation_fusion(deepseek_embed, roberta_embed):
    """
    Simple concatenation of embeddings
    Output: [deepseek; roberta]
    """
    fused = torch.cat([deepseek_embed, roberta_embed], dim=-1)
    return fused
```

**Characteristics:**
- Preserves all information
- Doubled embedding dimension
- No trainable parameters for fusion
- Fast computation

**Use Case:** Baseline fusion approach

---

#### 5.2.2 Averaging Fusion

```python
def averaging_fusion(deepseek_embed, roberta_embed):
    """
    Element-wise average of embeddings
    Output: (deepseek + roberta) / 2
    """
    # Ensure same dimensions
    fused = (deepseek_embed + roberta_embed) / 2
    return fused
```

**Characteristics:**
- Maintains original dimension
- Equal weight to both embeddings
- No trainable parameters
- Dimensionality reduction

**Use Case:** Memory-efficient fusion

---

#### 5.2.3 Weighted Fusion

```python
class WeightedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, deepseek_embed, roberta_embed):
        """
        Learnable weighted combination
        Output: alpha * deepseek + (1 - alpha) * roberta
        """
        fused = self.alpha * deepseek_embed + \
                (1 - self.alpha) * roberta_embed
        return fused
```

**Characteristics:**
- Learnable weight parameter
- Adaptive importance balancing
- Single trainable parameter
- Maintains original dimension

**Use Case:** Adaptive model selection

---

#### 5.2.4 Gating Fusion

```python
class GatingFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, deepseek_embed, roberta_embed):
        """
        Attention-based gating mechanism
        Output: gate * deepseek + (1 - gate) * roberta
        """
        combined = torch.cat([deepseek_embed, roberta_embed], dim=-1)
        gate_weights = self.gate(combined)
        fused = gate_weights * deepseek_embed + \
                (1 - gate_weights) * roberta_embed
        return fused
```

**Characteristics:**
- Context-dependent fusion
- Input-specific weighting
- Trainable gating network
- Most expressive fusion

**Use Case:** Maximum adaptability

---

### 5.3 Fusion Model Pipeline

```python
def train_fusion_model(n_categories, fusion_type="concatenation"):
    # 1. Load pre-computed embeddings
    deepseek_train = load_embeddings("deepseek", n_categories, "train")
    roberta_train = load_embeddings("roberta", n_categories, "train")
    
    # 2. Initialize fusion layer
    fusion_layer = create_fusion_layer(fusion_type, embed_dim)
    
    # 3. Create classification model
    model = FusionClassifier(
        fusion_layer=fusion_layer,
        input_dim=get_fused_dim(fusion_type),
        num_classes=n_categories
    )
    
    # 4. Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            ds_embed, rb_embed, labels = batch
            
            # Fuse embeddings
            fused = fusion_layer(ds_embed, rb_embed)
            
            # Classify
            logits = model(fused)
            loss = criterion(logits, labels)
            
            # Backprop
            loss.backward()
            optimizer.step()
    
    # 5. Evaluate
    results = evaluate_fusion_model(model, test_loader)
    
    # 6. Save
    save_fusion_model(model, model_path)
    
    return results
```

---

### 5.4 Fusion Evaluation

**Comparative Metrics:**
- Fusion vs individual models
- Fusion strategy comparison
- Computational overhead analysis
- Memory usage profiling

**Visualization:**
- Fusion weight distributions
- Gate activation patterns
- Performance comparison charts
- Embedding space visualizations (t-SNE)

---

## 6. Feature Extraction Pipeline

### 6.1 Feature Extractor Class

```python
class FeatureExtractor:
    """Unified feature extraction for all models"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.sbert_model = None
        self.glove_embeddings = None
    
    # TF-IDF features for ML
    def extract_tfidf_features(self, texts, n_categories)
    
    # SBERT features for ML
    def extract_sbert_features(self, texts, n_categories)
    
    # Word embeddings for DL
    def load_pretrained_embeddings(self, embedding_type)
    
    # DeepSeek embeddings for Fusion
    def extract_deepseek_embeddings(self, texts, n_categories)
    
    # RoBERTa embeddings for Fusion
    def extract_roberta_embeddings(self, texts, n_categories)
```

---

### 6.2 Feature Storage

**Directory Structure:**
```
features/
├── top_5_categories/
│   ├── tfidf_vectorizer.pkl
│   ├── sbert_train_features.npy
│   ├── sbert_test_features.npy
│   ├── deepseek_train_embeddings.npy
│   └── roberta_train_embeddings.npy
├── top_10_categories/
│   └── ...
└── top_20_categories/
    └── ...
```

---

## 7. Evaluation Framework

### 7.1 ModelEvaluator Class

```python
class ModelEvaluator:
    """Standardized evaluation across all model types"""
    
    # Core metrics calculation
    def calculate_metrics(self, y_true, y_pred, y_proba)
    
    # Top-K accuracy
    def calculate_top_k_accuracy(self, y_true, y_proba, k)
    
    # Visualization generation
    def generate_confusion_heatmap(self, cm, labels, model_name, 
                                   n_categories, feature_type, model_type)
    
    def generate_classification_report_csv(self, y_true, y_pred, 
                                          labels, model_name, 
                                          n_categories, feature_type, 
                                          model_type)
    
    # Results comparison
    def plot_results_comparison(self, results_file, output_dir, 
                               model_type)
    
    # Radar plots
    def generate_radar_plots(self, model_type, show_plots=False)
```

---

### 7.2 Standardized Outputs

**File Naming Convention:**
```python
class FileNamingStandard:
    @staticmethod
    def generate_model_filename(model_name, feature_type, 
                               n_categories, extension):
        """
        Examples:
        - LogisticRegression_tfidf_5cat.pkl
        - BiLSTM_glove_10cat.pth
        - RoBERTa_base_20cat.bin
        - Fusion_concat_5cat.pth
        """
        clean_name = standardize_model_name(model_name)
        return f"{clean_name}_{feature_type}_{n_categories}cat.{extension}"
```

---

### 7.3 Visualization Outputs

**Generated Charts:**
1. **Confusion Matrices**: Per-model heatmaps
2. **Comparison Plots**: Cross-model performance
3. **Radar Plots**: Multi-metric visualization
4. **Training Curves**: Loss/accuracy over epochs (DL/BERT)
5. **Bar Plots**: Category-size performance comparison

**Storage:**
```
results/
├── ml_models/
│   ├── comparisons/
│   │   └── charts/
│   └── top_X_categories/
│       ├── confusion_matrices/
│       └── classification_reports/
├── dl_models/
│   └── ...
├── bert_models/
│   └── ...
└── fusion_models/
    └── ...
```

---

## 8. Configuration Management

### 8.1 Central Configuration (config.py)

```python
# Model configurations
ML_CONFIG = {
    "models": ["LogisticRegression", "RandomForest", "XGBoost"],
    "logistic_regression": {...},
    "random_forest": {...},
    "xgboost": {...}
}

DL_CONFIG = {
    "embedding_dim": 300,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 20
}

BERT_CONFIG = {
    "models": ["roberta-base", "roberta-large"],
    "learning_rate": 2e-5,
    "batch_size": 16,
    "max_length": 512,
    "epochs": 5
}

FUSION_CONFIG = {
    "strategies": ["concatenation", "averaging", "weighted", "gating"],
    "deepseek_model": "deepseek-ai/deepseek-coder-1.3b",
    "roberta_model": "roberta-large"
}

# Category sizes
CATEGORY_SIZES = [5, 10, 20]

# Path configurations
SAVED_MODELS_CONFIG = {
    "ml_models_path": Path("models/ml_models"),
    "dl_models_path": Path("models/dl_models"),
    "bert_models_path": Path("models/bert_models"),
    "fusion_models_path": Path("models/fusion_models")
}

RESULTS_CONFIG = {
    "ml_comparisons_path": Path("results/ml_models/comparisons"),
    "dl_comparisons_path": Path("results/dl_models/comparisons"),
    "bert_comparisons_path": Path("results/bert_models/comparisons"),
    "fusion_comparisons_path": Path("results/fusion_models/comparisons")
}
```

---

### 8.2 Configuration Usage

All model files import and use centralized configuration:

```python
from src.config import (
    ML_CONFIG, DL_CONFIG, BERT_CONFIG, FUSION_CONFIG,
    CATEGORY_SIZES, SAVED_MODELS_CONFIG, RESULTS_CONFIG
)
```

---

## 9. Workflow & Integration

### 9.1 End-to-End Pipeline

```
1. DATA PREPARATION
   ├── Load raw dataset
   ├── Clean and preprocess text
   ├── Split by category sizes (5, 10, 20)
   └── Create train/val/test splits

2. FEATURE EXTRACTION
   ├── TF-IDF features (ML)
   ├── SBERT features (ML)
   ├── Word embeddings (DL)
   ├── DeepSeek embeddings (Fusion)
   └── RoBERTa embeddings (Fusion)

3. MODEL TRAINING
   ├── Train ML models (all category sizes, both features)
   ├── Train DL models (all category sizes)
   ├── Fine-tune BERT models (all category sizes, both variants)
   └── Train Fusion models (all strategies, all category sizes)

4. EVALUATION
   ├── Generate metrics for each model
   ├── Create visualizations
   ├── Save results in standardized format
   └── Generate comparison reports

5. ANALYSIS
   ├── Cross-model comparison
   ├── Feature type impact analysis
   ├── Category size scaling analysis
   └── Fusion strategy effectiveness
```

---

### 9.2 Execution Commands

**Train ML Models:**
```bash
python ml_models.py
```

**Train DL Models:**
```bash
python dl_models.py
```

**Train BERT Models:**
```bash
python bert_models.py
```

**Train Fusion Models:**
```bash
python fusion_models.py
```

**Generate Overall Comparison:**
```bash
python overall_performance_analyzer.py
```

---

### 9.3 Result Files

**Pickle Files (for programmatic access):**
- `ml_final_results.pkl`
- `dl_final_results.pkl`
- `bert_final_results.pkl`
- `fusion_final_results.pkl`

**JSON Files (for inspection):**
- `ml_final_results.json`
- `dl_final_results.json`
- `bert_final_results.json`
- `fusion_final_results.json`

**Visualization Files:**
- Confusion matrices (PNG)
- Comparison charts (PNG)
- Radar plots (PNG)
- Training curves (PNG)

---

## 10. Key Design Principles

### 10.1 Modularity

Each model type (ML, DL, BERT, Fusion) is implemented in a separate module with consistent interfaces.

### 10.2 Standardization

- Consistent file naming across all models
- Unified evaluation metrics
- Standardized result storage format
- Common visualization styles

### 10.3 Scalability

- Supports multiple category sizes
- Configurable hyperparameters
- Parallel training support
- Efficient feature caching

### 10.4 Reproducibility

- Fixed random seeds
- Saved model checkpoints
- Logged hyperparameters
- Version-controlled configurations

---

## 11. Performance Benchmarks

### 11.1 Expected Accuracy Ranges

**5 Categories:**
- ML Models: 75-85%
- DL Models: 80-88%
- BERT Models: 85-92%
- Fusion Models: 87-93%

**10 Categories:**
- ML Models: 65-75%
- DL Models: 70-80%
- BERT Models: 78-88%
- Fusion Models: 80-89%

**20 Categories:**
- ML Models: 55-65%
- DL Models: 60-72%
- BERT Models: 70-82%
- Fusion Models: 72-84%

---

### 11.2 Training Times (Approximate)

**Per Category Size:**
- ML Models: 5-15 minutes
- DL Models: 30-60 minutes
- BERT Models: 2-4 hours
- Fusion Models: 15-30 minutes

---

### 11.3 Inference Times (Per Sample)

- ML Models: <1ms
- DL Models: 2-5ms
- BERT Models: 10-20ms
- Fusion Models: 15-25ms

---

## 12. Troubleshooting Guide

### 12.1 Common Issues

**Out of Memory (OOM):**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Process category sizes sequentially

**Slow Training:**
- Enable GPU acceleration
- Use pre-computed features
- Reduce model complexity
- Implement early stopping

**Poor Performance:**
- Check data imbalance
- Verify preprocessing quality
- Tune hyperparameters
- Increase training epochs

---

## 13. Future Enhancements

### 13.1 Planned Features

1. **Multi-modal Fusion**: Incorporate image/metadata
2. **Ensemble Methods**: Combine all model predictions
3. **Active Learning**: Iterative model improvement
4. **Explainability**: LIME/SHAP integration
5. **API Deployment**: Real-time inference service

### 13.2 Optimization Opportunities

- Quantization for faster inference
- Knowledge distillation
- Neural architecture search
- Automated hyperparameter tuning

---


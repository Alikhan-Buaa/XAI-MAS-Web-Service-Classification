
# 📘 Functional Documentation

### Web Services Classification Project

*Configuration-driven Modular Data Pipeline (ML + DL + LLM)*

---

##  1. Overview

The **Web Services Classification Project** is designed as a unified, configuration-driven pipeline for analyzing, preprocessing, and extracting features from a dataset of web service descriptions.

The goal is to build a **consistent, scalable foundation** that supports multiple model families:

* **Machine Learning (ML)** – Logistic Regression, Random Forest, XGBoost
* **Deep Learning (DL)** – BiLSTM
* **Transformer-based Models** – RoBERTa (BERT), DeepSeek
* **Fusion Models** – Combined representation of RoBERTa + DeepSeek

The pipeline ensures reproducibility, naming consistency, and organized results across all these model types.

---

##  2. Configuration File (`config.py`)

### Purpose

Defines **all global settings**—directory paths, data structure, preprocessing behavior, model configurations, and naming conventions. This guarantees that all scripts (analysis, preprocessing, feature extraction, training, evaluation) reference consistent values.

---

###  Directory Structure

| Section        | Description                                    | Example Path                        |
| -------------- | ---------------------------------------------- | ----------------------------------- |
| `DATA_PATH`    | Stores raw, processed, split, and feature data | `/data/processed/top_50_categories` |
| `MODELS_PATH`  | Trained and saved models                       | `/models/saved_models/bert_models/` |
| `RESULTS_PATH` | Evaluation and metric results                  | `/results/ml/top_50_categories`     |
| `LOGS_PATH`    | Log files for pipeline stages                  | `/logs/training.log`                |

The helper function `create_all_directories()` ensures all folders are created automatically.

---

###  Data Configuration

| Key             | Purpose                                      |
| --------------- | -------------------------------------------- |
| `raw_data_path` | Original dataset file path                   |
| `text_column`   | Input text column name (Service Description) |
| `target_column` | Output label column (Service Classification) |

Also defines uniform folder naming for top-N categories, ensuring consistency like:
`/data/processed/top_50_categories`, `/features/sbert/top_50_categories`.

---

###  Preprocessing Config

Defines rules for text cleaning:

| Setting                                                                                  | Description                  |
| ---------------------------------------------------------------------------------------- | ---------------------------- |
| `remove_stopwords`, `remove_numbers`, `lemmatization`, `lowercase`, `remove_punctuation` | Text normalization pipeline  |
| `min_word_length`, `max_word_length`                                                     | Word length filters          |
| `custom_stopwords`                                                                       | User-defined words to remove |
| `remove_urls`, `remove_emails`                                                           | Advanced cleaning flags      |

---

###  Feature Extraction Config

| Type       | Description                                               |
| ---------- | --------------------------------------------------------- |
| **TF-IDF** | Max 10K features, bigram range (1,2), 0.95 max df         |
| **SBERT**  | Model: `all-MiniLM-L6-v2`, batch 32, normalize embeddings |

Also includes directories for saving plots and statistics (`feature_plots`, `feature_stats`).

---

###  Model Configurations

Each model type is fully defined:

| Model              | Key Highlights                                                                       |
| ------------------ | ------------------------------------------------------------------------------------ |
| **ML**             | Logistic Regression, RandomForest, XGBoost; with reproducibility (`random_state=42`) |
| **DL**             | BiLSTM with dropout, learning rate, early stopping, checkpointing                    |
| **BERT / RoBERTa** | Train/eval batch sizes, LR 2e-5, epochs 3, warmup steps 500                          |
| **DeepSeek**       | Fine-tuning config with LoRA, 4-bit quantization, gradient checkpointing             |
| **Fusion**         | Hybrid RoBERTa + DeepSeek embedding fusion (concat, average, weighted, gating)       |

---

###  File Naming Standards

Ensures all components use standardized names:

* **Model mapping:** `roberta-base` → `RoBERTa_Base`
* **Feature mapping:** `sbert` → `SBERT`, `concat` → `Concat`

This allows automated merging of results, cross-model comparisons, and unified dashboard generation.

---

##  3. Data Analysis (`data_analysis.py`)

### Purpose

Provides **comprehensive exploratory data analysis (EDA)** for both the full dataset and filtered Top-N category subsets (e.g., top 10, top 50). It produces visual and statistical summaries to understand the distribution, text properties, and category balance.

---

### Main Components

#### **Class: `DataAnalyzer`**

Handles all EDA operations, directory setup, and report generation.

| Method                    | Functionality                                                |
| ------------------------- | ------------------------------------------------------------ |
| `setup_directories()`     | Creates `/analysis/overall` and `/analysis/top_n_categories` |
| `load_data()`             | Loads dataset from configured CSV path                       |
| `analyze_overall()`       | Computes and visualizes global dataset statistics            |
| `analyze_topN()`          | Performs top-N category-specific exploration                 |
| `run_complete_analysis()` | Runs full EDA pipeline end-to-end                            |

---

###  Key Outputs

#### 1. **Overall Analysis**

* Summarizes dataset size, number of categories, missing values, duplicates.
* Generates:

  * `dataset_summary.json`
  * Distribution plots for:

    * Text length
    * Word count
    * Word length
    * Category frequency

#### 2. **Top-N Analysis**

* Filters dataset to most frequent `N` categories (e.g., 50).
* Computes per-category:

  * Avg/min/max/median text length and word count
  * Top 10 most common words
* Outputs:

  * CSV & JSON stats (`category_statistics_top50.json`)
  * Boxplots, bar charts (`text_length_boxplot_top50.png`, etc.)
  * Grouped comparison plots (avg text length vs. avg word count)

---

###  Example Insights

| Metric                  | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `avg_text_length`       | Average character length of service descriptions |
| `avg_word_count`        | Average number of tokens per entry               |
| `Top_10_Words`          | Most frequent words per category                 |
| `Category_Distribution` | Detects class imbalance                          |

---

##  4. Data Preprocessing (`data_preprocessing.py`)

### Purpose

Performs text cleaning, category filtering, label encoding, and dataset splitting.
Ensures uniform and clean data ready for ML/DL/LLM pipelines.

---

### Main Components

#### **Class: `DataPreprocessor`**

Encapsulates cleaning, filtering, and splitting logic.

| Method                          | Functionality                                             |
| ------------------------------- | --------------------------------------------------------- |
| `load_data()`                   | Reads the dataset and verifies schema                     |
| `clean_text()`                  | Cleans text using regex, tokenization, lemmatization      |
| `get_top_categories()`          | Retrieves top-N most frequent categories                  |
| `filter_by_categories()`        | Keeps only entries from those top-N categories            |
| `create_train_val_test_split()` | Splits dataset into 80/10/10 stratified sets              |
| `create_label_encoder()`        | Encodes category labels numerically                       |
| `save_label_mapping()`          | Saves YAML mapping: `label ↔ id`                          |
| `save_splits()`                 | Exports train/val/test CSVs and JSON metadata             |
| `save_full_cleaned_dataset()`   | Stores cleaned dataset per N category                     |
| `process_all_categories()`      | Loops through all `CATEGORY_SIZES` and runs full pipeline |

---

###  Cleaning Pipeline Steps

1. **Convert to lowercase**
2. **Remove URLs, emails, punctuation, and numbers**
3. **Filter words** based on length and stopword lists
4. **Lemmatize** using WordNet (ensures singular base form)
5. **Save cleaned text** to new column `cleaned_text`

---

### 🗂 Output Artifacts

| File                                                       | Description         |
| ---------------------------------------------------------- | ------------------- |
| `/data/processed/top_50_categories/cleaned_dataset.csv`    | Cleaned dataset     |
| `/data/processed/splits/top_50_categories/train.csv`       | Training split      |
| `/data/processed/splits/top_50_categories/val.csv`         | Validation split    |
| `/data/processed/splits/top_50_categories/test.csv`        | Test split          |
| `/data/processed/labels_top_50_categories.yaml`            | Label encoding info |
| `/data/processed/splits/top_50_categories/split_info.json` | Metadata summary    |

---

##  5. Feature Extraction (`feature_extraction.py`)

### Purpose

Transforms cleaned text data into **numerical feature representations** (TF-IDF and SBERT embeddings) to be consumed by different model types.

---

### Main Components

#### **Class: `FeatureExtractor`**

| Method                              | Description                                   |
| ----------------------------------- | --------------------------------------------- |
| `create_tfidf_features()`           | Builds TF-IDF vector representations          |
| `create_sbert_features()`           | Uses Sentence-BERT embeddings                 |
| `save_tfidf_features()`             | Stores `.pkl` and `feature_info.json`         |
| `save_sbert_features()`             | Saves `.npy` and `embedding_config.json`      |
| `extract_features_for_category()`   | Runs both TF-IDF and SBERT for a given N      |
| `extract_features_all_categories()` | Loops over all category sizes                 |
| `create_feature_summary()`          | Summarizes feature metadata and dataset sizes |
| `validate_features()`               | Checks existence of all output files          |

---

###  Feature Types

####  TF-IDF

* Sparse matrix representation
* Vocabulary limited to 10K most important terms
* Supports bigrams `(1,2)`
* Removes rare (min_df=2) and overly common (max_df=0.95) words

**Outputs:**

* `train_features.pkl`, `val_features.pkl`, `test_features.pkl`
* `vectorizer.pkl`
* `feature_info.json`

####  SBERT

* Dense semantic embeddings from SentenceTransformer
* Model: `all-MiniLM-L6-v2`
* Produces 384-D feature vectors
* Normalized embeddings for cosine similarity

**Outputs:**

* `train_embeddings.npy`, `val_embeddings.npy`, `test_embeddings.npy`
* `embedding_config.json`

---

### 📊 Feature Summary Example

```json
{
  "categories": {
    "50": {
      "dataset_sizes": {"train": 4000, "val": 500, "test": 500},
      "features": {
        "tfidf": {"vocabulary_size": 10000, "feature_count": 5120},
        "sbert": {"embedding_dimension": 384}
      }
    }
  }
}
```

---

##  6. Workflow Summary

```mermaid
flowchart TD
    A[Raw Data CSV] --> B[Data Analysis]
    B --> C[Data Preprocessing]
    C --> D[Feature Extraction]
    D --> E[Model Training (ML/DL/BERT/DeepSeek/Fusion)]
    E --> F[Evaluation & Reports]
```

| Stage                        | Input                | Output                               |
| ---------------------------- | -------------------- | ------------------------------------ |
| **1. Analysis**              | Raw dataset          | JSON + PNG summaries                 |
| **2. Preprocessing**         | Cleaned text, labels | Train/Val/Test splits                |
| **3. Feature Extraction**    | Cleaned splits       | TF-IDF vectors + SBERT embeddings    |
| **4. Modeling (Next Stage)** | Features             | Trained classification models        |
| **5. Evaluation**            | Predictions          | Accuracy, F1, confusion matrix, logs |

---

##  7. Key Benefits

* **Consistency:** Unified paths and naming conventions across modules
* **Modularity:** Independent execution of analysis, preprocessing, and features
* **Scalability:** Supports multiple model architectures and category sizes
* **Reproducibility:** Fixed `RANDOM_SEED` ensures identical splits and results
* **Extensibility:** Easy to plug in new model or embedding types

---

##  8. Execution Guide

| Command                            | Description                          |
| ---------------------------------- | ------------------------------------ |
| `python src/config.py`             | Initialize directories               |
| `python src/data_analysis.py`      | Run data exploration and EDA reports |
| `python src/data_preprocessing.py` | Clean and split dataset              |
| `python src/feature_extraction.py` | Generate TF-IDF and SBERT features   |

All output artifacts are automatically organized under `/data`, `/models`, `/results`, and `/logs`.




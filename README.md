# Web Service Classification – XAI

## Overview

1. **Objective:** Establish baseline performance for web service classification.
2. **Dataset:** Top-N categories (50).
3. **Models:**  
   - ML: Logistic Regression, Random Forest, XGBoost  
   - DL: BiLSTM  
   - RoBERTa: Small & Large  
   - DeepSeek: Embedding + Semantic Features  
   - Fusion: DeepSeek + RoBERTa + Classifier  
4. **Features:** TF-IDF and SBERT embeddings.
5. **Evaluation:** Top-1, Top-3, Top-5 accuracy, Macro/Micro F1, confusion matrices.
6. **Balanced Datasets:** Fixed 80/10/10 train/validation/test splits.
7. **Reproducibility:** Configurations stored in YAML files.
8. **Benchmarking:** Leaderboards, Top-K curves, confusion matrices.
9. **Total Models Trained:** 12  
(6 ML + 2 DL + 2 RoBERTa + 1 DeepSeek + 1 Fusion)
10. **Analysis:** Cross-model comparison, ranking quality, and category difficulty.

---

## 🚀 Steps to Run the Project

```bash
# 1 Clone the repository
git clone git@github.com:Alikhan-Buaa/XAI-MAS-Web-Service-Classification.git

# 2 Navigate to project directory
cd XAI-MAS-Web-Service-Classification/web_services_classification/

# 3 Install dependencies
pip install -r requirements.txt

# 4 Download NLTK resources
python -m nltk.downloader punkt stopwords wordnet

# ---- RUN PIPELINE ----

# 5 Data Analysis
python main.py --phase analysis

# 6 Preprocessing
python main.py --phase preprocessing

# 7 Feature Extraction
python main.py --phase features

# 8 Machine Learning Training
python main.py --phase ml_training

# 9 Deep Learning Training
python main.py --phase dl_training

# 10 DeepSeek Training
python main.py --phase deepseek_training

# 11 BERT / RoBERTa Training
python main.py --phase bert_training

# 12 Fusion Model Training (DeepSeek + RoBERTa)
python main.py --phase fusion_training

# 13 Evaluation
python main.py --phase evaluation

# 14 Visualization
python main.py --phase visualize


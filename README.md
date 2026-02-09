
# Web Service Classification – XAI

## Overview

1. **Objective:** Establish baseline and advanced performance for web service classification with Explainable AI support.

2. **Dataset:** Top-N categories (50). Balanced and stratified.

3. **Models:**

   * ML: Logistic Regression, Random Forest, XGBoost
   * DL: BiLSTM
   * RoBERTa: Small & Large
   * DeepSeek: Embedding + Semantic Features
   * Fusion: DeepSeek + RoBERTa + Classifier (Concat / Weighted / Gating variants)

4. **Features:** TF-IDF and SBERT embeddings.

5. **Evaluation:** Top-1, Top-3, Top-5 accuracy, Macro/Micro F1, Precision/Recall, confusion matrices.

6. **Balanced Datasets:** Fixed 80/10/10 train/validation/test splits.

7. **Reproducibility:** Configurations stored in YAML files + saved model artifacts.

8. **Benchmarking:** Leaderboards, Top-K curves, confusion matrices, comparison charts.

9. **Explainability (XAI):** SHAP, LIME, model attribution, fusion contribution analysis.

10. **Total Models Trained:** 18
    (6 ML + 2 DL + 4 RoBERTa + 2 DeepSeek + 4 Fusion)

11. **Analysis:** Cross-model comparison, ranking quality, and category difficulty.

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

# 7 Feature Extraction (TFIDF + SBERT)
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

# 13 Evaluation & Metrics
python main.py --phase evaluation

# 14 Visualization & Comparison Charts
python main.py --phase visualize

# ---- EXPLAINABILITY (XAI) ----

# 15 ML Explainability (SHAP + LIME)
python main.py --phase ml_explainability

# 16 DL Explainability
python main.py --phase dl_explainability

# 17 BERT / RoBERTa Explainability
python main.py --phase bert_explainability

# 18 DeepSeek Explainability
python main.py --phase deepseek_explainability

# 19 Fusion Explainability
python main.py --phase fusion_explainability

# 20 Overall Cross-Model Explainability
python main.py --phase overall_explainability

# ---- RUN EVERYTHING ----

# 21 Full End-to-End Pipeline
python main.py --phase all
```





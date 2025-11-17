

1. Objective: Establish baseline performance for web service classification (Top-50 categories).
2. Dataset: Top-50 categories with balanced 80/10/10 train/val/test splits.
3. Models: ML (Logistic Regression, Random Forest, XGBoost), DL (BiLSTM), Top-50 DL Extension (BERT-base, BERT-large).
4. Features: TF-IDF and SBERT embeddings.
5. Evaluation: Top-1, Top-3, Top-5 accuracy, Macro/Micro F1, confusion matrices.
6. Balanced datasets: Fixed 80/10/10 train/val/test splits.
7. Reproducibility: All configurations stored in YAML files.
8. Benchmarking: Leaderboards, Top-K curves, and confusion matrices.
9. Total models trained (Top-50 phase): 6 ML + 2 DL + 2 Top-50 DL = 10 models.
10. Analysis: Cross-model comparison, transformer vs. classical baselines, ranking quality, and category difficulty.


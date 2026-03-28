"""
ML Model Explainability Module
Thin caller — all shared logic lives in src/utils/explainability_utils.py
"""

import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import traceback
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
import os

import shap
from lime.lime_text import LimeTextExplainer

from src.config import (
    DATA_PATH, RESULTS_PATH, SAVED_MODELS_CONFIG, PREPROCESSING_CONFIG,
    RESULTS_CONFIG
)
from src.utils.explainability_utils import (
    STOPWORDS, FALLBACK_LABELS, TARGET_CATEGORIES,
    load_class_labels, get_shared_samples,
    top15_tokens, plot_bar, compute_metrics,
    run_global_category_bar, extract_global_tokens,
    run_global_lime, run_beeswarm, run_waterfall,
    save_metrics_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MLExplainability:
    def __init__(self, n_categories=50):
        self.feature_extractor = None
        self.plot_dpi = 300
        self.model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
        self.all_dominant_tokens = defaultdict(list)
        self.global_metrics_storage = []
        self.target_categories = TARGET_CATEGORIES   # from utils — single source
        self.setup_directories(n_categories)

    def setup_directories(self, n_categories):
        self.n_categories = n_categories
        base_path = RESULTS_CONFIG['ml_results_path'] / f"top_{n_categories}_categories" / "explainability"
        self.dirs = {
            'shap':       base_path / "shap",
            'lime':       base_path / "lime",
            'extra_lime': base_path / "lime" / "extra_lime_explainer",
            'global_bar': base_path / "shap" / "global_bar",
            'beeswarm':   base_path / "shap" / "beeswarm",
            'waterfall':  base_path / "shap" / "waterfall",
            'samples':    base_path / "shap" / "samples",
            'global_lime':base_path / "lime" / "global",
            'reports':    base_path / "reports",
            'metrics':    base_path / "metrics",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def load_model_and_data(self, model_name, feature_type="tfidf"):
        logger.info(f"Loading {model_name} ({feature_type})...")
        model_dir = SAVED_MODELS_CONFIG["ml_models_path"] / f"top_{self.n_categories}_categories"
        patterns = [
            f"{model_name}_{feature_type.upper()}_top_{self.n_categories}_categories_model.pkl",
            f"{model_name}_{feature_type.lower()}_top_{self.n_categories}_categories_model.pkl",
            f"{model_name}_{feature_type.upper()}_model.pkl",
        ]
        model_path = next((model_dir / p for p in patterns if (model_dir / p).exists()), None)
        if not model_path:
            logger.error(f"Model missing in {model_dir}. Checked: {patterns}")
            return None, None, None, None, None

        model = joblib.load(model_path)
        from src.preprocessing.feature_extraction import FeatureExtractor
        self.feature_extractor = FeatureExtractor()

        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df  = pd.read_csv(splits_dir / "test.csv")
        train_df = pd.read_csv(splits_dir / "train.csv")

        if feature_type == "tfidf":
            self.feature_extractor.load_tfidf_vectorizer(self.n_categories)
            if not hasattr(self.feature_extractor, 'tfidf_vectorizer'):
                vec_path = DATA_PATH / "features" / "tfidf" / f"top_{self.n_categories}_categories" / "tfidf_vectorizer.pkl"
                self.feature_extractor.tfidf_vectorizer = joblib.load(vec_path)
            X_train      = self.feature_extractor.tfidf_vectorizer.transform(train_df["cleaned_text"])
            feature_names = self.feature_extractor.tfidf_vectorizer.get_feature_names_out()
        else:
            X_train      = self.feature_extractor.load_sbert_features(self.n_categories, "train")
            feature_names = [f"dim_{i}" for i in range(X_train.shape[1])]

        # load_class_labels() from utils — no local duplication
        class_labels = load_class_labels(self.n_categories)
        return model, X_train, test_df, feature_names, class_labels

    def get_prediction_pipeline(self, model, feature_type):
        if feature_type == "tfidf":
            def tfidf_pipeline(texts):
                return model.predict_proba(
                    self.feature_extractor.tfidf_vectorizer.transform(texts))
            return tfidf_pipeline
        else:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer('all-MiniLM-L6-v2')
            def sbert_pipeline(texts):
                return model.predict_proba(sbert.encode(texts))
            return sbert_pipeline

    # ── shared helper: delegate to utils ──────────────────────────────────────
    def _plot_bar(self, items, title, output_path):
        """Thin wrapper: filter dim_ / numeric / empty, then call shared plot_bar."""
        clean = [(tok, w) for tok, w in items
                 if tok and not str(tok).startswith("dim_")
                 and not str(tok).isnumeric() and len(str(tok)) >= 2]
        plot_bar(clean, title, Path(output_path), plot_dpi=self.plot_dpi)

    def explain_model(self, model_name, feature_type):
        model, X_train, test_df, feature_names, class_labels = self.load_model_and_data(
            model_name, feature_type)
        if model is None:
            return

        pipeline_fn    = self.get_prediction_pipeline(model, feature_type)
        lime_explainer = LimeTextExplainer(class_names=class_labels)

        # ── 1. Global SHAP ────────────────────────────────────────────────────
        logger.info(f"Running Global SHAP for {model_name}...")
        bg = X_train[:5].toarray() if hasattr(X_train, "toarray") else X_train[:5]

        if model_name == "LogisticRegression":
            explainer        = shap.LinearExplainer(model, bg, feature_names=feature_names)
            shap_vals_global = explainer.shap_values(bg)
        else:
            explainer        = shap.TreeExplainer(model)
            shap_vals_global = explainer.shap_values(bg, check_additivity=False)

        # Global category bar (utils)
        run_global_category_bar(
            shap_vals_global, class_labels, model_name, self.target_categories,
            self.dirs['global_bar'] / f"global_{model_name}_{feature_type}.png",
            plot_dpi=self.plot_dpi,
        )

        # Extract per-category tokens (utils)
        cat_tokens = extract_global_tokens(shap_vals_global, class_labels, feature_names,
                                           self.target_categories)
        for cat, toks in cat_tokens.items():
            self.all_dominant_tokens[cat].extend(toks)

        # Beeswarm (TF-IDF only — summary_plot needs the raw array)
        if feature_type == "tfidf":
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_vals_global, bg, feature_names=feature_names,
                                  max_display=15, show=False)
                plt.title(f"Beeswarm Top 15 — {model_name}", fontsize=13, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.dirs['beeswarm'] / f"beeswarm_{model_name}_{feature_type}.png",
                            dpi=self.plot_dpi, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Beeswarm failed: {e}")

        # ── 2. Global LIME (utils) ────────────────────────────────────────────
        run_global_lime(
            lime_explainer, pipeline_fn, test_df, f"{model_name}_{feature_type}",
            self.dirs['global_lime'] / f"Global_LIME_{model_name}_{feature_type}.png",
            plot_dpi=self.plot_dpi,
        )

        # ── 3. Local samples — canonical shared index (utils) ─────────────────
        shared = get_shared_samples(
            test_df=test_df,
            n_categories=self.n_categories,
            results_root=self.dirs['reports'],
        )
        # One representative per category for waterfall + local bars
        indices_to_explain = []
        seen_cats = set()
        for row_i, cat_name in shared:
            if cat_name not in seen_cats:
                indices_to_explain.append(row_i)
                seen_cats.add(cat_name)
            if len(seen_cats) >= 5:
                break

        sbert_model = None
        if feature_type == "sbert":
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        beeswarm_rows = []
        waterfall_done = False

        for idx_count, i in enumerate(indices_to_explain):
            try:
                text    = test_df.iloc[i]['cleaned_text']
                probs   = pipeline_fn([text])[0]
                top_cls = int(np.argmax(probs))
                cat_name = class_labels[top_cls]

                # LIME local
                exp = lime_explainer.explain_instance(
                    text, pipeline_fn, num_features=30,
                    labels=[top_cls], num_samples=1000)
                try:
                    exp.save_to_file(str(
                        self.dirs['extra_lime'] / f"dashboard_{model_name}_{feature_type}_{i}.html"))
                except Exception:
                    pass

                lime_raw   = exp.as_list(label=top_cls)
                lime_clean = top15_tokens([f for f, _ in lime_raw], [w for _, w in lime_raw])
                self._plot_bar(
                    lime_clean,
                    f"Top 15 Tokens — LIME · {model_name} · {cat_name}",
                    self.dirs['lime'] / f"lime_{model_name}_{i}_{feature_type}.png",
                )

                # SHAP local
                shap_clean_for_metrics = []
                if feature_type == "sbert":
                    vec = sbert_model.encode([text]).reshape(1, -1)
                    if model_name == "LogisticRegression":
                        local_shap = explainer.shap_values(vec)
                    else:
                        local_shap = explainer.shap_values(vec, check_additivity=False)
                    if isinstance(local_shap, list):   sv = local_shap[top_cls][0]
                    elif local_shap.ndim == 3:         sv = local_shap[0, :, top_cls]
                    else:                              sv = local_shap[0]
                    shap_clean_for_metrics = top15_tokens(feature_names, sv)
                    self._plot_bar(
                        lime_clean,  # proxy words for SBERT visual
                        f"SHAP Tokens (Text Proxy) · {model_name} · {cat_name}",
                        self.dirs['samples'] / f"shap_sample_{i}_{model_name}_{feature_type}.png",
                    )
                else:
                    vec = self.feature_extractor.tfidf_vectorizer.transform([text]).toarray()
                    if model_name == "LogisticRegression":
                        local_shap = explainer.shap_values(vec)
                    else:
                        local_shap = explainer.shap_values(vec, check_additivity=False)
                    if isinstance(local_shap, list):   sv = local_shap[top_cls][0]
                    elif local_shap.ndim == 3:         sv = local_shap[0, :, top_cls]
                    else:                              sv = local_shap[0]

                    if np.max(np.abs(sv)) > 1000:
                        sv = sv / (np.sum(np.abs(sv)) + 1e-9)

                    shap_clean_for_metrics = top15_tokens(feature_names, sv)
                    self._plot_bar(
                        shap_clean_for_metrics,
                        f"SHAP Tokens · {model_name} · {cat_name}",
                        self.dirs['samples'] / f"shap_sample_{i}_{model_name}_{feature_type}.png",
                    )

                    # Beeswarm accumulation
                    for tok, val in shap_clean_for_metrics:
                        beeswarm_rows.append({'Token': tok, 'SHAP Value': float(val)})

                # Waterfall — first sample only (utils)
                if not waterfall_done and shap_clean_for_metrics:
                    base_val = 0.0
                    if hasattr(explainer, 'expected_value'):
                        ev = explainer.expected_value
                        base_val = float(
                            ev[top_cls] if isinstance(ev, (list, np.ndarray)) else ev)
                    run_waterfall(
                        shap_clean_for_metrics, base_val, model_name, cat_name,
                        self.dirs['waterfall'] / f"waterfall_{model_name}_{feature_type}_{i}.png",
                        plot_dpi=self.plot_dpi,
                    )
                    waterfall_done = True

                # Honest metrics (utils)
                mets = compute_metrics(exp.score, shap_clean_for_metrics, lime_clean)
                mets['model'] = f"{model_name}_{feature_type}"
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"Local sample failed idx={i}: {e}")
                traceback.print_exc()

        # Beeswarm (utils)
        if beeswarm_rows:
            run_beeswarm(
                beeswarm_rows, f"{model_name}_{feature_type}",
                self.dirs['beeswarm'] / f"beeswarm_{model_name}_{feature_type}_local.png",
                plot_dpi=self.plot_dpi,
            )

    def explain_all_models(self, n_categories=50, feature_types=None):
        self.n_categories = n_categories
        self.setup_directories(n_categories)
        if feature_types is None:
            feature_types = ["tfidf", "sbert"]
        for f_type in feature_types:
            for m_name in self.model_names:
                try:
                    self.explain_model(m_name, f_type)
                except Exception as e:
                    logger.error(f"Pipeline failed {m_name}/{f_type}: {e}")
        self.save_reports()
        return self.global_metrics_storage

    def save_reports(self):
        # Metrics CSV + chart (utils)
        save_metrics_report(
            self.global_metrics_storage, model_col='model',
            output_csv=self.dirs['metrics'] / "ML_Final_Metrics.csv",
            output_png=self.dirs['metrics'] / "ML_Metrics_Comparison.png",
            title="ML Explainability Metrics",
            plot_dpi=self.plot_dpi,
        )
        # Consolidated dominant tokens CSV
        rows = []
        for cat in self.target_categories:
            toks = [t for t in self.all_dominant_tokens.get(cat, [])
                    if not str(t).startswith("dim_")]
            top  = [w for w, _ in Counter(toks).most_common(15)]
            rows.append({'Category': cat,
                         'Consolidated_Top_15_Tokens': ", ".join(top) if top else "N/A"})
        pd.DataFrame(rows).to_csv(
            self.dirs['reports'] / "ML_Consolidated_Dominant_Tokens.csv", index=False)


if __name__ == "__main__":
    exp = MLExplainability()
    exp.explain_all_models(50)

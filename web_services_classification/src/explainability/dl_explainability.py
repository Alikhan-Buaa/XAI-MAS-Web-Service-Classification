"""
DL (BiLSTM) Explainability Module
Thin caller — all shared logic lives in src/utils/explainability_utils.py
"""

import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
import os
import tensorflow as tf

import shap
from lime.lime_text import LimeTextExplainer

from src.config import (
    DATA_PATH, RESULTS_PATH, SAVED_MODELS_CONFIG, PREPROCESSING_CONFIG,
    RESULTS_CONFIG, OVERALL_EXPLAINABILITY_CONFIG
)
from src.utils.explainability_utils import (
    STOPWORDS, TARGET_CATEGORIES,
    load_class_labels, get_shared_samples,
    top15_tokens, plot_bar, compute_metrics,
    run_global_category_bar, extract_global_tokens,
    run_global_lime, run_beeswarm, run_waterfall,
    save_metrics_report,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _nl in ['shap', 'lime', 'sentence_transformers', 'tensorflow']:
    logging.getLogger(_nl).setLevel(logging.ERROR)
    logging.getLogger(_nl).propagate = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
tf.compat.v1.enable_v2_behavior()
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DLExplainability:
    def __init__(self, n_categories=50):
        self.feature_extractor = None
        self.plot_dpi = 300
        self.model_names = ["BiLSTM"]
        self.all_dominant_tokens = defaultdict(list)
        self.global_metrics_storage = []
        self.target_categories = TARGET_CATEGORIES  # from utils
        self.setup_directories(n_categories)

    def setup_directories(self, n_categories):
        self.n_categories = n_categories
        base_path = RESULTS_CONFIG['dl_results_path'] / f"top_{n_categories}_categories" / "explainability"
        self.dirs = {
            'shap':        base_path / "shap",
            'lime':        base_path / "lime",
            'extra_lime':  base_path / "lime" / "extra_lime_explainer",
            'global_bar':  base_path / "shap" / "global_bar",
            'beeswarm':    base_path / "shap" / "beeswarm",
            'waterfall':   base_path / "shap" / "waterfall",
            'samples':     base_path / "shap" / "samples",
            'global_lime': base_path / "lime" / "global",
            'reports':     base_path / "reports",
            'metrics':     base_path / "metrics",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def load_model_and_data(self, model_name, feature_type="tfidf"):
        logger.info(f"Loading DL {model_name} ({feature_type})...")
        model_dir = SAVED_MODELS_CONFIG["dl_models_path"] / f"top_{self.n_categories}_categories"
        model_path = next(
            (model_dir / f for f in os.listdir(model_dir)
             if model_name in f and feature_type.lower() in f.lower()
             and (f.endswith(".h5") or f.endswith(".keras"))),
            None,
        ) if model_dir.exists() else None
        if not model_path:
            logger.error(f"Model missing in {model_dir}")
            return None, None, None, None, None

        model = tf.keras.models.load_model(model_path)

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
            X_train       = self.feature_extractor.tfidf_vectorizer.transform(train_df["cleaned_text"]).toarray()
            feature_names = self.feature_extractor.tfidf_vectorizer.get_feature_names_out()
        else:
            X_train       = self.feature_extractor.load_sbert_features(self.n_categories, "train")
            feature_names = [f"dim_{i}" for i in range(X_train.shape[1])]

        class_labels = load_class_labels(self.n_categories)
        return model, X_train, test_df, feature_names, class_labels

    def get_prediction_pipeline(self, model, feature_type, sbert_model=None):
        if feature_type == "tfidf":
            def tfidf_pipeline(texts):
                vecs = self.feature_extractor.tfidf_vectorizer.transform(texts).toarray()
                return model.predict(vecs, batch_size=128, verbose=0)
            return tfidf_pipeline
        else:
            def sbert_pipeline(texts):
                vecs = sbert_model.encode(texts, batch_size=128, show_progress_bar=False)
                return model.predict(vecs, batch_size=128, verbose=0)
            return sbert_pipeline

    def _plot_bar(self, items, title, output_path):
        clean = [(tok, w) for tok, w in items
                 if tok and not str(tok).startswith("dim_")
                 and not str(tok).isnumeric() and len(str(tok)) >= 2]
        plot_bar(clean, title, Path(output_path), plot_dpi=self.plot_dpi)

    def explain_model(self, model_name, feature_type):
        model, X_train, test_df, feature_names, class_labels = self.load_model_and_data(
            model_name, feature_type)
        if model is None:
            return

        sbert_model = None
        if feature_type == 'sbert':
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        pipeline_fn    = self.get_prediction_pipeline(model, feature_type, sbert_model)
        lime_explainer = LimeTextExplainer(class_names=class_labels)

        # ── 1. Global SHAP ────────────────────────────────────────────────────
        logger.info(f"Running Global SHAP for DL {model_name}...")
        bg_summary   = shap.kmeans(X_train, 5)
        def predict_wrap(x): return model.predict(x, batch_size=128, verbose=0)
        explainer    = shap.KernelExplainer(predict_wrap, bg_summary)
        shap_vals_global = explainer.shap_values(X_train[:50], silent=True)

        run_global_category_bar(
            shap_vals_global, class_labels, f"DL {model_name}", self.target_categories,
            self.dirs['global_bar'] / f"global_{model_name}_{feature_type}.png",
            plot_dpi=self.plot_dpi,
        )

        cat_tokens = extract_global_tokens(shap_vals_global, class_labels, feature_names,
                                           self.target_categories)
        for cat, toks in cat_tokens.items():
            self.all_dominant_tokens[cat].extend(toks)

        if feature_type == "tfidf":
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_vals_global, X_train[:50],
                                  feature_names=feature_names, max_display=15, show=False)
                plt.title(f"Beeswarm Top 15 — DL {model_name}", fontsize=13, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.dirs['beeswarm'] / f"beeswarm_{model_name}_{feature_type}.png",
                            dpi=self.plot_dpi, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Beeswarm failed: {e}")

        # ── 2. Global LIME (utils) ────────────────────────────────────────────
        run_global_lime(
            lime_explainer, pipeline_fn, test_df, f"DL_{model_name}_{feature_type}",
            self.dirs['global_lime'] / f"Global_LIME_{model_name}_{feature_type}.png",
            plot_dpi=self.plot_dpi,
        )

        # ── 3. Local samples — shared index (utils) ───────────────────────────
        shared = get_shared_samples(
            test_df=test_df,
            n_categories=self.n_categories,
            results_root=self.dirs['reports'],
        )
        # shared already returns exactly 1 row per category (5 total) — use directly
        indices_to_explain = list(shared)  # [(row_i, cat_name), ...]

        # SHAP Text Explainer for SBERT
        text_explainer = None
        if feature_type == 'sbert':
            def sbert_text_predict(texts):
                if isinstance(texts, np.ndarray): texts = texts.tolist()
                vecs = sbert_model.encode([str(t) for t in texts],
                                          batch_size=128, show_progress_bar=False)
                return model.predict(vecs, batch_size=128, verbose=0)
            text_explainer = shap.Explainer(sbert_text_predict, shap.maskers.Text(r"\W+"))

        beeswarm_rows = []

        for idx_count, (i, shared_cat) in enumerate(indices_to_explain):
            try:
                text     = str(test_df.iloc[i]['cleaned_text'])
                probs    = pipeline_fn([text])[0]
                top_cls  = int(np.argmax(probs))
                cat_name = class_labels[top_cls]

                exp = lime_explainer.explain_instance(
                    text, pipeline_fn, num_features=30,
                    labels=[top_cls], num_samples=100)
                try:
                    safe_cat = shared_cat.replace(" ", "_")
                    exp.save_to_file(str(
                        self.dirs['extra_lime'] / f"dashboard_{model_name}_{feature_type}_{safe_cat}_{i}.html"))
                except Exception:
                    pass

                lime_raw   = exp.as_list(label=top_cls)
                lime_clean = top15_tokens([f for f, _ in lime_raw], [w for _, w in lime_raw])
                self._plot_bar(lime_clean,
                               f"Top 15 Tokens — LIME · DL {model_name} · {cat_name}",
                               self.dirs['lime'] / f"lime_{model_name}_{i}_{feature_type}.png")

                shap_clean = []
                exp_obj    = None

                if feature_type == 'sbert':
                    shap_obj = text_explainer([text], max_evals=150)
                    sv_raw   = shap_obj[0].values[:, top_cls]
                    words_raw = shap_obj[0].data
                    base_val  = shap_obj[0].base_values[top_cls]
                    word_agg  = defaultdict(float)
                    new_base  = float(base_val)
                    for w, val in zip(words_raw, sv_raw):
                        ws = str(w).lower().strip()
                        if ws in STOPWORDS or len(ws) < 2: new_base += val
                        else: word_agg[ws] += val
                    words = list(word_agg.keys())
                    sv    = np.array(list(word_agg.values()))
                    if np.max(np.abs(sv), initial=0) > 100:
                        sv = sv / (np.sum(np.abs(sv)) + 1e-9)
                        new_base /= (np.sum(np.abs(sv)) + 1e-9)
                    shap_clean = top15_tokens(words, sv)
                    exp_obj    = shap.Explanation(values=np.array([v for _, v in shap_clean]),
                                                  base_values=new_base,
                                                  data=np.array([t for t, _ in shap_clean]),
                                                  feature_names=[t for t, _ in shap_clean])
                else:
                    vec      = self.feature_extractor.tfidf_vectorizer.transform([text]).toarray()
                    local_sv = explainer.shap_values(vec, silent=True)
                    base_val = 0.0
                    if hasattr(explainer, 'expected_value'):
                        ev = explainer.expected_value
                        base_val = float(ev[top_cls] if isinstance(ev, (list, np.ndarray)) else ev)
                    if isinstance(local_sv, list):   sv = local_sv[top_cls][0]
                    elif local_sv.ndim == 3:         sv = local_sv[0, :, top_cls]
                    else:                            sv = local_sv[0]
                    word_agg = defaultdict(float)
                    new_base = base_val
                    for w, val in zip(feature_names, sv):
                        ws = str(w).lower().strip()
                        if ws in STOPWORDS or len(ws) < 2: new_base += val
                        else: word_agg[ws] += val
                    fnames = list(word_agg.keys())
                    sv     = np.array(list(word_agg.values()))
                    if np.max(np.abs(sv), initial=0) > 100:
                        sv = sv / (np.sum(np.abs(sv)) + 1e-9)
                    shap_clean = top15_tokens(fnames, sv)
                    exp_obj    = shap.Explanation(values=np.array([v for _, v in shap_clean]),
                                                  base_values=new_base,
                                                  data=np.zeros(len(shap_clean)),
                                                  feature_names=[t for t, _ in shap_clean])
                    for tok, val in shap_clean:
                        beeswarm_rows.append({'Token': tok, 'SHAP Value': float(val)})

                self._plot_bar(shap_clean,
                               f"SHAP Tokens · DL {model_name} · {cat_name}",
                               self.dirs['samples'] / f"shap_sample_{i}_{model_name}_{feature_type}.png")

                # Waterfall — one per category
                if shap_clean:
                    safe_cat_wf = shared_cat.replace(" ", "_")
                    run_waterfall(shap_clean, float(exp_obj.base_values) if exp_obj else 0.0,
                                  f"DL_{model_name}", shared_cat,
                                  self.dirs['waterfall'] / f"waterfall_{model_name}_{feature_type}_{safe_cat_wf}.png",
                                  plot_dpi=self.plot_dpi)

                mets = compute_metrics(exp.score, shap_clean, lime_clean)
                mets['model'] = f"{model_name}_{feature_type}"
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"Local sample failed idx={i}: {e}")
                traceback.print_exc()

        if beeswarm_rows:
            run_beeswarm(beeswarm_rows, f"DL_{model_name}_{feature_type}",
                         self.dirs['beeswarm'] / f"beeswarm_{model_name}_{feature_type}_local.png",
                         plot_dpi=self.plot_dpi)

    def explain_all_models(self, n_categories=50, feature_types=None):
        self.n_categories = n_categories
        self.setup_directories(n_categories)
        if feature_types is None:
            feature_types = ["tfidf", "sbert"]
        for f_type in feature_types:
            for m_name in self.model_names:
                try: self.explain_model(m_name, f_type)
                except Exception as e: logger.error(f"Pipeline failed {m_name}/{f_type}: {e}")
        self.save_reports()
        return self.global_metrics_storage

    def save_reports(self):
        save_metrics_report(
            self.global_metrics_storage, model_col='model',
            output_csv=self.dirs['metrics'] / "DL_Final_Metrics.csv",
            output_png=self.dirs['metrics'] / "DL_Metrics_Comparison.png",
            title="DL Explainability Metrics",
            plot_dpi=self.plot_dpi,
        )
        rows = []
        for cat in self.target_categories:
            toks = [t for t in self.all_dominant_tokens.get(cat, [])
                    if not str(t).startswith("dim_")]
            top  = [w for w, _ in Counter(toks).most_common(15)]
            rows.append({'Category': cat,
                         'Consolidated_Top_15_Tokens': ", ".join(top) if top else "N/A"})
        pd.DataFrame(rows).to_csv(
            self.dirs['reports'] / OVERALL_EXPLAINABILITY_CONFIG['token_files']['dl'], index=False)


if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    args = parser.parse_args()
    t0 = time.time()
    DLExplainability(n_categories=args.categories).explain_all_models(args.categories)
    logger.info(f"PHASE COMPLETED: DL_EXPLAINABILITY ({time.time()-t0:.2f}s)")

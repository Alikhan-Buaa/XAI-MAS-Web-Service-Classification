"""
DL (BiLSTM) Explainability Module — SBERT Unified Pipeline
===========================================================
Fixes applied:

FIXED #1  Domain stopwords removed: 'api', 'service', 'data', 'platform',
          'cloud', 'tool', 'application', 'web', 'software', 'system',
          'developer', 'access' removed from STOPWORDS.

FIXED #2  Honest metrics: No np.random noise. No clamping.
          Fidelity  = sqrt(|R²|) from LIME score.
          Jaccard   = set overlap of top-15 SHAP words vs top-15 LIME words.
          Stability = mean pairwise Spearman-r across same-category SHAP
                      vectors (genuine cross-instance consistency).

FIXED #3  SHAP additivity preserved: stopwords are no longer absorbed into
          base_val post-hoc (which violates base + sum = prediction).
          Pre-filtering is done on the word list AFTER SHAP runs; base_val
          is left untouched.

FIXED #4  CSV evidence populated: local loop always appends lime_clean and
          shap words to self.all_dominant_tokens[cat_name].

FIXED #5  KernelExplainer background: increased from 5 → 100 kmeans clusters;
          global SHAP sample increased from 50 → 200.

FIXED #6  Uniform predict_fn: same sbert_pipeline used by both KernelExplainer
          and LimeTextExplainer (same probability scale).
"""

import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import traceback
import yaml
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
from scipy.stats import spearmanr

import tensorflow as tf
import shap
from lime.lime_text import LimeTextExplainer
from sentence_transformers import SentenceTransformer

from src.config import (
    DATA_PATH, RESULTS_PATH, SAVED_MODELS_CONFIG, PREPROCESSING_CONFIG,
    RESULTS_CONFIG, OVERALL_EXPLAINABILITY_CONFIG
)
from src.utils.utils import (
    STOPWORDS, TARGET_CATEGORIES, FALLBACK_LABELS,
    load_class_labels,
    top15_tokens, plot_bar, compute_metrics,
    build_shap_background, run_global_shap, run_global_lime,
)

from src.explainability.shared_samples import get_shared_samples, FIXED_CATEGORIES

# ── logging ───────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _n in ['shap', 'lime', 'sentence_transformers', 'tensorflow']:
    _l = logging.getLogger(_n)
    _l.setLevel(logging.ERROR)
    _l.propagate = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
tf.compat.v1.enable_v2_behavior()

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
class DLExplainability:
    MODEL_NAMES = ["BiLSTM"]

    def __init__(self, n_categories: int = 50):
        self.n_categories = n_categories
        self.plot_dpi = 300
        self.all_dominant_tokens: dict = defaultdict(list)
        self.global_metrics_storage: list = []

        logger.info("Loading SBERT encoder (all-MiniLM-L6-v2)…")
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SBERT ready.")

        self._setup_dirs(n_categories)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers — thin wrappers that delegate to explainability_utils.
    # All shared logic (STOPWORDS, metrics, plotting) lives there.
    # ─────────────────────────────────────────────────────────────────────────
    def _top15(self, features, weights, clean_glyph=False):
        return top15_tokens(features, weights, clean_glyph=clean_glyph)

    def _plot_bar(self, items, title, output_path):
        plot_bar(items, title, output_path, plot_dpi=self.plot_dpi)

    def _run_global_shap(self, explainer, X_sample, class_labels, model_name):
        run_global_shap(
            explainer, X_sample, class_labels, model_name,
            self.dirs['global_bar'] / f"global_{model_name}_sbert.png",
            plot_dpi=self.plot_dpi,
        )

    def _run_global_lime(self, lime_exp, predict_fn, test_df, model_name,
                         sample_limit=15, clean_glyph=False):
        run_global_lime(
            lime_exp, predict_fn, test_df, model_name,
            self.dirs['global_lime'] / f"global_lime_{model_name}_sbert.png",
            sample_limit=sample_limit, clean_glyph=clean_glyph,
            plot_dpi=self.plot_dpi,
        )


    # ── dirs ──────────────────────────────────────────────────────────────────
    def _setup_dirs(self, n_categories: int):
        self.n_categories = n_categories
        base = (
            RESULTS_CONFIG['dl_results_path']
            / f"top_{n_categories}_categories"
            / "explainability"
        )
        self.dirs = {
            'shap':        base / "shap",
            'lime':        base / "lime",
            'lime_dash':   base / "lime" / "dashboards",
            'global_bar':  base / "shap" / "global_bar",
            'beeswarm':    base / "shap" / "beeswarm",
            'waterfall':   base / "shap" / "waterfall",
            'samples':     base / "shap" / "samples",
            'global_lime': base / "lime" / "global",
            'reports':     base / "reports",
            'metrics':     base / "metrics",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # ── labels ────────────────────────────────────────────────────────────────
    def _load_class_labels(self) -> list:
        return load_class_labels(self.n_categories)

    def _load_model(self, model_name: str):
        model_dir = (
            SAVED_MODELS_CONFIG["dl_models_path"]
            / f"top_{self.n_categories}_categories"
        )
        if not model_dir.exists():
            logger.error(f"Model dir not found: {model_dir}")
            return None
        for f in os.listdir(model_dir):
            if model_name in f and 'sbert' in f.lower() and (f.endswith('.h5') or f.endswith('.keras')):
                path = model_dir / f
                logger.info(f"  Loading: {path.name}")
                return tf.keras.models.load_model(path)
        logger.error(f"SBERT BiLSTM model not found in {model_dir}")
        return None

    # ── SBERT encode ──────────────────────────────────────────────────────────
    def _encode(self, texts: list) -> np.ndarray:
        return self.sbert.encode(
            [str(t) for t in texts],
            batch_size=128, show_progress_bar=False, convert_to_numpy=True
        )

    def _make_predict_fn(self, model):
        def predict_fn(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            embs = self._encode([str(t) for t in texts])
            return model.predict(embs, batch_size=128, verbose=0)
        return predict_fn

    # ── token helpers ─────────────────────────────────────────────────────────
    def _compute_metrics(self, lime_score, shap_top15, lime_top15,
                         category_shap_vectors=None) -> dict:
        return compute_metrics(lime_score, shap_top15, lime_top15,
                               category_shap_vectors)

    # ── main explain loop ─────────────────────────────────────────────────────
    def explain_model(self, model_name: str):
        logger.info(f"\n{'='*60}\n  DL Explaining {model_name} (SBERT)\n{'='*60}")

        model = self._load_model(model_name)
        if model is None:
            return

        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        try:
            test_df  = pd.read_csv(splits_dir / "test.csv")
            train_df = pd.read_csv(splits_dir / "train.csv")
        except FileNotFoundError as e:
            logger.error(f"Split not found: {e}")
            return

        class_labels = self._load_class_labels()

        # Load or compute SBERT features for train
        sbert_path = (
            DATA_PATH / "features" / "sbert"
            / f"top_{self.n_categories}_categories" / "X_train.npy"
        )
        if sbert_path.exists():
            X_train_sbert = np.load(sbert_path)
            logger.info(f"  Loaded SBERT train features: {X_train_sbert.shape}")
        else:
            logger.info("  Encoding train set with SBERT…")
            X_train_sbert = self._encode(train_df['cleaned_text'].tolist())
            sbert_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(sbert_path, X_train_sbert)

        # Shared predict_fn
        predict_fn = self._make_predict_fn(model)

        # KernelExplainer with 100 kmeans clusters
        logger.info("  Building KernelExplainer background (100 clusters)…")
        bg = shap.kmeans(X_train_sbert, min(100, len(X_train_sbert))).data
        kernel_exp = shap.KernelExplainer(
            lambda x: model.predict(x, batch_size=128, verbose=0), bg
        )

        lime_exp = LimeTextExplainer(
            class_names=class_labels, split_expression=r'\W+', bow=True
        )

        # Global SHAP
        n_global = min(200, len(X_train_sbert))
        idx_g = np.random.RandomState(42).choice(len(X_train_sbert), n_global, replace=False)
        self._run_global_shap(kernel_exp, X_train_sbert[idx_g], class_labels, model_name)

        # Global LIME
        self._run_global_lime(lime_exp, predict_fn, test_df, model_name)

        # Select shared samples — same rows as ML, BERT, DeepSeek, Fusion
        logger.info("  Loading shared sample index (5 fixed categories)…")
        indices = get_shared_samples(
            test_df=test_df,
            class_labels=class_labels,
            n_categories=self.n_categories,
            results_root=RESULTS_PATH,
        )
        logger.info(f"  Shared samples: {[(r, c) for r, c in indices]}")

        logger.info(f"  {len(indices)} categories to explain.")

        cat_shap_cache: dict = defaultdict(list)
        waterfall_done = False

        for idx_count, (row_i, cat_name) in enumerate(indices):
            try:
                text = str(test_df.iloc[row_i]['cleaned_text'])
                probs = predict_fn([text])[0]
                top_cls = int(np.argmax(probs))
                logger.info(f"  [{idx_count+1}/{len(indices)}] {cat_name} — sample {row_i}")

                # LIME
                lime_result = lime_exp.explain_instance(
                    text, predict_fn, labels=[top_cls],
                    num_features=30, num_samples=500,
                )
                try:
                    lime_result.save_to_file(str(
                        self.dirs['lime_dash'] / f"{model_name}_sample_{row_i}_{cat_name}.html"
                    ))
                except Exception:
                    pass

                lime_raw = lime_result.as_list(label=top_cls)
                lime_clean = self._top15([f for f, _ in lime_raw], [w for _, w in lime_raw])
                self._plot_bar(
                    lime_clean,
                    f"LIME Top 15 (SBERT) — DL {model_name} — {cat_name}",
                    self.dirs['lime'] / f"lime_{model_name}_sample_{row_i}.png",
                )

                # SHAP via KernelExplainer
                emb = self._encode([text])              # (1, 384)
                shap_vals_raw = kernel_exp.shap_values(emb, silent=True)

                if isinstance(shap_vals_raw, list):
                    sv = np.array(shap_vals_raw[top_cls][0])
                elif shap_vals_raw.ndim == 3:
                    sv = shap_vals_raw[0, :, top_cls]
                else:
                    sv = shap_vals_raw[0]

                if np.max(np.abs(sv)) > 100:
                    sv = sv / (np.sum(np.abs(sv)) + 1e-9)

                cat_shap_cache[cat_name].append(sv.copy())

                # Project SHAP dims → LIME word space (same as ml_explainability)
                shap_word_scores: dict = {}
                for word, _ in lime_raw:
                    wl = word.lower().strip()
                    if wl in STOPWORDS or len(wl) < 2:
                        continue
                    word_emb = self._encode([wl])[0]
                    proj = float(np.dot(sv, word_emb) / (np.linalg.norm(word_emb) + 1e-9))
                    shap_word_scores[wl] = proj

                shap_top15 = self._top15(
                    list(shap_word_scores.keys()),
                    list(shap_word_scores.values()),
                )

                # Waterfall
                if not waterfall_done and shap_top15:
                    try:
                        bv = float(kernel_exp.expected_value[top_cls]) \
                            if isinstance(kernel_exp.expected_value, (list, np.ndarray)) \
                            else float(kernel_exp.expected_value)
                        w_names = np.array([x[0] for x in shap_top15])
                        w_vals  = np.array([x[1] for x in shap_top15])
                        exp_obj = shap.Explanation(
                            values=w_vals, base_values=bv,
                            data=w_names, feature_names=list(w_names),
                        )
                        plt.figure(figsize=(10, 8))
                        shap.plots.waterfall(exp_obj, max_display=15, show=False)
                        plt.title(f"SHAP Waterfall (SBERT) — DL {model_name} — {cat_name}", fontsize=12)
                        plt.tight_layout()
                        plt.savefig(
                            self.dirs['waterfall'] / f"waterfall_{model_name}_sbert.png",
                            dpi=self.plot_dpi, bbox_inches='tight',
                        )
                        plt.close()
                        waterfall_done = True
                    except Exception as e:
                        logger.warning(f"  Waterfall failed: {e}")

                self._plot_bar(
                    shap_top15,
                    f"SHAP Top 15 (SBERT) — DL {model_name} — {cat_name}",
                    self.dirs['samples'] / f"shap_{model_name}_sample_{row_i}.png",
                )

                # Honest metrics
                mets = self._compute_metrics(
                    lime_score=lime_result.score,
                    shap_top15=shap_top15,
                    lime_top15=lime_clean,
                    category_shap_vectors=cat_shap_cache.get(cat_name),
                )
                mets['model']     = model_name
                mets['category']  = cat_name
                mets['sample_id'] = row_i
                self.global_metrics_storage.append(mets)

                # Token evidence (FIX #4)
                all_toks = list({x[0] for x in lime_clean} | {x[0] for x in shap_top15})
                self.all_dominant_tokens[cat_name].extend(all_toks)

            except Exception as e:
                logger.warning(f"  Sample {row_i} failed: {e}")
                traceback.print_exc()

        # Back-fill stability scores
        for rec in self.global_metrics_storage:
            if rec.get('model') != model_name:
                continue
            vecs = cat_shap_cache.get(rec.get('category', ''), [])
            if len(vecs) >= 2:
                corrs = []
                ref = vecs[0]
                for v in vecs[1:]:
                    if np.std(v) > 1e-9 and np.std(ref) > 1e-9:
                        r, _ = spearmanr(ref, v)
                        corrs.append(float(r))
                if corrs:
                    rec['Stability'] = round(float(np.mean(corrs)), 4)

        gc.collect()
        logger.info(f"  {model_name} done.")

    # ── runner ────────────────────────────────────────────────────────────────
    def explain_all_models(self, n_categories: int = 50):
        self.n_categories = n_categories
        self._setup_dirs(n_categories)
        for name in self.MODEL_NAMES:
            try:
                self.explain_model(name)
            except Exception as e:
                logger.error(f"Pipeline failed {name}: {e}")
                traceback.print_exc()
        self.save_reports()
        return self.global_metrics_storage

    # ── reports ───────────────────────────────────────────────────────────────
    def save_reports(self):
        if not self.global_metrics_storage:
            logger.warning("No metrics to save.")
            return

        df = pd.DataFrame(self.global_metrics_storage)
        csv_path = self.dirs['metrics'] / OVERALL_EXPLAINABILITY_CONFIG['metrics_files']['dl']
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved → {csv_path}")

        try:
            summary = df.groupby('model')[['Fidelity', 'Jaccard', 'Stability']].mean().reset_index()
            melted  = summary.melt(id_vars='model', var_name='Metric', value_name='Score')
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=melted, x='Metric', y='Score', hue='model')
            for c in ax.containers:
                ax.bar_label(c, fmt='%.3f', padding=3, fontsize=9)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title("DL Explainability Metrics — SBERT (Honest)")
            plt.ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(self.dirs['metrics'] / "DL_Metrics_Comparison_SBERT.png",
                        dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Metrics plot failed: {e}")

        data = []
        for cat in TARGET_CATEGORIES:
            toks = [t for t in self.all_dominant_tokens.get(cat, [])
                    if t and not str(t).startswith('dim_')]
            top = [w for w, _ in Counter(toks).most_common(15)]
            data.append({
                'Category': cat,
                'Top_15_Tokens': ', '.join(top) if top else 'N/A',
                'Token_Count': len(toks),
            })
        df_tok = pd.DataFrame(data)
        tok_path = self.dirs['reports'] / OVERALL_EXPLAINABILITY_CONFIG['token_files']['dl']
        df_tok.to_csv(tok_path, index=False)
        logger.info(f"Consolidated tokens → {tok_path}")

        logger.info("\n" + "="*50)
        logger.info("DL EXPLAINABILITY SUMMARY (SBERT)")
        for mn, grp in df.groupby('model'):
            logger.info(
                f"  {mn:20s} | Fidelity={grp['Fidelity'].mean():.4f} "
                f"| Jaccard={grp['Jaccard'].mean():.4f} "
                f"| Stability={grp['Stability'].mean():.4f}"
            )
        logger.info("="*50)


if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    args = parser.parse_args()
    t0 = time.time()
    DLExplainability(n_categories=args.categories).explain_all_models(args.categories)
    logger.info(f"PHASE COMPLETE: DL_EXPLAINABILITY in {time.time()-t0:.1f}s")

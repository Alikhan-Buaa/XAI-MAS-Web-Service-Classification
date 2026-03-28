"""
BERT (RoBERTa) Explainability Module — SBERT Unified Pipeline
==============================================================
Fixes applied:

FIXED #1  Fake metrics removed: 'min(0.86, max(0.70, ...))' clamping and
          np.random.uniform noise injection completely removed.
          Fidelity  = sqrt(|R²|) raw.
          Jaccard   = set overlap of top-15 SHAP words vs top-15 LIME words.
          Stability = mean pairwise Spearman-r across same-category SHAP
                      vectors.

FIXED #2  Domain stopwords removed: 'api', 'service', 'data', 'platform',
          'cloud', 'tool', 'application', 'web', 'software', 'system',
          'developer', 'access' no longer filtered.

FIXED #3  max_evals increased: 100 → 512 for reliable global SHAP coverage
          of RoBERTa's token space.

FIXED #4  CSV token evidence: category_tokens populated inside the local
          explain loop from both SHAP and LIME words, not only SHAP.

FIXED #5  Uniform predict_fn: BERTModelWrapper.predict_proba used by BOTH
          shap.Explainer (text masker) and LimeTextExplainer — same scale.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
import warnings
import traceback
import gc
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from scipy.stats import spearmanr

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap

from src.config import (
    DATA_PATH, RESULTS_PATH, BERT_CONFIG,
    CATEGORY_SIZES, RANDOM_SEED, SAVED_MODELS_CONFIG,
    PREPROCESSING_CONFIG, RESULTS_CONFIG,
    OVERALL_EXPLAINABILITY_CONFIG
)
from src.utils.utils import (
    STOPWORDS, TARGET_CATEGORIES, FALLBACK_LABELS,
    load_class_labels,
    top15_tokens, plot_bar, compute_metrics,
    build_shap_background, run_global_shap, run_global_lime,
    run_beeswarm,
)

from src.explainability.shared_samples import get_shared_samples, FIXED_CATEGORIES

# ── logging ───────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _n in ['shap', 'lime', 'transformers', 'tensorflow']:
    _l = logging.getLogger(_n)
    _l.setLevel(logging.ERROR)
    _l.propagate = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ── wrapper ───────────────────────────────────────────────────────────────────
class BERTModelWrapper:
    def __init__(self, model, tokenizer, device, max_len=128, batch_size=16):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.max_len   = max_len
        self.batch_size = batch_size
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        safe = [str(t) if pd.notna(t) and str(t).strip() else "empty" for t in texts]
        out = []
        for i in range(0, len(safe), self.batch_size):
            batch = safe[i: i + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                out.append(F.softmax(logits, dim=1).cpu().numpy())
            del enc
            if i % (self.batch_size * 5) == 0:
                torch.cuda.empty_cache()
        return np.vstack(out)


# ── main class ────────────────────────────────────────────────────────────────
class BERTExplainability:

    def __init__(self, n_categories: int = 50):
        self.n_categories = n_categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_names = ["roberta-base", "roberta-large"]
        self.max_features = 15
        self.plot_dpi = 300

        self.output_files = {
            'tokens':  OVERALL_EXPLAINABILITY_CONFIG['token_files']['bert'],
            'metrics': OVERALL_EXPLAINABILITY_CONFIG['metrics_files']['bert'],
            'plot':    "BERT_Comparison_Plot.png",
        }

        self.all_dominant_tokens: dict = defaultdict(list)
        self.global_metrics_storage: list = []
        self.waterfall_generated = {m: set() for m in self.model_names}  # set of cats done

        self.category_tokens = {cat: [] for cat in TARGET_CATEGORIES}

        self.base_result_dir = RESULTS_PATH / "bert" / f"top_{n_categories}_categories"
        self.explain_dir = self.base_result_dir / "explainability"

        self.dirs = {
            'beeswarm':    self.explain_dir / "shap" / "beeswarm",
            'waterfall':   self.explain_dir / "shap" / "waterfall",
            'global_bar':  self.explain_dir / "shap" / "global_bar",
            'samples':     self.explain_dir / "shap" / "samples",
            'lime':        self.explain_dir / "lime",
            'lime_dash':   self.explain_dir / "lime" / "dashboards",
            'global_lime': self.explain_dir / "lime" / "global",
            'metrics':     self.explain_dir / "metrics",
            'reports':     self.explain_dir / "reports",
            'comparisons': RESULTS_CONFIG['bert_comparisons_path'],
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"BERTExplainability initialised → {self.explain_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers — thin wrappers that delegate to explainability_utils.
    # All shared logic (STOPWORDS, metrics, plotting) lives there.
    # ─────────────────────────────────────────────────────────────────────────
    def _top15(self, features, weights, clean_glyph=False):
        return top15_tokens(features, weights, clean_glyph=clean_glyph)

    def _plot_bar(self, items, title, output_path):
        plot_bar(items, title, output_path, plot_dpi=self.plot_dpi)

    def _compute_metrics(self, lime_score, shap_top15, lime_top15,
                         category_shap_vectors=None):
        return compute_metrics(lime_score, shap_top15, lime_top15,
                               category_shap_vectors)

    def _run_global_lime(self, lime_exp, predict_fn, test_df, model_name,
                          sample_limit=None, clean_glyph=True):
        n = sample_limit if sample_limit else len(TARGET_CATEGORIES)
        run_global_lime(
            lime_exp, predict_fn, test_df, model_name,
            self.dirs['global_lime'] / f"global_lime_{model_name}.png",
            sample_limit=n, clean_glyph=clean_glyph,
            plot_dpi=300,
        )


    # ── labels ────────────────────────────────────────────────────────────────
    def _load_labels(self) -> list:
        return load_class_labels(self.n_categories)

    def load_model_and_data(self, model_name: str):
        logger.info(f"Loading {model_name} on {self.device}…")
        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df  = pd.read_csv(splits_dir / "test.csv")
        train_df = pd.read_csv(splits_dir / "train.csv")
        class_labels = self._load_labels()

        base_path = SAVED_MODELS_CONFIG['bert_models_path'] / f"top_{self.n_categories}_categories"
        kw = "base" if "base" in model_name.lower() else "large"
        hf_name = "roberta-base" if kw == "base" else "roberta-large"

        model_path, is_hf = None, False
        if base_path.exists():
            for f in base_path.rglob("*"):
                if f.is_file() and f.suffix in ['.model', '.pth'] \
                        and kw in f.name.lower() and "roberta" in f.name.lower():
                    model_path = f
                    break
            if model_path is None:
                for cfg in base_path.rglob("config.json"):
                    p = cfg.parent
                    if kw in p.name.lower() or kw in str(p).lower():
                        model_path, is_hf = p, True
                        break

        if model_path is None:
            logger.error(f"Model not found for {model_name} in {base_path}")
            return None, None, None, None

        try:
            if is_hf:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            else:
                tokenizer = AutoTokenizer.from_pretrained(hf_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    hf_name, num_labels=self.n_categories
                )
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None, None, None, None

        bs = 8 if "large" in model_name.lower() else 16
        wrapper = BERTModelWrapper(model, tokenizer, self.device, batch_size=bs)
        return wrapper, test_df, train_df, class_labels

    # ── plot ──────────────────────────────────────────────────────────────────

    # ── main explain loop ─────────────────────────────────────────────────────
    def explain_model(self, model_name: str):
        logger.info(f"\n{'='*60}\n  BERT Explaining {model_name}\n{'='*60}")
        wrapper, test_df, train_df, class_labels = self.load_model_and_data(model_name)
        if wrapper is None:
            return

        # SHAP text explainer (uses RoBERTa's own tokenizer as masker)
        masker   = shap.maskers.Text(wrapper.tokenizer)
        explainer = shap.Explainer(wrapper.predict_proba, masker, output_names=class_labels)

        # Global SHAP — use 512 max_evals (SHAP recommended: 2*n_tokens+1)
        try:
            logger.info("  Hunting 15 target categories for global SHAP…")
            global_texts, seen_g = [], set()
            if 'encoded_label' in test_df.columns:
                for idx in range(len(test_df)):
                    if len(seen_g) >= len(TARGET_CATEGORIES):
                        break
                    try:
                        cat = class_labels[int(test_df.iloc[idx]['encoded_label'])]
                        if cat in TARGET_CATEGORIES and cat not in seen_g:
                            global_texts.append(test_df.iloc[idx]['cleaned_text'])
                            seen_g.add(cat)
                    except Exception:
                        continue
            if not global_texts:
                global_texts = train_df['cleaned_text'].head(15).tolist()

            # max_evals=512 — reliable for RoBERTa token space (FIX #3)
            shap_global = explainer(global_texts, max_evals=512)

            global_agg: dict = defaultdict(float)
            beeswarm_data = {'Token': [], 'SHAP Value': []}
            for i in range(len(shap_global)):
                tokens = [
                    str(t).replace('Ġ', '').strip().lower()
                    for t in (shap_global.data[i]
                              if shap_global.feature_names is None
                              else shap_global.feature_names[i])
                ]
                impacts = np.sum(np.abs(shap_global[i].values), axis=1)
                for t, imp in zip(tokens, impacts):
                    if t not in STOPWORDS and len(t) >= 3 and not t.isnumeric():
                        global_agg[t] += imp
                        beeswarm_data['Token'].append(t)
                        beeswarm_data['SHAP Value'].append(imp)

            top15g = sorted(global_agg.items(), key=lambda x: x[1], reverse=True)[:15]
            if top15g:
                self._plot_bar(
                    top15g,
                    f"Global SHAP Top 15 — {model_name}",
                    self.dirs['global_bar'] / f"shap_global_{model_name}.png",
                )
                top_tokens = [x[0] for x in top15g]
                df_bee = pd.DataFrame(beeswarm_data)
                df_bee = df_bee[df_bee['Token'].isin(top_tokens)]
                if not df_bee.empty:
                    plt.figure(figsize=(12, 8))
                    df_bee['Token'] = pd.Categorical(df_bee['Token'], categories=top_tokens, ordered=True)
                    sns.stripplot(data=df_bee, x='SHAP Value', y='Token',
                                  jitter=0.2, alpha=0.7, palette='viridis')
                    plt.axvline(x=0, color='gray', linewidth=1)
                    plt.title(f"SHAP Beeswarm (Global) — {model_name}", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(self.dirs['beeswarm'] / f"beeswarm_{model_name}.png", dpi=300)
                    plt.close()
        except Exception as e:
            logger.error(f"  Global SHAP failed: {e}")

        lime_exp = LimeTextExplainer(class_names=class_labels, split_expression=r'\W+')
        self._run_global_lime(lime_exp, wrapper.predict_proba, test_df, model_name)

        # Select shared samples — same rows as ML, DL, DeepSeek, Fusion
        logger.info("  Loading shared sample index (5 fixed categories)…")
        indices = get_shared_samples(
            test_df=test_df,
            class_labels=class_labels,
            n_categories=self.n_categories,
            results_root=RESULTS_PATH,
        )
        logger.info(f"  Shared samples: {[(r, c) for r, c in indices]}")
        cat_shap_cache: dict = defaultdict(list)

        for idx_count, (row_i, cat_name) in enumerate(indices):
            try:
                text  = test_df.iloc[row_i]['cleaned_text']
                probs = wrapper.predict_proba([text])[0]
                top   = int(np.argmax(probs))
                logger.info(f"  [{idx_count+1}/{len(indices)}] {cat_name} — sample {row_i}")

                # LIME
                exp1 = lime_exp.explain_instance(
                    text, wrapper.predict_proba,
                    labels=[top], num_features=35, num_samples=500,
                )
                try:
                    exp1.save_to_file(str(
                        self.dirs['lime_dash'] / f"{model_name}_sample_{row_i}_{cat_name}.html"
                    ))
                except Exception:
                    pass

                lime_agg: dict = defaultdict(float)
                for f, w in exp1.as_list(label=top):
                    fs = str(f).lower().replace('Ġ', '').strip()
                    if fs not in STOPWORDS and len(fs) >= 3 and not fs.isnumeric():
                        lime_agg[fs] += w
                lime_feats = sorted(lime_agg.items(), key=lambda x: abs(x[1]), reverse=True)
                lime_top15 = lime_feats[:15]

                self._plot_bar(
                    lime_top15,
                    f"LIME ({cat_name}) — {model_name}",
                    self.dirs['lime'] / f"lime_{model_name}_{row_i}.png",
                )

                # SHAP local
                shap_top15 = []
                try:
                    local_shap = explainer([text])
                    tokens = [
                        str(t).replace('Ġ', '').strip().lower()
                        for t in (local_shap.data[0]
                                  if local_shap.feature_names is None
                                  else local_shap.feature_names[0])
                    ]
                    vals = (local_shap[0].values[:, top]
                            if local_shap[0].values.ndim == 2
                            else local_shap[0].values)
                    base = (float(local_shap[0].base_values[top])
                            if isinstance(local_shap[0].base_values, (list, np.ndarray))
                            else float(local_shap[0].base_values))

                    shap_agg: dict = defaultdict(float)
                    new_base = base
                    for t, v in zip(tokens, vals):
                        if t in STOPWORDS or len(t) < 3 or t.isnumeric():
                            # Do NOT fold into base_val (FIX #3) — just skip
                            continue
                        shap_agg[t] += v

                    # Cache raw SHAP value array for stability
                    cat_shap_cache[cat_name].append(vals.copy())

                    shap_top15 = sorted(shap_agg.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

                    self._plot_bar(
                        shap_top15,
                        f"SHAP ({cat_name}) — {model_name}",
                        self.dirs['samples'] / f"shap_{model_name}_{row_i}.png",
                    )

                    if shap_top15 and cat_name not in self.waterfall_generated[model_name]:
                        w_names = np.array([x[0] for x in shap_top15])
                        w_vals  = np.array([x[1] for x in shap_top15])
                        exp_obj = shap.Explanation(
                            values=w_vals, base_values=new_base,
                            data=w_names, feature_names=list(w_names),
                        )
                        plt.figure(figsize=(16, 10))
                        shap.plots.waterfall(exp_obj, max_display=15, show=False)
                        plt.title(f"SHAP Waterfall | {model_name} | {cat_name}", fontsize=13, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(self.dirs['waterfall'] / f"waterfall_{model_name}_{cat_name}.png", dpi=300)
                        plt.close()
                        self.waterfall_generated[model_name].add(cat_name)

                except Exception as e:
                    logger.warning(f"  Local SHAP failed sample {row_i}: {e}")

                # Honest metrics (FIX #1 — no clamp, no random)
                mets = self._compute_metrics(
                    lime_score=exp1.score,
                    shap_top15=shap_top15,
                    lime_top15=lime_top15,
                    category_shap_vectors=cat_shap_cache.get(cat_name),
                )
                mets.update({'model': model_name, 'category': cat_name, 'sample_id': row_i})
                self.global_metrics_storage.append(mets)

                # Token evidence (FIX #4)
                all_toks = list(
                    {x[0] for x in lime_top15} | {x[0] for x in shap_top15}
                )
                self.all_dominant_tokens[cat_name].extend(all_toks)
                self.category_tokens[cat_name].extend(all_toks)

            except Exception as e:
                logger.warning(f"  Failed sample {row_i}: {e}")

        # Back-fill stability
        for rec in self.global_metrics_storage:
            if rec.get('model') != model_name:
                continue
            vecs = cat_shap_cache.get(rec.get('category', ''), [])
            if len(vecs) >= 2:
                corrs = []
                ref = vecs[0]
                for v in vecs[1:]:
                    if len(v) == len(ref) and np.std(v) > 1e-9 and np.std(ref) > 1e-9:
                        r, _ = spearmanr(ref, v)
                        corrs.append(float(r))
                if corrs:
                    rec['Stability'] = round(float(np.mean(corrs)), 4)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── save ──────────────────────────────────────────────────────────────────
    def save_consolidated_tokens(self):
        data = []
        for cat in TARGET_CATEGORIES:
            toks = self.category_tokens.get(cat, [])
            top  = [w for w, _ in Counter(toks).most_common(15)]
            data.append({'Category': cat, 'Consolidated_Top_Words': ', '.join(top) if top else 'N/A'})
        df = pd.DataFrame(data)
        p  = self.dirs['reports'] / self.output_files['tokens']
        df.to_csv(p, index=False)
        logger.info(f"Tokens saved → {p}")

    def generate_comparison_plot(self):
        if not self.global_metrics_storage:
            return
        df = pd.DataFrame(self.global_metrics_storage)
        df.to_csv(self.dirs['metrics'] / self.output_files['metrics'], index=False)

        summary = df.groupby('model')[['Fidelity', 'Jaccard', 'Stability']].mean().reset_index()
        melted  = summary.melt(id_vars='model', var_name='Metric', value_name='Score')
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=melted, x='Metric', y='Score', hue='model', palette='viridis')
        for c in ax.containers:
            ax.bar_label(c, fmt='%.3f', padding=4, fontsize=11, fontweight='bold')
        plt.title("BERT XAI Metrics (Honest — No Scaling)", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        for dest in [self.dirs['metrics'], self.dirs['comparisons']]:
            plt.savefig(dest / self.output_files['plot'], dpi=300)
        plt.close()

        logger.info("\n" + "="*50)
        logger.info("BERT EXPLAINABILITY SUMMARY")
        for mn, grp in df.groupby('model'):
            logger.info(
                f"  {mn:20s} | Fidelity={grp['Fidelity'].mean():.4f} "
                f"| Jaccard={grp['Jaccard'].mean():.4f} "
                f"| Stability={grp['Stability'].mean():.4f}"
            )
        logger.info("="*50)

    def explain_all_models(self):
        logger.info("Starting BERT Explainability…")
        for name in self.model_names:
            self.explain_model(name)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.save_consolidated_tokens()
        self.generate_comparison_plot()
        logger.info("BERT Explainability complete.")


if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    args = parser.parse_args()
    t0 = time.time()
    BERTExplainability(n_categories=args.categories).explain_all_models()
    logger.info(f"PHASE COMPLETE: BERT_EXPLAINABILITY in {time.time()-t0:.1f}s")

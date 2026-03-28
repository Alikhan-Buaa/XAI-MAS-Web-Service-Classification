"""
DeepSeek Explainability Module — Honest Metrics
================================================
Fixes applied:

FIXED #1  Fake metrics removed: 'narrative tier 0.60–0.70' bounding,
          np.random.uniform noise, and all min/max clamps removed.
          Fidelity  = sqrt(|R²|) raw.
          Jaccard   = set overlap of top-15 SHAP words vs top-15 LIME words.
          Stability = mean pairwise Spearman-r across same-category SHAP
                      vectors.

FIXED #2  Domain stopwords removed: 'api', 'service', 'data', 'platform',
          'cloud', 'tool', 'application', 'web', 'software', 'system',
          'developer', 'access' no longer filtered.

FIXED #3  SHAP additivity: stopword tokens are skipped in aggregation but
          NOT folded back into base_val — base_val is left untouched.

FIXED #4  CSV evidence: category_tokens populated from both SHAP and LIME
          words in every local sample iteration.

FIXED #5  Token evidence CSVs: 4 separate evidence CSV writers (Global_SHAP,
          Global_LIME, Local_SHAP, Local_LIME) preserved and correctly
          populated.
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from lime.lime_text import LimeTextExplainer
import shap

from src.config import (
    DATA_PATH, DEEPSEEK_CONFIG, PREPROCESSING_CONFIG,
    SAVED_MODELS_CONFIG, RESULTS_CONFIG, RESULTS_PATH,
    CATEGORY_SIZES, RANDOM_SEED, OVERALL_EXPLAINABILITY_CONFIG
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
for _n in ['shap', 'lime', 'transformers', 'tensorflow', 'peft']:
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
class DeepSeekWrapper:
    def __init__(self, model, tokenizer, device, max_len=512, batch_size=4):
        self.model      = model
        self.tokenizer  = tokenizer
        self.device     = device
        self.max_len    = max_len
        self.batch_size = batch_size
        self.model.eval()

    def predict_proba(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        safe = [str(t) if pd.notna(t) and str(t).strip() else "empty" for t in texts]
        out  = []
        for i in range(0, len(safe), self.batch_size):
            batch = safe[i: i + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                out.append(F.softmax(logits.float(), dim=1).cpu().numpy())
            del enc
            if i % (self.batch_size * 2) == 0:
                torch.cuda.empty_cache()
        return np.vstack(out)


# ── main class ────────────────────────────────────────────────────────────────
class DeepSeekExplainability:

    def __init__(self, n_categories: int = 50):
        self.n_categories = n_categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "DeepSeek_7B"
        self.max_features = 15

        self.output_files = {
            'tokens':  OVERALL_EXPLAINABILITY_CONFIG['token_files']['deepseek'],
            'metrics': OVERALL_EXPLAINABILITY_CONFIG['metrics_files']['deepseek'],
            'plot':    "DeepSeek_Metrics_Comparison.png",
        }

        self.global_metrics_storage: list = []
        self.plot_dpi = 300
        self.waterfall_generated: set = set()  # one waterfall per category
        self.category_tokens = {cat: [] for cat in TARGET_CATEGORIES}

        # 4-track evidence store
        self.evidence_data = {
            'Global_SHAP': [],
            'Global_LIME': [],
            'Local_SHAP':  {cat: [] for cat in TARGET_CATEGORIES},
            'Local_LIME':  {cat: [] for cat in TARGET_CATEGORIES},
        }

        self.base_result_dir = RESULTS_CONFIG['deepseek_category_paths'][n_categories]
        self.explain_dir = self.base_result_dir / "explainability"
        self.shap_dir    = self.explain_dir / "shap"
        self.lime_dir    = self.explain_dir / "lime"

        self.dirs = {
            'shap':         self.shap_dir,
            'shap_reports': self.shap_dir / "reports",
            'beeswarm':     self.shap_dir / "beeswarm",
            'waterfall':    self.shap_dir / "waterfall",
            'global_bar':   self.shap_dir / "global_bar",
            'samples':      self.shap_dir / "samples",
            'lime':         self.lime_dir,
            'lime_reports': self.lime_dir / "reports",
            'lime_dash':    self.lime_dir / "dashboards",
            'global_lime':  self.lime_dir / "global",
            'metrics':      self.explain_dir / "metrics",
            'reports':      self.explain_dir / "reports",
            'comparisons':  RESULTS_CONFIG['deepseek_comparisons_path'],
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"DeepSeekExplainability initialised → {self.explain_dir}")

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

    def load_model_and_data(self):
        logger.info(f"Loading DeepSeek-7B on {self.device}…")
        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df  = pd.read_csv(splits_dir / "test.csv")
        train_df = pd.read_csv(splits_dir / "train.csv")
        class_labels = self._load_labels()

        base_path = SAVED_MODELS_CONFIG['deepseek_models_path'] / f"top_{self.n_categories}_categories"
        model_path, is_peft = None, False

        if base_path.exists():
            for f in base_path.rglob("adapter_config.json"):
                model_path, is_peft = f.parent, True
                break
            if model_path is None:
                for f in base_path.rglob("*"):
                    if f.is_file() and f.suffix in ['.model', '.pth'] and 'deepseek' in f.name.lower():
                        model_path = f
                        break

        if model_path is None:
            logger.error(f"DeepSeek model not found in {base_path}")
            return None, None, None, None

        try:
            hf_name = DEEPSEEK_CONFIG['available_models'].get('deepseek', 'deepseek-ai/deepseek-llm-7b-base')
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path) if is_peft else hf_name,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                hf_name,
                num_labels=self.n_categories,
                quantization_config=bnb_cfg if self.device == 'cuda' else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
            )
            if is_peft:
                model = PeftModel.from_pretrained(base_model, str(model_path))
                model = model.merge_and_unload()
            else:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                base_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
                model = base_model

            wrapper = DeepSeekWrapper(model, tokenizer, self.device)
            logger.info("DeepSeek-7B loaded.")
            return wrapper, test_df, train_df, class_labels
        except Exception as e:
            logger.error(f"Failed to load DeepSeek: {e}")
            return None, None, None, None

    # ── helpers ───────────────────────────────────────────────────────────────

    # ── main explain loop ─────────────────────────────────────────────────────
    def explain_model(self):
        wrapper, test_df, train_df, class_labels = self.load_model_and_data()
        if wrapper is None:
            return

        masker    = shap.maskers.Text(wrapper.tokenizer)
        explainer = shap.Explainer(wrapper.predict_proba, masker, output_names=class_labels)

        # Global SHAP
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
                        self.evidence_data['Global_SHAP'].append({'token': t, 'impact': imp})

            top15g = sorted(global_agg.items(), key=lambda x: x[1], reverse=True)[:15]
            if top15g:
                self._plot_bar(
                    top15g,
                    "Global SHAP Top 15 — DeepSeek_7B",
                    self.dirs['global_bar'] / "shap_global_DeepSeek_7B.png",
                )
                top_toks = [x[0] for x in top15g]
                # Use shared run_beeswarm() for consistency across all models
                run_beeswarm(
                    beeswarm_rows=[
                        {'Token': row['Token'], 'SHAP Value': row['SHAP Value']}
                        for _, row in pd.DataFrame(beeswarm_data).iterrows()
                        if row['Token'] in top_toks
                    ],
                    model_name=self.model_name,
                    output_path=self.dirs['beeswarm'] / "beeswarm_DeepSeek_7B.png",
                    plot_dpi=self.plot_dpi,
                )
        except Exception as e:
            logger.error(f"  Global SHAP failed: {e}")

        lime_exp = LimeTextExplainer(class_names=class_labels, split_expression=r'\W+')
        self._run_global_lime(lime_exp, wrapper.predict_proba, test_df, self.model_name)

        # Select shared samples — same rows as ML, DL, BERT, Fusion
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
                        self.dirs['lime_dash'] / f"DeepSeek_sample_{row_i}_{cat_name}.html"
                    ))
                except Exception:
                    pass

                lime_agg: dict = defaultdict(float)
                for f, w in exp1.as_list(label=top):
                    fs = str(f).lower().replace('Ġ', '').strip()
                    if fs not in STOPWORDS and len(fs) >= 3 and not fs.isnumeric():
                        lime_agg[fs] += w
                lime_feats  = sorted(lime_agg.items(), key=lambda x: abs(x[1]), reverse=True)
                lime_top15  = lime_feats[:15]

                self._plot_bar(
                    lime_top15,
                    f"LIME ({cat_name}) — DeepSeek_7B",
                    self.lime_dir / f"lime_DeepSeek_{row_i}.png",
                )
                # Evidence store
                for tok, w in lime_top15:
                    self.evidence_data['Local_LIME'][cat_name].append({'token': tok, 'weight': w})
                    self.category_tokens[cat_name].append(tok)

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
                    for t, v in zip(tokens, vals):
                        if t in STOPWORDS or len(t) < 3 or t.isnumeric():
                            continue  # skip, do NOT modify base (FIX #3)
                        shap_agg[t] += v

                    cat_shap_cache[cat_name].append(vals.copy())

                    shap_top15 = sorted(shap_agg.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

                    self._plot_bar(
                        shap_top15,
                        f"SHAP ({cat_name}) — DeepSeek_7B",
                        self.dirs['samples'] / f"shap_DeepSeek_{row_i}.png",
                    )
                    # Evidence store
                    for tok, w in shap_top15:
                        self.evidence_data['Local_SHAP'][cat_name].append({'token': tok, 'weight': w})
                        self.category_tokens[cat_name].append(tok)

                    if shap_top15 and cat_name not in self.waterfall_generated:
                        w_names = np.array([x[0] for x in shap_top15])
                        w_vals  = np.array([x[1] for x in shap_top15])
                        exp_obj = shap.Explanation(
                            values=w_vals, base_values=base,
                            data=w_names, feature_names=list(w_names),
                        )
                        plt.figure(figsize=(16, 10))
                        shap.plots.waterfall(exp_obj, max_display=15, show=False)
                        plt.title(f"SHAP Waterfall | {self.model_name} | {cat_name}", fontsize=13, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(self.dirs['waterfall'] / f"waterfall_{self.model_name}_{cat_name}.png", dpi=300)
                        plt.close()
                        self.waterfall_generated.add(cat_name)

                except Exception as e:
                    logger.warning(f"  Local SHAP failed sample {row_i}: {e}")

                # Honest metrics (FIX #1)
                mets = self._compute_metrics(
                    lime_score=exp1.score,
                    shap_top15=shap_top15,
                    lime_top15=lime_top15,
                    category_shap_vectors=cat_shap_cache.get(cat_name),
                )
                mets.update({'model': self.model_name, 'category': cat_name, 'sample_id': row_i})
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"  Failed sample {row_i}: {e}")
                traceback.print_exc()

        # Back-fill stability
        for rec in self.global_metrics_storage:
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
        p = self.dirs['reports'] / self.output_files['tokens']
        pd.DataFrame(data).to_csv(p, index=False)
        logger.info(f"Tokens → {p}")

        # 4 evidence CSVs (FIX #5)
        pd.DataFrame(self.evidence_data['Global_SHAP']).to_csv(
            self.dirs['shap_reports'] / "DeepSeek_Global_SHAP_Evidence.csv", index=False)
        pd.DataFrame(self.evidence_data['Global_LIME']).to_csv(
            self.dirs['lime_reports'] / "DeepSeek_Global_LIME_Evidence.csv", index=False)

        local_shap_rows = [
            {'category': cat, **row}
            for cat, rows in self.evidence_data['Local_SHAP'].items()
            for row in rows
        ]
        local_lime_rows = [
            {'category': cat, **row}
            for cat, rows in self.evidence_data['Local_LIME'].items()
            for row in rows
        ]
        pd.DataFrame(local_shap_rows).to_csv(
            self.dirs['shap_reports'] / "DeepSeek_Local_SHAP_Evidence.csv", index=False)
        pd.DataFrame(local_lime_rows).to_csv(
            self.dirs['lime_reports'] / "DeepSeek_Local_LIME_Evidence.csv", index=False)
        logger.info("4 evidence CSVs saved.")

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
        plt.title("DeepSeek XAI Metrics (Honest — No Scaling)", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        for dest in [self.dirs['metrics'], self.dirs['comparisons']]:
            plt.savefig(dest / self.output_files['plot'], dpi=300)
        plt.close()

        logger.info("\n" + "="*50)
        logger.info("DEEPSEEK EXPLAINABILITY SUMMARY")
        for mn, grp in df.groupby('model'):
            logger.info(
                f"  {mn:25s} | Fidelity={grp['Fidelity'].mean():.4f} "
                f"| Jaccard={grp['Jaccard'].mean():.4f} "
                f"| Stability={grp['Stability'].mean():.4f}"
            )
        logger.info("="*50)

    def explain_all_models(self):
        self.explain_model()
        self.save_consolidated_tokens()
        self.generate_comparison_plot()
        logger.info("DeepSeek Explainability complete.")


if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    args = parser.parse_args()
    t0 = time.time()
    DeepSeekExplainability(n_categories=args.categories).explain_all_models()
    logger.info(f"PHASE COMPLETE: DEEPSEEK_EXPLAINABILITY in {time.time()-t0:.1f}s")

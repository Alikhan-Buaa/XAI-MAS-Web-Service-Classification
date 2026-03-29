"""
BERT (RoBERTa) Explainability Module
Thin caller — all shared logic lives in src/utils/explainability_utils.py
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
import warnings
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from pathlib import Path
from collections import defaultdict, Counter
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap

from src.config import (
    DATA_PATH, RESULTS_PATH, BERT_CONFIG,
    CATEGORY_SIZES, RANDOM_SEED, SAVED_MODELS_CONFIG,
    PREPROCESSING_CONFIG, RESULTS_CONFIG,
    OVERALL_EXPLAINABILITY_CONFIG
)
from src.utils.explainability_utils import (
    STOPWORDS, TARGET_CATEGORIES,
    load_class_labels, get_shared_samples,
    top15_tokens, plot_bar, compute_metrics,
    run_global_lime, run_beeswarm, run_waterfall,
    save_metrics_report,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _nl in ['shap', 'lime', 'transformers', 'tensorflow']:
    logging.getLogger(_nl).setLevel(logging.ERROR)
    logging.getLogger(_nl).propagate = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BERTModelWrapper:
    def __init__(self, model, tokenizer, device, max_len=128, batch_size=16):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts):
        if isinstance(texts, np.ndarray): texts = texts.tolist()
        safe = [str(t) if pd.notna(t) and str(t).strip() != "" else "empty text" for t in texts]
        all_probs = []
        for i in range(0, len(safe), self.batch_size):
            inputs = self.tokenizer(
                safe[i:i+self.batch_size], padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.model(**inputs).logits, dim=1).cpu().numpy()
                all_probs.append(probs)
            del inputs
            if i % (self.batch_size * 5) == 0: torch.cuda.empty_cache()
        return np.vstack(all_probs)


class BERTExplainability:
    def __init__(self, n_categories=50):
        self.n_categories = n_categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_names = ["roberta-base", "roberta-large"]
        self.max_features = 15
        self.output_files = {
            'tokens':  OVERALL_EXPLAINABILITY_CONFIG['token_files']['bert'],
            'metrics': OVERALL_EXPLAINABILITY_CONFIG['metrics_files']['bert'],
            'plot':    "BERT_Comparison_Plot.png",
        }
        self.all_dominant_tokens = defaultdict(dict)
        self.global_metrics_storage = []
        self.waterfall_generated = {m: False for m in self.model_names}
        self.target_categories = TARGET_CATEGORIES  # from utils
        self.category_tokens   = {cat: [] for cat in self.target_categories}

        self.base_result_dir = RESULTS_PATH / "bert" / f"top_{n_categories}_categories"
        self.explain_dir     = self.base_result_dir / "explainability"
        self.shap_dir        = self.explain_dir / "shap"
        self.lime_dir        = self.explain_dir / "lime"
        self.dirs = {
            'beeswarm':   self.shap_dir / "beeswarm",
            'waterfall':  self.shap_dir / "waterfall",
            'global_bar': self.shap_dir / "global_bar",
            'samples':    self.shap_dir / "samples",
            'lime':       self.lime_dir,
            'lime_dash':  self.lime_dir / "lime_dashboards",
            'global_lime':self.lime_dir / "global",
            'metrics':    self.explain_dir / "metrics",
            'reports':    self.explain_dir / "reports",
            'comparisons':RESULTS_CONFIG['bert_comparisons_path'],
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def load_model_and_data(self, model_name):
        splits_dir   = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df      = pd.read_csv(splits_dir / "test.csv")
        train_df     = pd.read_csv(splits_dir / "train.csv")
        class_labels = load_class_labels(self.n_categories)   # from utils

        base_path      = SAVED_MODELS_CONFIG['bert_models_path'] / f"top_{self.n_categories}_categories"
        target_keyword = "base" if "base" in model_name.lower() else "large"
        hf_model_name  = "roberta-base" if "base" in model_name.lower() else "roberta-large"
        model_path, is_hf_dir = None, False

        if base_path.exists():
            for f in base_path.rglob("*"):
                if f.is_file() and f.suffix in ['.model', '.pth'] \
                        and target_keyword in f.name.lower() and "roberta" in f.name.lower():
                    model_path = f
                    break
            if model_path is None:
                for cfg in base_path.rglob("config.json"):
                    p = cfg.parent
                    if target_keyword in p.name.lower() or target_keyword in str(p).lower():
                        model_path, is_hf_dir = p, True
                        break

        if model_path is None:
            logger.error(f"CRITICAL: Could not find saved model for {model_name}")
            return None, None, None, None

        try:
            if is_hf_dir:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            else:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                model     = AutoModelForSequenceClassification.from_pretrained(
                    hf_model_name, num_labels=self.n_categories)
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None, None, None, None

        bs      = 8 if "large" in model_name.lower() else 16
        wrapper = BERTModelWrapper(model, tokenizer, self.device, batch_size=bs)
        return wrapper, test_df, train_df, class_labels

    # ── Thin plot wrapper ──────────────────────────────────────────────────────
    def _plot_manual_bar(self, features, weights, title, output_path):
        items = [(str(f), float(w) if not hasattr(w, 'item') else w.item())
                 for f, w in zip(features, weights)
                 if f and not str(f).startswith("dim_")
                 and not str(f).isnumeric() and len(str(f)) >= 2]
        plot_bar(items, title, Path(output_path), plot_dpi=300)

    def explain_model(self, model_name):
        wrapper, test_df, train_df, class_labels = self.load_model_and_data(model_name)
        if wrapper is None:
            return

        masker   = shap.maskers.Text(wrapper.tokenizer)
        explainer = shap.Explainer(wrapper.predict_proba, masker, output_names=class_labels)

        # ── Global SHAP + beeswarm ────────────────────────────────────────────
        try:
            global_texts, seen_for_global = [], set()
            if 'encoded_label' in test_df.columns:
                for idx in range(len(test_df)):
                    if len(seen_for_global) >= len(self.target_categories): break
                    cat = class_labels[test_df.iloc[idx]['encoded_label']]
                    if cat in self.target_categories and cat not in seen_for_global:
                        global_texts.append(test_df.iloc[idx]['cleaned_text'])
                        seen_for_global.add(cat)
            if not global_texts:
                global_texts = train_df['cleaned_text'].head(15).tolist()

            shap_values_global = explainer(global_texts, max_evals=100)

            global_word_agg = defaultdict(float)
            beeswarm_rows   = []
            for i in range(len(shap_values_global)):
                raw_tokens = [str(t).replace('Ġ', '').strip().lower()
                              for t in (shap_values_global.data[i]
                                        if shap_values_global.feature_names is None
                                        else shap_values_global.feature_names[i])]
                impacts = np.sum(np.abs(shap_values_global[i].values), axis=1)
                for t, imp in zip(raw_tokens, impacts):
                    if t not in STOPWORDS and len(t) >= 3 and not t.isnumeric():
                        global_word_agg[t] += imp
                        beeswarm_rows.append({'Token': t, 'SHAP Value': float(imp)})

            top_15_global = sorted(global_word_agg.items(), key=lambda x: x[1], reverse=True)[:15]
            if top_15_global:
                self._plot_manual_bar(
                    [x[0] for x in top_15_global], [x[1] for x in top_15_global],
                    f"Global SHAP Top 15 — {model_name}",
                    self.dirs['global_bar'] / f"shap_global_{model_name}.png",
                )
                run_beeswarm(beeswarm_rows, model_name,
                             self.dirs['beeswarm'] / f"shap_beeswarm_{model_name}.png",
                             plot_dpi=300)

        except Exception as e:
            logger.error(f"Global SHAP failed: {e}")

        # ── Global LIME (utils) ───────────────────────────────────────────────
        lime_explainer = LimeTextExplainer(class_names=class_labels, split_expression=r"\W+")
        run_global_lime(
            lime_explainer, wrapper.predict_proba, test_df, model_name,
            self.dirs['global_lime'] / f"global_lime_{model_name}.png",
            plot_dpi=300, clean_glyph=True,
        )

        # ── Local samples — shared index (utils) ──────────────────────────────
        shared = get_shared_samples(
            test_df=test_df, n_categories=self.n_categories,
            results_root=self.dirs['reports'])

        # shared already returns exactly 1 row per category (5 total) — use directly
        indices_to_explain = list(shared)  # [(row_i, cat_name), ...]

        for i, category_name in indices_to_explain:
            try:
                text      = test_df.iloc[i]['cleaned_text']
                probs     = wrapper.predict_proba([text])[0]
                top_label = int(np.argmax(probs))

                exp = lime_explainer.explain_instance(
                    text, wrapper.predict_proba,
                    num_features=25, labels=[top_label], num_samples=250)

                lime_raw   = exp.as_list(label=top_label)
                lime_clean = top15_tokens(
                    [str(f).replace('Ġ', '').strip() for f, _ in lime_raw],
                    [w for _, w in lime_raw], clean_glyph=True)
                self._plot_manual_bar(
                    [t for t, _ in lime_clean], [w for _, w in lime_clean],
                    f"LIME ({category_name}) — {model_name}",
                    self.dirs['lime'] / f"lime_{model_name}_{i}.png")

                # SHAP local
                try:
                    local_shap = explainer([text])
                    raw_tokens = [str(t).replace('Ġ', '').strip().lower()
                                  for t in (local_shap.data[0]
                                            if local_shap.feature_names is None
                                            else local_shap.feature_names[0])]
                    vals     = (local_shap[0].values[:, top_label]
                                if local_shap[0].values.ndim == 2
                                else local_shap[0].values)
                    base_val = (local_shap[0].base_values[top_label]
                                if isinstance(local_shap[0].base_values, (list, np.ndarray))
                                else local_shap[0].base_values)

                    shap_agg, new_base = defaultdict(float), float(base_val)
                    for t, v in zip(raw_tokens, vals):
                        if t in STOPWORDS or len(t) < 3 or t.isnumeric(): new_base += v
                        else: shap_agg[t] += v

                    shap_clean = top15_tokens(list(shap_agg.keys()),
                                              list(shap_agg.values()), clean_glyph=True)
                    self._plot_manual_bar(
                        [t for t, _ in shap_clean], [w for _, w in shap_clean],
                        f"SHAP ({category_name}) — {model_name}",
                        self.dirs['samples'] / f"shap_{model_name}_{i}.png")

                    # Waterfall — once per model (utils)
                    if not self.waterfall_generated[model_name] and shap_clean:
                        run_waterfall(shap_clean, new_base, model_name, category_name,
                                      self.dirs['waterfall'] / f"waterfall_{model_name}_{i}.png",
                                      plot_dpi=300)
                        self.waterfall_generated[model_name] = True

                    self.category_tokens.setdefault(category_name, []).extend(
                        [t for t, _ in shap_clean])

                except Exception as e:
                    logger.warning(f"SHAP local failed for {i}: {e}")
                    shap_clean = []

                mets = compute_metrics(exp.score, shap_clean, lime_clean)
                mets.update({'model': model_name, 'sample_id': i})
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"Failed sample {i}: {e}")

    def explain_all_models(self):
        for m in self.model_names:
            try: self.explain_model(m)
            except Exception as e: logger.error(f"BERT failed {m}: {e}")
        self.save_reports()

    def save_reports(self):
        save_metrics_report(
            self.global_metrics_storage, model_col='model',
            output_csv=self.dirs['metrics'] / self.output_files['metrics'],
            output_png=self.dirs['metrics'] / self.output_files['plot'],
            title="BERT XAI Metrics Comparison",
        )
        # Also save to comparisons dir
        if self.global_metrics_storage:
            import shutil
            src = self.dirs['metrics'] / self.output_files['plot']
            dst = self.dirs['comparisons'] / self.output_files['plot']
            if src.exists():
                shutil.copy2(src, dst)

        rows = []
        for cat in self.target_categories:
            toks = self.category_tokens.get(cat, [])
            top  = [w for w, _ in Counter(toks).most_common(15)]
            rows.append({'Category': cat,
                         'Consolidated_Top_15_Tokens': ", ".join(top) if top else "N/A"})
        pd.DataFrame(rows).to_csv(
            self.dirs['reports'] / self.output_files['tokens'], index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    args = parser.parse_args()
    BERTExplainability(n_categories=args.categories).explain_all_models()

"""
DeepSeek Explainability Module
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from lime.lime_text import LimeTextExplainer
import shap

from src.config import (
    DATA_PATH, DEEPSEEK_CONFIG, PREPROCESSING_CONFIG,
    SAVED_MODELS_CONFIG, RESULTS_CONFIG, RESULTS_PATH,
    CATEGORY_SIZES, RANDOM_SEED, OVERALL_EXPLAINABILITY_CONFIG
)
from src.utils.explainability_utils import (
    STOPWORDS, TARGET_CATEGORIES,
    load_class_labels, get_shared_samples,
    top15_tokens, plot_bar, compute_metrics,
    run_global_lime, run_beeswarm, run_waterfall,
    save_metrics_report,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _nl in ['shap', 'lime', 'transformers', 'tensorflow', 'peft']:
    logging.getLogger(_nl).setLevel(logging.ERROR)
    logging.getLogger(_nl).propagate = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DeepSeekWrapper:
    def __init__(self, model, tokenizer, device, max_len=512, batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
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
                probs = F.softmax(self.model(**inputs).logits.float(), dim=1).cpu().numpy()
                all_probs.append(probs)
            del inputs
            if i % (self.batch_size * 2) == 0: torch.cuda.empty_cache()
        return np.vstack(all_probs)


class DeepSeekExplainability:
    def __init__(self, n_categories=50):
        self.n_categories = n_categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "DeepSeek_7B"
        self.max_features = 15
        self.output_files = {
            'tokens':  OVERALL_EXPLAINABILITY_CONFIG['token_files']['deepseek'],
            'metrics': OVERALL_EXPLAINABILITY_CONFIG['metrics_files']['deepseek'],
            'plot':    "DeepSeek_Metrics_Comparison.png",
        }
        self.global_metrics_storage = []
        self.target_categories = TARGET_CATEGORIES  # from utils
        self.category_tokens   = {cat: [] for cat in self.target_categories}

        # Evidence trackers (kept for evidence CSVs)
        self.evidence_data = {
            'Global_SHAP': [], 'Global_LIME': [],
            'Local_SHAP':  {cat: [] for cat in self.target_categories},
            'Local_LIME':  {cat: [] for cat in self.target_categories},
        }

        self.base_result_dir = RESULTS_CONFIG['deepseek_category_paths'][n_categories]
        self.explain_dir     = self.base_result_dir / "explainability"
        self.shap_dir        = self.explain_dir / "shap"
        self.lime_dir        = self.explain_dir / "lime"
        self.dirs = {
            'shap':        self.shap_dir,
            'shap_reports':self.shap_dir / "reports",
            'beeswarm':    self.shap_dir / "beeswarm",
            'waterfall':   self.shap_dir / "waterfall",
            'global_bar':  self.shap_dir / "global_bar",
            'samples':     self.shap_dir / "samples",
            'lime':        self.lime_dir,
            'lime_reports':self.lime_dir / "reports",
            'lime_dash':   self.lime_dir / "lime_dashboards",
            'global_lime': self.lime_dir / "global",
            'metrics':     self.explain_dir / "metrics",
            'reports':     self.explain_dir / "reports",
            'comparisons': RESULTS_CONFIG['deepseek_comparisons_path'],
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def load_model_and_data(self):
        splits_dir   = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df      = pd.read_csv(splits_dir / "test.csv")
        train_df     = pd.read_csv(splits_dir / "train.csv")
        class_labels = load_class_labels(self.n_categories)  # from utils

        base_models_path = SAVED_MODELS_CONFIG['deepseek_models_path'] / f"top_{self.n_categories}_categories"
        adapter_candidates = [
            base_models_path / "DeepSeek_7B_Base_RawText_top_50_categories_model.model",
            base_models_path / "DeepSeek_7B_Base_top_50_categories",
            base_models_path / "DeepSeek_7B_Base_top_50_categories" / "checkpoint-final",
        ]
        if base_models_path.exists():
            for p in base_models_path.rglob("adapter_config.json"):
                adapter_candidates.append(p.parent)

        adapter_path = next(
            (c for c in adapter_candidates
             if c.exists() and (c / "adapter_config.json").exists()), None)
        if adapter_path is None:
            logger.error("CRITICAL: adapter_config.json not found.")
            return None, None, None, None

        try:
            base_model_name = DEEPSEEK_CONFIG['models'][0]
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=self.n_categories,
                quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
            base_model.config.pad_token_id = tokenizer.pad_token_id
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        except Exception as e:
            logger.error(f"Failed to load DeepSeek: {e}")
            return None, None, None, None

        return DeepSeekWrapper(model, tokenizer, self.device, batch_size=2), test_df, train_df, class_labels

    def _plot_manual_bar(self, features, weights, title, output_path):
        items = [(str(f), float(w) if not hasattr(w, 'item') else w.item())
                 for f, w in zip(features, weights)
                 if f and not str(f).startswith("dim_")
                 and not str(f).isnumeric() and len(str(f)) >= 2]
        plot_bar(items, title, Path(output_path), plot_dpi=300)

    def explain(self):
        wrapper, test_df, train_df, class_labels = self.load_model_and_data()
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

            shap_vals_global = explainer(global_texts, max_evals=100)
            global_word_agg  = defaultdict(float)
            beeswarm_rows    = []
            for i in range(len(shap_vals_global)):
                raw_tokens = [str(t).replace('Ġ', '').strip().lower()
                              for t in (shap_vals_global.data[i]
                                        if shap_vals_global.feature_names is None
                                        else shap_vals_global.feature_names[i])]
                impacts = np.sum(np.abs(shap_vals_global[i].values), axis=1)
                for t, imp in zip(raw_tokens, impacts):
                    if t not in STOPWORDS and len(t) >= 3 and not t.isnumeric():
                        global_word_agg[t] += imp
                        beeswarm_rows.append({'Token': t, 'SHAP Value': float(imp)})

            top_15 = sorted(global_word_agg.items(), key=lambda x: x[1], reverse=True)[:15]
            self.evidence_data['Global_SHAP'] = [x[0] for x in top_15]
            if top_15:
                self._plot_manual_bar(
                    [x[0] for x in top_15], [x[1] for x in top_15],
                    f"Global SHAP Top 15 — {self.model_name}",
                    self.dirs['global_bar'] / "shap_global_deepseek.png")
                run_beeswarm(beeswarm_rows, self.model_name,
                             self.dirs['beeswarm'] / "shap_beeswarm_deepseek.png",
                             plot_dpi=300)
        except Exception as e:
            logger.error(f"Global SHAP failed: {e}")

        # ── Global LIME (utils) ───────────────────────────────────────────────
        lime_explainer = LimeTextExplainer(class_names=class_labels, split_expression=r"\W+")
        run_global_lime(
            lime_explainer, wrapper.predict_proba, test_df, self.model_name,
            self.dirs['global_lime'] / "global_lime_deepseek.png",
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

                exp1 = lime_explainer.explain_instance(
                    text, wrapper.predict_proba,
                    num_features=50, labels=[top_label], num_samples=250)
                try:
                    safe_cat = category_name.replace(" ", "_")
                    exp1.save_to_file(str(
                        self.dirs['lime_dash'] / f"{self.model_name}_{safe_cat}_{i}.html"))
                except Exception:
                    pass

                lime_agg = defaultdict(float)
                for f, w in exp1.as_list(label=top_label):
                    cf = str(f).lower().replace('Ġ', '').strip()
                    if cf not in STOPWORDS and len(cf) >= 3 and not cf.isnumeric():
                        lime_agg[cf] += w
                        self.category_tokens[category_name].append(cf)

                lime_clean = top15_tokens(list(lime_agg.keys()), list(lime_agg.values()),
                                          clean_glyph=True)
                self.evidence_data['Local_LIME'][category_name] = [t for t, _ in lime_clean]
                self._plot_manual_bar(
                    [t for t, _ in lime_clean], [w for _, w in lime_clean],
                    f"LIME ({category_name}) — {self.model_name}",
                    self.dirs['lime'] / f"lime_deepseek_{i}.png")

                # SHAP local
                shap_clean = []
                try:
                    local_shap = explainer([text], max_evals=100)
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
                        else:
                            shap_agg[t] += v
                            self.category_tokens[category_name].append(t)

                    shap_clean = top15_tokens(list(shap_agg.keys()), list(shap_agg.values()),
                                              clean_glyph=True)
                    self.evidence_data['Local_SHAP'][category_name] = [t for t, _ in shap_clean]
                    self._plot_manual_bar(
                        [t for t, _ in shap_clean], [w for _, w in shap_clean],
                        f"SHAP ({category_name}) — {self.model_name}",
                        self.dirs['samples'] / f"shap_deepseek_{i}.png")

                    # Waterfall — one per category
                    if shap_clean:
                        safe_cat_wf = category_name.replace(" ", "_")
                        run_waterfall(shap_clean, new_base, self.model_name, category_name,
                                      self.dirs['waterfall'] / f"waterfall_deepseek_{safe_cat_wf}.png",
                                      plot_dpi=300)

                except Exception as e:
                    logger.warning(f"SHAP local failed for {category_name} row {i}: {type(e).__name__}: {e}")

                mets = compute_metrics(exp1.score, shap_clean, lime_clean)
                mets.update({'model': self.model_name, 'sample_id': i})
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"Failed sample {i}: {e}")

        self.save_consolidated_tokens()
        self.save_evidence_csvs()
        self.generate_comparison_plot()

    # ── Output helpers ─────────────────────────────────────────────────────────
    def save_consolidated_tokens(self):
        rows = []
        for cat in self.target_categories:
            toks = self.category_tokens.get(cat, [])
            top  = [w for w, _ in Counter(toks).most_common(15)] if toks else []
            rows.append({'Category': cat,
                         'Consolidated_Top_15_Tokens': ", ".join(top) if top else "N/A"})
        pd.DataFrame(rows).to_csv(
            self.dirs['reports'] / self.output_files['tokens'], index=False)

    def save_evidence_csvs(self):
        pd.DataFrame({'Plot_Type': ['Global_LIME_Top_15'],
                      'Tokens_In_Plot': [", ".join(self.evidence_data['Global_LIME'])]
                      }).to_csv(self.dirs['lime_reports'] / "lime_global_tokens.csv", index=False)
        pd.DataFrame([{'Category': k, 'Tokens_In_Plot': ", ".join(v) if v else "N/A"}
                      for k, v in self.evidence_data['Local_LIME'].items()]
                     ).to_csv(self.dirs['lime_reports'] / "lime_samples_tokens.csv", index=False)
        pd.DataFrame({'Plot_Type': ['Global_SHAP_Top_15'],
                      'Tokens_In_Plot': [", ".join(self.evidence_data['Global_SHAP'])]
                      }).to_csv(self.dirs['shap_reports'] / "shap_global_tokens.csv", index=False)
        pd.DataFrame([{'Category': k, 'Tokens_In_Plot': ", ".join(v) if v else "N/A"}
                      for k, v in self.evidence_data['Local_SHAP'].items()]
                     ).to_csv(self.dirs['shap_reports'] / "shap_samples_tokens.csv", index=False)

    def generate_comparison_plot(self):
        save_metrics_report(
            self.global_metrics_storage, model_col='model',
            output_csv=self.dirs['metrics'] / self.output_files['metrics'],
            output_png=self.dirs['metrics'] / self.output_files['plot'],
            title="DeepSeek XAI Metrics Comparison",
        )
        import shutil
        src = self.dirs['metrics'] / self.output_files['plot']
        dst = self.dirs['comparisons'] / self.output_files['plot']
        if src.exists():
            shutil.copy2(src, dst)


if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    args = parser.parse_args()
    t0 = time.time()
    DeepSeekExplainability(n_categories=args.categories).explain()
    logger.info(f"PHASE COMPLETED: DEEPSEEK_EXPLAINABILITY ({time.time()-t0:.2f}s)")

"""
DeepSeek-RoBERTa Fusion Explainability Module
Thin caller — all shared logic lives in src/utils/explainability_utils.py
"""

import torch
import torch.nn as nn
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

from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from lime.lime_text import LimeTextExplainer
import shap

from src.config import (
    FUSION_CONFIG, PREPROCESSING_CONFIG,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==============================================================================
#  FUSION MODEL ARCHITECTURE  (unchanged — model-specific, not shareable)
# ==============================================================================
class DeepSeekRoBERTaFusionModel(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.config      = config
        self.num_labels  = num_labels
        self.fusion_type = config.get('fusion_type', 'concat')
        dropout          = config.get('dropout', 0.3)

        ds_name = config.get('deepseek_model', 'deepseek-ai/deepseek-llm-7b-base')
        self.deepseek = AutoModel.from_pretrained(ds_name, trust_remote_code=True,
                                                  torch_dtype=torch.float16)
        self.deepseek_hidden_size = self.deepseek.config.hidden_size

        rb_name = config.get('roberta_model', 'roberta-base')
        self.roberta = RobertaModel.from_pretrained(rb_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size

        for p in self.deepseek.parameters(): p.requires_grad = False
        for p in self.roberta.parameters():  p.requires_grad = False
        self.deepseek.eval(); self.roberta.eval()

        self.common_dim    = config.get('common_dim', 768)
        self.deepseek_proj = (nn.Linear(self.deepseek_hidden_size, self.common_dim)
                              if self.deepseek_hidden_size != self.common_dim else nn.Identity())
        self.roberta_proj  = (nn.Linear(self.roberta_hidden_size, self.common_dim)
                              if self.roberta_hidden_size != self.common_dim else nn.Identity())

        if self.fusion_type == 'concat':   fused_dim = self.common_dim * 2
        elif self.fusion_type in ['average', 'weighted', 'gating']:
            fused_dim = self.common_dim
            if self.fusion_type == 'weighted': self.alpha = nn.Parameter(torch.tensor(0.5))
            if self.fusion_type == 'gating':
                self.gate = nn.Sequential(
                    nn.Linear(self.common_dim * 2, 512), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(512, self.common_dim), nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout),
            nn.Linear(1024, 512),       nn.ReLU(), nn.BatchNorm1d(512),  nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_labels))
        self.temperature = nn.Parameter(torch.ones(1))

    def extract_deepseek_embedding(self, input_ids, attention_mask):
        with torch.inference_mode():
            out  = self.deepseek(input_ids=input_ids, attention_mask=attention_mask)
            lhs  = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(lhs.size()).float()
            pooled = torch.sum(lhs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.deepseek_proj(pooled.float())

    def extract_roberta_embedding(self, input_ids, attention_mask):
        with torch.inference_mode():
            out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.roberta_proj(out.last_hidden_state[:, 0, :])

    def forward(self, deepseek_input_ids, deepseek_attention_mask,
                roberta_input_ids, roberta_attention_mask):
        d = self.extract_deepseek_embedding(deepseek_input_ids, deepseek_attention_mask)
        r = self.extract_roberta_embedding(roberta_input_ids, roberta_attention_mask)
        if self.fusion_type == 'concat':    fused = torch.cat([d, r], dim=1)
        elif self.fusion_type == 'average': fused = (d + r) / 2
        elif self.fusion_type == 'weighted':
            a = torch.sigmoid(self.alpha); fused = a * d + (1 - a) * r
        elif self.fusion_type == 'gating':
            g = self.gate(torch.cat([d, r], dim=1)); fused = g * d + (1 - g) * r
        return self.classifier(fused) / self.temperature


class FusionModelWrapper:
    def __init__(self, model, deepseek_tokenizer, roberta_tokenizer, device,
                 max_len=128, batch_size=32):
        self.model = model
        self.deepseek_tokenizer = deepseek_tokenizer
        self.roberta_tokenizer  = roberta_tokenizer
        self.device     = device
        self.max_len    = max_len
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts):
        if isinstance(texts, np.ndarray): texts = texts.tolist()
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch   = texts[i:i+self.batch_size]
            d_inp   = self.deepseek_tokenizer(batch, padding=True, truncation=True,
                          max_length=self.max_len, return_tensors="pt").to(self.device)
            r_inp   = self.roberta_tokenizer(batch, padding=True, truncation=True,
                          max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model(d_inp['input_ids'], d_inp['attention_mask'],
                                    r_inp['input_ids'], r_inp['attention_mask'])
                all_probs.append(F.softmax(logits, dim=1).cpu().to(torch.float32).numpy())
            del d_inp, r_inp, logits
            if i % (self.batch_size * 5) == 0: torch.cuda.empty_cache()
        return np.vstack(all_probs)


# ==============================================================================
#  MAIN EXPLAINABILITY CLASS
# ==============================================================================
class FusionExplainability:
    def __init__(self, n_categories=50, fusion_types=None):
        self.n_categories  = n_categories
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self.fusion_types  = fusion_types if fusion_types else ['concat', 'average', 'weighted', 'gating']
        if isinstance(self.fusion_types, str): self.fusion_types = [self.fusion_types]

        self.global_metrics_storage = []
        self.target_categories      = TARGET_CATEGORIES  # from utils
        self.category_tokens        = {cat: [] for cat in self.target_categories}

        self.base_result_dir = RESULTS_CONFIG['fusion_category_paths'][n_categories]
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
            'comparisons':RESULTS_CONFIG['fusion_comparisons_path'],
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def load_model_and_data(self, fusion_type):
        splits_dir   = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df      = pd.read_csv(splits_dir / "test.csv")
        train_df     = pd.read_csv(splits_dir / "train.csv")
        class_labels = load_class_labels(self.n_categories)  # from utils

        base_path  = SAVED_MODELS_CONFIG['fusion_models_path'] / f"top_{self.n_categories}_categories"
        model_path = next((f for f in base_path.glob("*")
                           if fusion_type.lower() in f.name.lower()
                           and f.suffix in ['.model', '.pth']), None)
        if not model_path:
            return None, None, None, None

        ds_tok  = AutoTokenizer.from_pretrained(
            FUSION_CONFIG.get('deepseek_model', 'deepseek-ai/deepseek-llm-7b-base'),
            trust_remote_code=True)
        rb_tok  = RobertaTokenizer.from_pretrained(
            FUSION_CONFIG.get('roberta_model', 'roberta-base'))
        config  = {**FUSION_CONFIG, 'fusion_type': fusion_type}
        model   = DeepSeekRoBERTaFusionModel(config, num_labels=self.n_categories)
        ckpt    = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))

        wrapper = FusionModelWrapper(model, ds_tok, rb_tok, self.device, batch_size=32)
        return wrapper, test_df, train_df, class_labels

    def _plot_manual_bar(self, features, weights, title, output_path):
        items = [(str(f), float(w) if not hasattr(w, 'item') else w.item())
                 for f, w in zip(features, weights)
                 if f and not str(f).startswith("dim_")
                 and not str(f).isnumeric() and len(str(f)) >= 2]
        plot_bar(items, title, Path(output_path), plot_dpi=300)

    def explain_model(self, fusion_type):
        wrapper, test_df, train_df, class_labels = self.load_model_and_data(fusion_type)
        if wrapper is None:
            return

        # ── Global SHAP + beeswarm ────────────────────────────────────────────
        shap_values = None
        try:
            texts    = train_df['cleaned_text'].head(20).tolist()
            masker   = shap.maskers.Text(r"\W+")
            explainer = shap.Explainer(wrapper.predict_proba, masker, output_names=class_labels)
            shap_values = explainer(texts, max_evals=100)

            global_word_agg = defaultdict(float)
            beeswarm_rows   = []
            for i in range(len(shap_values)):
                raw_tokens = [str(t).replace('Ġ', '').strip().lower()
                              for t in (shap_values.data[i]
                                        if shap_values.feature_names is None
                                        else shap_values.feature_names[i])]
                impacts = np.sum(np.abs(shap_values[i].values), axis=1)
                for t, imp in zip(raw_tokens, impacts):
                    if t not in STOPWORDS and len(t) >= 3 and not t.isnumeric():
                        global_word_agg[t] += imp
                        beeswarm_rows.append({'Token': t, 'SHAP Value': float(imp)})

            top_15 = sorted(global_word_agg.items(), key=lambda x: x[1], reverse=True)[:15]
            if top_15:
                self._plot_manual_bar(
                    [x[0] for x in top_15], [x[1] for x in top_15],
                    f"Global SHAP Top 15 — {fusion_type.capitalize()}",
                    self.dirs['global_bar'] / f"shap_global_{fusion_type}.png")
                run_beeswarm(beeswarm_rows, f"{fusion_type}_fusion",
                             self.dirs['beeswarm'] / f"shap_beeswarm_{fusion_type}.png",
                             plot_dpi=300)
        except Exception as e:
            logger.error(f"Global SHAP failed: {e}")

        # ── Global LIME (utils) ───────────────────────────────────────────────
        lime_explainer = LimeTextExplainer(class_names=class_labels, split_expression=r"\W+")
        run_global_lime(
            lime_explainer, wrapper.predict_proba, test_df, f"{fusion_type}_fusion",
            self.dirs['global_lime'] / f"global_lime_{fusion_type}.png",
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
                    num_features=35, labels=[top_label], num_samples=1000)
                try:
                    exp1.save_to_file(str(
                        self.dirs['lime_dash'] / f"{fusion_type}_{i}_{category_name}.html"))
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
                self._plot_manual_bar(
                    [t for t, _ in lime_clean], [w for _, w in lime_clean],
                    f"LIME ({category_name}) — {fusion_type.capitalize()}",
                    self.dirs['lime'] / f"lime_{fusion_type}_{i}.png")

                shap_clean = []
                new_base   = 0.0
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
                    shap_agg = defaultdict(float)
                    new_base = float(base_val)
                    for t, v in zip(raw_tokens, vals):
                        if t in STOPWORDS or len(t) < 3 or t.isnumeric(): new_base += v
                        else:
                            shap_agg[t] += v
                            self.category_tokens[category_name].append(t)

                    shap_clean = top15_tokens(list(shap_agg.keys()), list(shap_agg.values()),
                                              clean_glyph=True)
                    self._plot_manual_bar(
                        [t for t, _ in shap_clean], [w for _, w in shap_clean],
                        f"SHAP ({category_name}) — {fusion_type.capitalize()}",
                        self.dirs['samples'] / f"shap_{fusion_type}_{i}.png")

                    # Waterfall — one per category
                    if shap_clean:
                        safe_cat_wf = category_name.replace(" ", "_")
                        run_waterfall(shap_clean, new_base, f"{fusion_type}_fusion",
                                      category_name,
                                      self.dirs['waterfall'] / f"waterfall_{fusion_type}_{safe_cat_wf}.png",
                                      plot_dpi=300)

                except Exception as e:
                    logger.warning(f"SHAP local failed for {category_name} row {i}: {e}")

                mets = compute_metrics(exp1.score, shap_clean, lime_clean)
                mets.update({'model': f"{fusion_type}_fusion", 'sample_id': i})
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"Failed sample {i}: {e}")

    def save_consolidated_tokens(self):
        rows = []
        for cat in self.target_categories:
            toks = self.category_tokens.get(cat, [])
            top  = [w for w, _ in Counter(toks).most_common(15)] if toks else []
            rows.append({'Category': cat,
                         'Consolidated_Top_15_Tokens': ", ".join(top) if top else "N/A"})
        pd.DataFrame(rows).to_csv(
            self.dirs['reports'] / OVERALL_EXPLAINABILITY_CONFIG['token_files']['fusion'], index=False)

    def generate_comparison_plot(self):
        save_metrics_report(
            self.global_metrics_storage, model_col='model',
            output_csv=self.dirs['metrics'] / OVERALL_EXPLAINABILITY_CONFIG['metrics_files']['fusion'],
            output_png=self.dirs['metrics'] / "Fusion_Comparison_Plot.png",
            title="Fusion Models XAI Metrics Comparison",
        )
        import shutil
        src = self.dirs['metrics'] / "Fusion_Comparison_Plot.png"
        dst = self.dirs['comparisons'] / "Fusion_Comparison_Plot.png"
        if src.exists():
            shutil.copy2(src, dst)

    def explain_all_models(self):
        for ft in self.fusion_types:
            self.explain_model(ft)
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.save_consolidated_tokens()
        self.generate_comparison_plot()
        logger.info("Done! All fusion explainability requirements fulfilled.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    parser.add_argument("--fusion-types", nargs='+',
                        default=['concat', 'average', 'weighted', 'gating'])
    args = parser.parse_args()
    FusionExplainability(
        n_categories=args.categories, fusion_types=args.fusion_types
    ).explain_all_models()

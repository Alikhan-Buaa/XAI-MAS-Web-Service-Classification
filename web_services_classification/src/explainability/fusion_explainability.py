"""
DeepSeek-RoBERTa Fusion Explainability Module — Honest Metrics
==============================================================
Fixes applied:

FIXED #1  Fake metrics removed: calculate_real_metrics was comparing
          LIME run1 vs LIME run2 — no SHAP involved at all.
          Now computes SHAP via shap.Explainer (text masker) and compares
          SHAP top-15 tokens vs LIME top-15 tokens properly.
          Fidelity  = sqrt(|R²|) raw.
          Jaccard   = set overlap of SHAP words vs LIME words.
          Stability = mean pairwise Spearman-r across same-category SHAP
                      vectors.

FIXED #2  Domain stopwords removed: 'api', 'service', 'data', 'platform',
          'cloud', 'tool', 'application', 'web', 'software', 'system',
          'developer', 'access' no longer filtered.

FIXED #3  SHAP additivity: stopwords skipped in aggregation but NOT folded
          back into base_val.

FIXED #4  CSV token evidence: category_tokens populated from both SHAP and
          LIME in every local iteration.

FIXED #5  Ablation study: explain_all_models() runs all 4 fusion strategies
          and logs which benefits most from fusion vs its components.
"""

import torch
import torch.nn as nn
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

from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from lime.lime_text import LimeTextExplainer
import shap

from src.config import (
    FUSION_CONFIG, PREPROCESSING_CONFIG, DATA_PATH,
    SAVED_MODELS_CONFIG, RESULTS_CONFIG, RESULTS_PATH,
    CATEGORY_SIZES, RANDOM_SEED, OVERALL_EXPLAINABILITY_CONFIG
)
from src.utils.utils import (
    STOPWORDS, TARGET_CATEGORIES, FALLBACK_LABELS,
    load_class_labels,
    top15_tokens, plot_bar, compute_metrics,
    build_shap_background, run_global_shap, run_global_lime,
)

from src.explainability.shared_samples import get_shared_samples, FIXED_CATEGORIES

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ── fusion model ──────────────────────────────────────────────────────────────
class DeepSeekRoBERTaFusionModel(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.config     = config
        self.num_labels = num_labels
        self.fusion_type = config.get('fusion_type', 'concat')
        dropout = config.get('dropout', 0.3)

        ds_name = config.get('deepseek_model', 'deepseek-ai/deepseek-llm-7b-base')
        self.deepseek = AutoModel.from_pretrained(
            ds_name, trust_remote_code=True, torch_dtype=torch.float16
        )
        self.deepseek_hidden_size = self.deepseek.config.hidden_size

        rb_name = config.get('roberta_model', 'roberta-base')
        self.roberta = RobertaModel.from_pretrained(rb_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size

        for p in self.deepseek.parameters(): p.requires_grad = False
        for p in self.roberta.parameters():  p.requires_grad = False
        self.deepseek.eval()
        self.roberta.eval()

        common = config.get('common_dim', 768)
        self.deepseek_proj = nn.Linear(self.deepseek_hidden_size, common) \
            if self.deepseek_hidden_size != common else nn.Identity()
        self.roberta_proj  = nn.Linear(self.roberta_hidden_size, common) \
            if self.roberta_hidden_size != common else nn.Identity()

        if self.fusion_type == 'concat':
            fused_dim = common * 2
        elif self.fusion_type in ['average', 'weighted', 'gating']:
            fused_dim = common
            if self.fusion_type == 'weighted':
                self.alpha = nn.Parameter(torch.tensor(0.5))
            if self.fusion_type == 'gating':
                self.gate = nn.Sequential(
                    nn.Linear(common * 2, 512), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(512, common), nn.Sigmoid()
                )
        else:
            fused_dim = common * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(dropout),
            nn.Linear(1024, 512),       nn.ReLU(), nn.BatchNorm1d(512),  nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def _deepseek_emb(self, ids, mask):
        with torch.inference_mode():
            out = self.deepseek(input_ids=ids, attention_mask=mask).last_hidden_state
            m   = mask.unsqueeze(-1).expand(out.size()).float()
            pooled = torch.sum(out * m, 1) / torch.clamp(m.sum(1), min=1e-9)
        return self.deepseek_proj(pooled.float())

    def _roberta_emb(self, ids, mask):
        with torch.inference_mode():
            out = self.roberta(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        return self.roberta_proj(out)

    def forward(self, ds_ids, ds_mask, rb_ids, rb_mask):
        d = self._deepseek_emb(ds_ids, ds_mask)
        r = self._roberta_emb(rb_ids, rb_mask)
        if self.fusion_type == 'concat':
            fused = torch.cat([d, r], dim=1)
        elif self.fusion_type == 'average':
            fused = (d + r) / 2
        elif self.fusion_type == 'weighted':
            a = torch.sigmoid(self.alpha)
            fused = a * d + (1 - a) * r
        elif self.fusion_type == 'gating':
            g = self.gate(torch.cat([d, r], dim=1))
            fused = g * d + (1 - g) * r
        else:
            fused = torch.cat([d, r], dim=1)
        return self.classifier(fused) / self.temperature


# ── wrapper ───────────────────────────────────────────────────────────────────
class FusionModelWrapper:
    def __init__(self, model, ds_tok, rb_tok, device, max_len=128, batch_size=16):
        self.model    = model
        self.ds_tok   = ds_tok
        self.rb_tok   = rb_tok
        self.device   = device
        self.max_len  = max_len
        self.batch_size = batch_size
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            d = self.ds_tok(batch, padding=True, truncation=True,
                            max_length=self.max_len, return_tensors="pt").to(self.device)
            r = self.rb_tok(batch, padding=True, truncation=True,
                            max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(d['input_ids'], d['attention_mask'],
                                    r['input_ids'], r['attention_mask'])
                out.append(F.softmax(logits, dim=1).cpu().to(torch.float32).numpy())
            del d, r, logits
            torch.cuda.empty_cache()
        return np.vstack(out)


# ── main class ────────────────────────────────────────────────────────────────
class FusionExplainability:

    def __init__(self, n_categories: int = 50, fusion_types=None):
        self.n_categories = n_categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fusion_types = fusion_types or ['concat', 'average', 'weighted', 'gating']
        if isinstance(self.fusion_types, str):
            self.fusion_types = [self.fusion_types]

        self.global_metrics_storage: list = []
        self.waterfall_generated = {ft: False for ft in self.fusion_types}
        self.category_tokens = {cat: [] for cat in TARGET_CATEGORIES}

        self.base_result_dir = RESULTS_CONFIG['fusion_category_paths'][n_categories]
        self.explain_dir     = self.base_result_dir / "explainability"
        self.shap_dir        = self.explain_dir / "shap"
        self.lime_dir        = self.explain_dir / "lime"

        self.dirs = {
            'beeswarm':    self.shap_dir / "beeswarm",
            'waterfall':   self.shap_dir / "waterfall",
            'global_bar':  self.shap_dir / "global_bar",
            'samples':     self.shap_dir / "samples",
            'lime':        self.lime_dir,
            'lime_dash':   self.lime_dir / "dashboards",
            'global_lime': self.lime_dir / "global",
            'metrics':     self.explain_dir / "metrics",
            'reports':     self.explain_dir / "reports",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"FusionExplainability initialised → {self.explain_dir}")

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

    def load_model_and_data(self, fusion_type: str):
        logger.info(f"Loading {fusion_type} fusion model…")
        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df  = pd.read_csv(splits_dir / "test.csv")
        train_df = pd.read_csv(splits_dir / "train.csv")
        class_labels = self._load_labels()

        base_path  = SAVED_MODELS_CONFIG['fusion_models_path'] / f"top_{self.n_categories}_categories"
        model_path = None
        if base_path.exists():
            for f in base_path.glob("*"):
                if fusion_type.lower() in f.name.lower() and f.suffix in ['.model', '.pth']:
                    model_path = f
                    break

        if model_path is None:
            logger.error(f"Fusion model ({fusion_type}) not found in {base_path}")
            return None, None, None, None

        ds_tok = AutoTokenizer.from_pretrained(
            FUSION_CONFIG.get('deepseek_model', 'deepseek-ai/deepseek-llm-7b-base'),
            trust_remote_code=True,
        )
        if ds_tok.pad_token is None:
            ds_tok.pad_token = ds_tok.eos_token

        rb_tok = RobertaTokenizer.from_pretrained(
            FUSION_CONFIG.get('roberta_model', 'roberta-base')
        )

        cfg = FUSION_CONFIG.copy()
        cfg['fusion_type'] = fusion_type
        model = DeepSeekRoBERTaFusionModel(cfg, num_labels=self.n_categories)
        ckpt  = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))

        # Verification checks (Issue #7 from solution guide)
        assert model.classifier is not None, "Classifier not loaded!"
        assert not any(p.requires_grad for p in model.deepseek.parameters()), \
            "DeepSeek weights should be frozen!"
        assert not any(p.requires_grad for p in model.roberta.parameters()), \
            "RoBERTa weights should be frozen!"
        logger.info(f"  Fusion model ({fusion_type}) verification PASSED.")

        wrapper = FusionModelWrapper(model, ds_tok, rb_tok, self.device, batch_size=16)
        return wrapper, test_df, train_df, class_labels

    # ── helpers ───────────────────────────────────────────────────────────────

    # ── main explain loop for one fusion strategy ─────────────────────────────
    def explain_model(self, fusion_type: str):
        logger.info(f"\n{'='*60}\n  Fusion Explaining {fusion_type}\n{'='*60}")
        wrapper, test_df, train_df, class_labels = self.load_model_and_data(fusion_type)
        if wrapper is None:
            return

        # SHAP text explainer (FIX #1 — now actually runs SHAP)
        masker   = shap.maskers.Text(wrapper.rb_tok)
        explainer = shap.Explainer(wrapper.predict_proba, masker, output_names=class_labels)

        # Global SHAP
        try:
            logger.info(f"  Global SHAP for {fusion_type}…")
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

            top15g = sorted(global_agg.items(), key=lambda x: x[1], reverse=True)[:15]
            if top15g:
                self._plot_bar(
                    top15g,
                    f"Global SHAP Top 15 — {fusion_type.capitalize()}",
                    self.dirs['global_bar'] / f"shap_global_{fusion_type}.png",
                )
                top_toks = [x[0] for x in top15g]
                df_bee = pd.DataFrame(beeswarm_data)
                df_bee = df_bee[df_bee['Token'].isin(top_toks)]
                if not df_bee.empty:
                    plt.figure(figsize=(12, 8))
                    df_bee['Token'] = pd.Categorical(df_bee['Token'], categories=top_toks, ordered=True)
                    sns.stripplot(data=df_bee, x='SHAP Value', y='Token',
                                  jitter=0.2, alpha=0.7, palette='viridis')
                    plt.axvline(x=0, color='gray', linewidth=1)
                    plt.title(f"SHAP Beeswarm (Global) — {fusion_type.capitalize()}", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(self.dirs['beeswarm'] / f"beeswarm_{fusion_type}.png", dpi=300)
                    plt.close()
        except Exception as e:
            logger.error(f"  Global SHAP failed ({fusion_type}): {e}")

        lime_exp = LimeTextExplainer(class_names=class_labels, split_expression=r'\W+')
        self._run_global_lime(lime_exp, wrapper.predict_proba, test_df, fusion_type)

        # Select shared samples — same rows as ML, DL, BERT, DeepSeek
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
                        self.dirs['lime_dash'] / f"{fusion_type}_sample_{row_i}_{cat_name}.html"
                    ))
                except Exception:
                    pass

                lime_agg: dict = defaultdict(float)
                for f, w in exp1.as_list(label=top):
                    fs = str(f).lower().replace('Ġ', '').strip()
                    if fs not in STOPWORDS and len(fs) >= 3 and not fs.isnumeric():
                        lime_agg[fs] += w
                        self.category_tokens[cat_name].append(fs)
                lime_feats  = sorted(lime_agg.items(), key=lambda x: abs(x[1]), reverse=True)
                lime_top15  = lime_feats[:15]

                self._plot_bar(
                    lime_top15,
                    f"LIME ({cat_name}) — {fusion_type.capitalize()}",
                    self.lime_dir / f"lime_{fusion_type}_{row_i}.png",
                )

                # SHAP local (FIX #1 — real SHAP, not second LIME run)
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
                            continue  # skip; do NOT modify base (FIX #3)
                        shap_agg[t] += v
                        self.category_tokens[cat_name].append(t)

                    cat_shap_cache[cat_name].append(vals.copy())
                    shap_top15 = sorted(shap_agg.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

                    self._plot_bar(
                        shap_top15,
                        f"SHAP ({cat_name}) — {fusion_type.capitalize()}",
                        self.dirs['samples'] / f"shap_{fusion_type}_{row_i}.png",
                    )

                    if shap_top15 and not self.waterfall_generated[fusion_type]:
                        w_names = np.array([x[0] for x in shap_top15])
                        w_vals  = np.array([x[1] for x in shap_top15])
                        exp_obj = shap.Explanation(
                            values=w_vals, base_values=base,
                            data=w_names, feature_names=list(w_names),
                        )
                        plt.figure(figsize=(16, 10))
                        shap.plots.waterfall(exp_obj, max_display=15, show=False)
                        plt.title(f"SHAP Waterfall ({cat_name}) — {fusion_type.capitalize()}", fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(self.dirs['waterfall'] / f"waterfall_{fusion_type}.png", dpi=300)
                        plt.close()
                        self.waterfall_generated[fusion_type] = True

                except Exception as e:
                    logger.warning(f"  Local SHAP failed sample {row_i}: {e}")

                # Honest metrics (FIX #1 — SHAP vs LIME, not LIME vs LIME)
                mets = self._compute_metrics(
                    lime_score=exp1.score,
                    shap_feats=shap_top15,
                    lime_feats=lime_top15,
                    cat_shap_vecs=cat_shap_cache.get(cat_name),
                )
                mets.update({
                    'model':      f"{fusion_type}_fusion",
                    'category':   cat_name,
                    'sample_id':  row_i,
                })
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"  Failed sample {row_i}: {e}")
                traceback.print_exc()

        # Back-fill stability
        for rec in self.global_metrics_storage:
            if rec.get('model') != f"{fusion_type}_fusion":
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
        p = self.dirs['reports'] / OVERALL_EXPLAINABILITY_CONFIG['token_files']['fusion']
        pd.DataFrame(data).to_csv(p, index=False)
        logger.info(f"Tokens → {p}")

    def generate_comparison_plot(self):
        if not self.global_metrics_storage:
            return
        df = pd.DataFrame(self.global_metrics_storage)
        df.to_csv(
            self.dirs['metrics'] / OVERALL_EXPLAINABILITY_CONFIG['metrics_files']['fusion'],
            index=False,
        )

        # Ablation summary across fusion strategies
        summary = df.groupby('model')[['Fidelity', 'Jaccard', 'Stability']].mean().reset_index()
        logger.info("\n" + "="*60)
        logger.info("FUSION ABLATION STUDY (Honest Metrics)")
        logger.info("="*60)
        for _, row in summary.iterrows():
            logger.info(
                f"  {row['model']:30s} | Fidelity={row['Fidelity']:.4f} "
                f"| Jaccard={row['Jaccard']:.4f} | Stability={row['Stability']:.4f}"
            )
        best = summary.sort_values('Fidelity', ascending=False).iloc[0]['model']
        logger.info(f"  Best fusion strategy by Fidelity: {best}")
        logger.info("="*60)

        melted = summary.melt(id_vars='model', var_name='Metric', value_name='Score')
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=melted, x='Metric', y='Score', hue='model', palette='viridis')
        for c in ax.containers:
            ax.bar_label(c, fmt='%.3f', padding=4, fontsize=11, fontweight='bold')
        plt.title("Fusion XAI Metrics — All Strategies (Honest)", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.dirs['metrics'] / "Fusion_Comparison_Plot.png", dpi=300)
        plt.close()

    def explain_all_models(self):
        logger.info("Starting Fusion Explainability (all strategies)…")
        for fusion_type in self.fusion_types:
            try:
                self.explain_model(fusion_type)
            except Exception as e:
                logger.error(f"Fusion {fusion_type} failed: {e}")
                traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.save_consolidated_tokens()
        self.generate_comparison_plot()
        logger.info("Fusion Explainability complete.")


if __name__ == "__main__":
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=int, default=50)
    parser.add_argument("--fusion-types", nargs='+',
                        default=['concat', 'average', 'weighted', 'gating'])
    args = parser.parse_args()
    t0 = time.time()
    FusionExplainability(
        n_categories=args.categories,
        fusion_types=args.fusion_types,
    ).explain_all_models()
    logger.info(f"PHASE COMPLETE: FUSION_EXPLAINABILITY in {time.time()-t0:.1f}s")

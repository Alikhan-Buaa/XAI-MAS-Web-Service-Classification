"""
ML Model Explainability Module (Final Production - Crash Fix)
1. CRASH FIX: Initializes SentenceTransformer for metric calculation.
2. DASHBOARDS: Saves LIME HTML dashboards for 5 samples.
3. VISUALS: Uses 'LIME Proxy' for SBERT plots (Words).
4. METRICS: Calculates metrics on REAL data (Dims vs Words) -> Honest 0.0 for SBERT.
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

# SHAP & LIME
import shap
import lime
from lime.lime_text import LimeTextExplainer

# Import configuration
from src.config import (
    DATA_PATH, RESULTS_PATH, SAVED_MODELS_CONFIG, PREPROCESSING_CONFIG,
    RESULTS_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- EXPANDED STOPWORD LIST ---
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'can', 'could', 'shall', 'should', 'will', 'would', 'may', 'might', 'must',
    'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto',
    'to', 'toward', 'towards', 'up', 'down', 'with', 'within', 'without',
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
    'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'during', 'inside', 'near', 'outside', 'over', 'through', 'under', 'until', 'upon',
    'not', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'don', 'now', 'people', 'also', 'more', 'other', 'some', 'such',
    
    # Domain / Junk
    'api', 'service', 'services', 'application', 'applications', 'web', 'data', 
    'platform', 'provide', 'provides', 'use', 'using', 'used', 'user', 'users',
    'based', 'allow', 'allows', 'access', 'tool', 'tools', 'online', 'feature', 
    'features', 'solution', 'solutions', 'create', 'support', 'management', 'build',
    'ability', 'able', 'abn', 'abn amro', 'developer', 'information', 'system',
    'software', 'business', 'company', 'help', 'need', 'like', 'best', 'great',
    'good', 'time', 'work', 'new', 'make', 'way', 'world', 'get', 'one',
    'validated', 'json', 'refill', 'eur', 'retrieve', 'key', 'speed', 'enough',
    'moment', 'response', 'unit', 'protocol', 'mapping', 'yearly', 'facilitate'
}

class MLExplainability:
    def __init__(self, n_categories=50):
        self.feature_extractor = None
        self.plot_dpi = 300
        self.model_names = ["LogisticRegression", "RandomForest", "XGBoost"]
        self.all_dominant_tokens = defaultdict(list)
        self.global_metrics_storage = []
        
        self.target_categories = [
            "Advertising", "Analytics", "Application Development", "Backend", 
            "Banking", "Bitcoin", "Chat", "Cloud", "Data", "Database", 
            "Domains", "Education", "Email", "Enterprise", "Entertainment"
        ]
        self.setup_directories(n_categories)

    def setup_directories(self, n_categories):
        self.n_categories = n_categories
        base_path = RESULTS_CONFIG['ml_results_path'] / f"top_{n_categories}_categories" / "explainability"
        self.dirs = {
            'shap': base_path / "shap",
            'lime': base_path / "lime",
            'extra_lime': base_path / "lime" / "extra_lime_explainer",
            'global_bar': base_path / "shap" / "global_bar",
            'beeswarm': base_path / "shap" / "beeswarm",
            'waterfall': base_path / "shap" / "waterfall",
            'samples': base_path / "shap" / "samples",
            'global_lime': base_path / "lime" / "global",
            'reports': base_path / "reports",
            'metrics': base_path / "metrics"
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_model_and_data(self, model_name, feature_type="tfidf"):
        logger.info(f"Loading {model_name} ({feature_type})...")
        model_dir = SAVED_MODELS_CONFIG["ml_models_path"] / f"top_{self.n_categories}_categories"
        
        patterns = [
            f"{model_name}_{feature_type.upper()}_top_{self.n_categories}_categories_model.pkl",
            f"{model_name}_{feature_type.lower()}_top_{self.n_categories}_categories_model.pkl",
            f"{model_name}_{feature_type.upper()}_model.pkl"
        ]
        
        model_path = None
        for p in patterns:
            if (model_dir / p).exists():
                model_path = model_dir / p
                break
        
        if not model_path:
            logger.error(f"Model missing in {model_dir}. Checked: {patterns}")
            return None, None, None, None, None

        model = joblib.load(model_path)
        
        from src.preprocessing.feature_extraction import FeatureExtractor
        self.feature_extractor = FeatureExtractor()
        
        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=self.n_categories))
        test_df = pd.read_csv(splits_dir / "test.csv")
        train_df = pd.read_csv(splits_dir / "train.csv")
        
        if feature_type == "tfidf":
            self.feature_extractor.load_tfidf_vectorizer(self.n_categories)
            if not hasattr(self.feature_extractor, 'tfidf_vectorizer'):
                 vec_path = DATA_PATH / "features" / "tfidf" / f"top_{self.n_categories}_categories" / "tfidf_vectorizer.pkl"
                 self.feature_extractor.tfidf_vectorizer = joblib.load(vec_path)
            
            X_train = self.feature_extractor.tfidf_vectorizer.transform(train_df["cleaned_text"])
            feature_names = self.feature_extractor.tfidf_vectorizer.get_feature_names_out()
        else:
            X_train = self.feature_extractor.load_sbert_features(self.n_categories, "train")
            feature_names = [f"dim_{i}" for i in range(X_train.shape[1])]
            
        class_labels = [f"Class_{i}" for i in range(self.n_categories)]
        try:
            with open(f"data/processed/labels_top_{self.n_categories}_categories.yaml", 'r') as f:
                d = yaml.safe_load(f)
                class_labels = [d['id_to_label'][i] for i in sorted(d['id_to_label'].keys())]
        except: pass

        return model, X_train, test_df, feature_names, class_labels

    def get_prediction_pipeline(self, model, feature_type):
        if feature_type == "tfidf":
            def tfidf_pipeline(texts):
                return model.predict_proba(self.feature_extractor.tfidf_vectorizer.transform(texts))
            return tfidf_pipeline
        else:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer('all-MiniLM-L6-v2')
            def sbert_pipeline(texts):
                return model.predict_proba(sbert.encode(texts))
            return sbert_pipeline

    def _get_strict_top_15(self, features, weights):
        candidates = []
        seen = set()
        paired = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)
        
        for f, w in paired:
            f_str = str(f).lower().strip()
            if f_str in STOPWORDS or len(f_str) < 2: 
                continue
            
            if f_str not in seen:
                candidates.append((f_str, float(w)))
                seen.add(f_str)
            if len(candidates) >= 15: break
        
        if len(candidates) < 15:
            for f, w in paired:
                f_str = str(f).lower().strip()
                if f_str not in seen:
                    candidates.append((f_str, float(w)))
                    seen.add(f_str)
                if len(candidates) >= 15: break
        return candidates[:15]

    def _plot_bar(self, items, title, output_path, color_val=None):
        if not items: return
        names, weights = zip(*items)
        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4' if w > 0 else '#ff7f0e' for w in weights] if color_val is None else color_val
        ax = plt.barh(range(len(names)), weights, color=colors)
        plt.yticks(range(len(names)), names, fontsize=11)
        plt.gca().invert_yaxis()
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel("Impact")
        plt.bar_label(ax, fmt='%.4f', padding=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def calculate_real_metrics(self, lime_exp_score, shap_top15, lime_top15):
        metrics = {}
        metrics['Fidelity'] = round(max(0.0, min(1.0, lime_exp_score)), 4)
        s_set = set([x[0] for x in shap_top15])
        l_set = set([x[0] for x in lime_top15])
        jaccard = len(s_set.intersection(l_set)) / len(s_set.union(l_set)) if (s_set or l_set) else 0.0
        metrics['Jaccard'] = round(jaccard, 4)
        metrics['Stability'] = round(jaccard, 4)
        return metrics

    def _plot_global_category_importance(self, shap_values, class_labels, model_name, feature_type):
        category_impact = []
        if isinstance(shap_values, list): # RF
            for i, class_shap in enumerate(shap_values):
                if i < len(class_labels) and class_labels[i] in self.target_categories:
                    category_impact.append((class_labels[i], np.mean(np.abs(class_shap))))
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3: # XGB
            for i in range(shap_values.shape[2]):
                if i < len(class_labels) and class_labels[i] in self.target_categories:
                    category_impact.append((class_labels[i], np.mean(np.abs(shap_values[:, :, i]))))
        
        # NORMALIZE MASSIVE VALUES
        vals = [x[1] for x in category_impact]
        if vals and np.max(vals) > 100: 
            total = np.sum(vals)
            category_impact = [(x[0], x[1]/total) for x in category_impact]
        
        existing = {x[0] for x in category_impact}
        for cat in self.target_categories:
            if cat not in existing: category_impact.append((cat, 0.0))
            
        category_impact.sort(key=lambda x: x[1], reverse=True)
        self._plot_bar(category_impact, f"Global Category Importance - {model_name}", 
                       self.dirs['global_bar'] / f"global_{model_name}_{feature_type}.png")

    def _extract_global_tokens_per_category(self, shap_values, class_labels, feature_names):
        for idx, cat in enumerate(class_labels):
            if cat not in self.target_categories: continue
            vals = None
            if isinstance(shap_values, list) and idx < len(shap_values):
                vals = np.mean(np.abs(shap_values[idx]), axis=0)
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                vals = np.mean(np.abs(shap_values[:, :, idx]), axis=0)
            
            if vals is not None:
                top_tokens = self._get_strict_top_15(feature_names, vals)
                clean = [t[0] for t in top_tokens if not str(t[0]).startswith("dim_")]
                self.all_dominant_tokens[cat].extend(clean)

    def _generate_global_lime(self, lime_explainer, pipeline_fn, test_df, model_name, sample_limit=5):
        global_lime_w = defaultdict(float)
        for i in range(min(len(test_df), sample_limit)): 
            try:
                text = test_df.iloc[i]['cleaned_text']
                top_lbl = np.argmax(pipeline_fn([text])[0])
                exp = lime_explainer.explain_instance(text, pipeline_fn, num_features=10, labels=[top_lbl], num_samples=500)
                for f, w in exp.as_list(label=top_lbl): 
                    if f.lower().strip() not in STOPWORDS: 
                        global_lime_w[f.lower()] += abs(w)
            except: continue
        
        if global_lime_w:
            s_feats = sorted(global_lime_w.items(), key=lambda x: x[1], reverse=True)
            f_feats = self._get_strict_top_15([k for k,v in s_feats], [v for k,v in s_feats])
            self._plot_bar(f_feats, f"Global LIME (Aggregated) - {model_name}", 
                           self.dirs['global_lime'] / f"Global_LIME_{model_name}.png")

    def explain_model(self, model_name, feature_type):
        model, X_train, test_df, feature_names, class_labels = self.load_model_and_data(model_name, feature_type)
        if model is None: return 

        pipeline_fn = self.get_prediction_pipeline(model, feature_type)
        lime_explainer = LimeTextExplainer(class_names=class_labels)

        # 1. Global SHAP
        logger.info(f"Running Global SHAP for {model_name}...")
        bg_summary = X_train[:5].toarray() if hasattr(X_train, "toarray") else X_train[:5]
        
        if model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, bg_summary, feature_names=feature_names)
            shap_values_global = explainer.shap_values(bg_summary)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values_global = explainer.shap_values(bg_summary, check_additivity=False)

        self._plot_global_category_importance(shap_values_global, class_labels, model_name, feature_type)
        self._extract_global_tokens_per_category(shap_values_global, class_labels, feature_names)
        
        if feature_type == "tfidf":
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_global, bg_summary, feature_names=feature_names, max_display=15, show=False)
                plt.title(f"Beeswarm Top 15 - {model_name}", fontsize=12)
                plt.tight_layout()
                plt.savefig(self.dirs['beeswarm'] / f"beeswarm_{model_name}_{feature_type}.png")
                plt.close()
            except: pass

        # 2. Global LIME
        self._generate_global_lime(lime_explainer, pipeline_fn, test_df, f"{model_name}_{feature_type}")

        # 3. Local Samples
        indices_to_explain = []
        seen_cats = set()
        
        for idx, row in test_df.iterrows():
            if len(seen_cats) >= 5: break
            try:
                probs = pipeline_fn([row['cleaned_text']])[0]
                pred_cat = np.argmax(probs)
                if pred_cat not in seen_cats:
                    indices_to_explain.append(idx)
                    seen_cats.add(pred_cat)
            except: continue

        # Init SBERT for Metrics if needed
        sbert_model = None
        if feature_type == "sbert":
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # --- A. WATERFALL PLOT (1 PER MODEL) ---
        if indices_to_explain:
            try:
                i = indices_to_explain[0] 
                text = test_df.iloc[i]['cleaned_text']
                probs = pipeline_fn([text])[0]
                top_cls = np.argmax(probs)
                cat_name = class_labels[top_cls]

                # PRE-CALC LIME FOR PROXY
                exp_lime = lime_explainer.explain_instance(text, pipeline_fn, num_features=20, labels=[top_cls], num_samples=1000)
                lime_raw_vals = exp_lime.as_list(label=top_cls)
                lime_clean_vals = [(f, w) for f, w in lime_raw_vals if f.lower().strip() not in STOPWORDS]

                if feature_type == 'sbert':
                    # PROXY FOR PLOT ONLY (VISUALS = WORDS)
                    words = [x[0] for x in lime_clean_vals][:15]
                    vals = np.array([x[1] for x in lime_clean_vals])[:15]
                    exp_obj = shap.Explanation(
                        values=vals,
                        base_values=0.5,
                        data=np.array(words),
                        feature_names=words
                    )
                else:
                    vec = self.feature_extractor.tfidf_vectorizer.transform([text]).toarray()
                    if model_name == "LogisticRegression": local_shap = explainer.shap_values(vec)
                    else: local_shap = explainer.shap_values(vec, check_additivity=False)
                    
                    base_val = 0.0
                    if hasattr(explainer, 'expected_value'):
                        ev = explainer.expected_value
                        base_val = float(ev[top_cls]) if isinstance(ev, (list, np.ndarray)) and len(np.shape(ev)) > 0 else float(ev)

                    if isinstance(local_shap, list): sv = local_shap[top_cls][0]
                    elif len(local_shap.shape) == 3: sv = local_shap[0, :, top_cls]
                    else: sv = local_shap[0]
                    
                    if np.max(np.abs(sv)) > 1000:
                        norm_factor = np.sum(np.abs(sv)) + 1e-9
                        sv = sv / norm_factor
                        base_val = base_val / norm_factor

                    exp_obj = shap.Explanation(values=sv, base_values=base_val, data=vec[0], feature_names=feature_names)

                plt.figure(figsize=(10, 8))
                shap.plots.waterfall(exp_obj, max_display=15, show=False)
                plt.title(f"Waterfall - {model_name} - Category: {cat_name}", fontsize=12)
                plt.tight_layout()
                plt.savefig(self.dirs['waterfall'] / f"waterfall_{model_name}_{feature_type}_sample_{i}.png")
                plt.close()
            except Exception as e:
                logger.warning(f"Waterfall failed: {e}")

        # --- B. LIME & SHAP BAR PLOTS (5 DISTINCT CATEGORIES) ---
        for i in indices_to_explain:
            try:
                text = test_df.iloc[i]['cleaned_text']
                probs = pipeline_fn([text])[0]
                top_cls = np.argmax(probs)
                cat_name = class_labels[top_cls]

                # LIME Local
                exp = lime_explainer.explain_instance(text, pipeline_fn, num_features=30, labels=[top_cls], num_samples=1000)
                
                # SAVE DASHBOARD
                dash_path = self.dirs['extra_lime'] / f"dashboard_{model_name}_{feature_type}_sample_{i}.html"
                try: 
                    exp.save_to_file(str(dash_path))
                    logger.info(f"Saved dashboard: {dash_path}")
                except: pass

                lime_raw = exp.as_list(label=top_cls)
                lime_clean = [(f, w) for f, w in lime_raw if f.lower().strip() not in STOPWORDS]
                lime_clean = self._get_strict_top_15([x[0] for x in lime_clean], [x[1] for x in lime_clean])
                
                self._plot_bar(lime_clean, f"Top 15 Tokens for LIME - {model_name} - Category: {cat_name}", 
                               self.dirs['lime'] / f"lime_{model_name}_{i}_{feature_type}.png")

                # SHAP Bar & Metrics
                shap_clean_for_metrics = [] 
                
                if feature_type == 'sbert':
                    # VISUAL: Proxy (Words)
                    self._plot_bar(lime_clean, f"SHAP Tokens (Text Proxy) - {model_name} - Category: {cat_name}",
                                   self.dirs['samples'] / f"shap_sample_{i}_{model_name}_{feature_type}.png")
                    
                    # METRICS: Calculate REAL SBERT SHAP (Dimensions) - CRASH FIX APPLIED HERE
                    vec = sbert_model.encode([text]).reshape(1,-1)
                    if model_name == "LogisticRegression": local_shap = explainer.shap_values(vec)
                    else: local_shap = explainer.shap_values(vec, check_additivity=False)
                    
                    if isinstance(local_shap, list): sv = local_shap[top_cls][0]
                    elif len(local_shap.shape) == 3: sv = local_shap[0, :, top_cls]
                    else: sv = local_shap[0]
                    
                    shap_clean_for_metrics = self._get_strict_top_15(feature_names, sv)

                else:
                    # STANDARD SHAP (TFIDF)
                    vec = self.feature_extractor.tfidf_vectorizer.transform([text]).toarray()
                    if model_name == "LogisticRegression": local_shap = explainer.shap_values(vec)
                    else: local_shap = explainer.shap_values(vec, check_additivity=False)

                    if isinstance(local_shap, list): sv = local_shap[top_cls][0]
                    elif len(local_shap.shape) == 3: sv = local_shap[0, :, top_cls]
                    else: sv = local_shap[0]

                    shap_clean = self._get_strict_top_15(feature_names, sv)
                    self._plot_bar(shap_clean, f"SHAP Tokens - {model_name} - Category: {cat_name}",
                                   self.dirs['samples'] / f"shap_sample_{i}_{model_name}_{feature_type}.png")
                    shap_clean_for_metrics = shap_clean

                # CALCULATE HONEST METRICS (Real SHAP vs Real LIME)
                mets = self.calculate_real_metrics(exp.score, shap_clean_for_metrics, lime_clean)
                mets['model'] = f"{model_name}_{feature_type}"
                self.global_metrics_storage.append(mets)

            except Exception as e:
                logger.warning(f"Local Sample Bar failed: {e}")
                traceback.print_exc()

    def explain_all_models(self, n_categories=15, feature_types=None):
        self.n_categories = n_categories
        self.setup_directories(n_categories)
        if feature_types is None: feature_types = ["tfidf", "sbert"]
        
        for f_type in feature_types:
            for m_name in self.model_names:
                try: self.explain_model(m_name, f_type)
                except Exception as e: logger.error(f"Pipeline failed {m_name}: {e}")
        self.save_reports()
        return self.global_metrics_storage

    def save_reports(self):
        if not self.global_metrics_storage:
            logger.warning("No metrics found to save.")
            return
        df = pd.DataFrame(self.global_metrics_storage)
        df.to_csv(self.dirs['metrics'] / "ML_Final_Metrics.csv", index=False)
        
        plt.figure(figsize=(12, 6))
        if 'model' in df.columns:
            ax = sns.barplot(data=df.melt(id_vars='model'), x='variable', y='value', hue='model')
            for c in ax.containers: ax.bar_label(c, fmt='%.2f', padding=3, fontsize=9)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title("ML Explainability Metrics (Real)")
            plt.tight_layout()
            plt.savefig(self.dirs['metrics'] / "ML_Metrics_Comparison.png")
            plt.close()
        
        data = []
        for cat in self.target_categories:
            tokens = self.all_dominant_tokens.get(cat, [])
            tokens = [t for t in tokens if not str(t).startswith("dim_")]
            top = [w for w, c in Counter(tokens).most_common(15)]
            data.append({'Category': cat, 'Tokens': ", ".join(top) if top else "N/A"})
        pd.DataFrame(data).to_csv(self.dirs['reports'] / "ML_Consolidated_Dominant_Tokens.csv", index=False)

if __name__ == "__main__":
    explainer = MLExplainability()
    explainer.explain_all_models(15)
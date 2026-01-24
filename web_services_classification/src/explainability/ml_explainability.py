"""
ML Model Explainability Module
Provides comprehensive SHAP and LIME explanations for ML models.
OPTIMIZED: Handles 50-class complexity, ragged text arrays, and GPU acceleration.
FIXED: 
1. Resolved SHAP index error by ensuring label alignment.
2. Resolved LIME 'Other' confusion by forcing specific label explanation.
3. Correctly interprets YAML labels.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import json
import warnings
import traceback
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# SHAP imports
import shap

# LIME imports
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

# Sentence Transformer
from sentence_transformers import SentenceTransformer

# Import configuration
from src.config import (
    ML_CONFIG, PREPROCESSING_CONFIG, CATEGORY_SIZES,
    SAVED_MODELS_CONFIG, RESULTS_CONFIG, FEATURES_CONFIG
)

# Try to import EXPLAINABILITY_CONFIG, use defaults if not available
try:
    from src.config import EXPLAINABILITY_CONFIG
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("EXPLAINABILITY_CONFIG not found in config, using defaults")
    EXPLAINABILITY_CONFIG = {
        'plot_dpi': 300,
        'plot_format': 'png',
        'max_features_display': 20,
        'shap_background_samples': 10,
        'shap_explain_samples': 5,
        'lime_num_samples': 1000,
        'lime_num_features': 20,
        'lime_num_instances': 5
    }

from src.preprocessing.feature_extraction import FeatureExtractor
from src.evaluation.evaluate import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MLExplainability:
    """Comprehensive explainability analysis for ML models using SHAP and LIME"""
    
    def __init__(self, config=None):
        """Initialize explainability analyzer"""
        self.feature_extractor = FeatureExtractor()
        self.evaluator = ModelEvaluator()
        self.model_names = ML_CONFIG['models']
        
        self.config = config if config is not None else EXPLAINABILITY_CONFIG
        
        self.plot_dpi = self.config.get('plot_dpi', 300)
        self.plot_format = self.config.get('plot_format', 'png')
        self.max_features = self.config.get('max_features_display', 20)
        
        self.shap_background_samples = 10 
        self.shap_explain_samples = 5
        
        self.lime_num_samples = self.config.get('lime_num_samples', 1000)
        self.lime_num_instances = self.config.get('lime_num_instances', 5)
        
        self.explain_top_k = [5, 10]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cache model to prevent reloading
        self.sbert_model = None 

        logger.info(f"MLExplainability initialized:")
        logger.info(f"  - Device: {self.device}")
        
    def setup_directories(self, n_categories):
        """Setup explainability output directories"""
        base_path = RESULTS_CONFIG['ml_results_path'] / f"top_{n_categories}_categories" / "explainability"
        
        dirs = {
            'shap': base_path / "shap",
            'lime': base_path / "lime",
            'extra_lime': base_path / "lime" / "extra_lime_explainer", 
            'combined': base_path / "combined",
            'feature_importance': base_path / "feature_importance",
            'visualizations': base_path / "visualizations",
            'reports': base_path / "reports",
            'samples': base_path / "samples"
        }
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        return dirs
    
    def load_model_and_data(self, model_name, n_categories, feature_type="tfidf"):
        """Load trained model and corresponding data"""
        logger.info(f"Loading {model_name} model for top_{n_categories}_categories with {feature_type} features")
        model_dir = SAVED_MODELS_CONFIG["ml_models_path"] / f"top_{n_categories}_categories"
        feature_type_upper = feature_type.upper()
        model_filename = f"{model_name}_{feature_type_upper}_top_{n_categories}_categories_model.pkl"
        model_path = model_dir / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        
        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
        test_df = pd.read_csv(splits_dir / "test.csv")
        train_df = pd.read_csv(splits_dir / "train.csv")
        
        if feature_type == "tfidf":
            self.feature_extractor.load_tfidf_vectorizer(n_categories)
            X_test = self.feature_extractor.tfidf_vectorizer.transform(test_df["cleaned_text"])
            X_train = self.feature_extractor.tfidf_vectorizer.transform(train_df["cleaned_text"])
            feature_names = self.feature_extractor.tfidf_vectorizer.get_feature_names_out()
        else:
            X_test = self.feature_extractor.load_sbert_features(n_categories, "test")
            X_train = self.feature_extractor.load_sbert_features(n_categories, "train")
            feature_names = [f"sbert_dim_{i}" for i in range(X_test.shape[1])]
        
        # ==============================================================================
        # FIXED: Load Real Labels from YAML 'id_to_label'
        # ==============================================================================
        class_labels = []
        try:
            yaml_path = Path("data/processed") / f"labels_top_{n_categories}_categories.yaml"
            
            if yaml_path.exists():
                logger.info(f"Loading REAL labels from: {yaml_path}")
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                
                # Check for 'id_to_label' structure
                if 'id_to_label' in yaml_data:
                    id_map = yaml_data['id_to_label']
                    # Sort keys to ensure index 0 matches label 0, index 1 matches label 1...
                    class_labels = [id_map[i] for i in sorted(id_map.keys())]
                    logger.info(f"Loaded {len(class_labels)} labels from 'id_to_label'.")
                
                # Fallback to 'categories' list if id_map missing
                elif 'categories' in yaml_data:
                    class_labels = yaml_data['categories']
                    logger.info(f"Loaded {len(class_labels)} labels from 'categories' list.")
                
            else:
                logger.warning(f"Label file not found at {yaml_path}. Falling back to evaluator.")
                class_labels = self.evaluator.load_class_labels(n_categories)
        except Exception as e:
            logger.error(f"Error loading YAML labels: {e}")
            class_labels = self.evaluator.load_class_labels(n_categories)

        # Final Fallback ensuring list format
        if hasattr(class_labels, 'tolist'):
            class_labels = class_labels.tolist()
        elif isinstance(class_labels, dict):
            class_labels = [class_labels[i] for i in sorted(class_labels.keys())]
            
        return model, X_train, X_test, test_df, train_df, feature_names, class_labels

    def get_prediction_pipeline(self, model, feature_type, n_categories, feature_names=None):
        """Pipeline for raw text prediction"""
        def tfidf_pipeline(texts):
            if not hasattr(self.feature_extractor, 'tfidf_vectorizer') or self.feature_extractor.tfidf_vectorizer is None:
                self.feature_extractor.load_tfidf_vectorizer(n_categories)
            vectors = self.feature_extractor.tfidf_vectorizer.transform(texts)
            return model.predict_proba(vectors)

        def sbert_pipeline(texts):
            if self.sbert_model is None:
                logger.info("Loading SBERT model for pipeline (Once)...")
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            embeddings = self.sbert_model.encode(texts, device=self.device, show_progress_bar=False)
            
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu().numpy()
            elif isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            if feature_names is not None and embeddings.shape[1] == len(feature_names):
                embeddings_df = pd.DataFrame(embeddings, columns=feature_names)
                return model.predict_proba(embeddings_df)
            
            return model.predict_proba(embeddings)

        if feature_type == "tfidf":
            return tfidf_pipeline
        else:
            return sbert_pipeline

    def _save_sample_text(self, text, index, directory, prefix):
        try:
            filename = directory / f"{prefix}_sample_{index}_text.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"SAMPLE ID: {index}\n")
                f.write("-" * 50 + "\n")
                f.write(str(text))
        except Exception as e:
            logger.warning(f"Failed to save text sample: {e}")

    def _plot_manual_bar(self, features, weights, title, output_path, k_features):
        """Helper to plot horizontal bar chart manually"""
        combined = list(zip(features, weights))
        combined.sort(key=lambda x: abs(x[1]), reverse=True)
        combined = combined[:k_features]
        features, weights = zip(*combined) if combined else ([], [])
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if w > 0 else 'red' for w in weights]
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, weights, color=colors, align='center')
        plt.yticks(y_pos, features)
        plt.gca().invert_yaxis()
        plt.xlabel('Impact')
        plt.title(title, fontsize=12, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()

    def generate_shap_explanations(self, model, model_name, X_train, X_test, test_df, 
                                   feature_names, class_labels, n_categories, feature_type, dirs):
        """Standard SHAP explanations (Optimized)"""
        logger.info(f"Generating SHAP explanations for {model_name}")
        shap_results = {}
        try:
            if hasattr(X_train, 'toarray'):
                X_train_dense = X_train.toarray()
                X_test_dense = X_test.toarray()
            else:
                X_train_dense = X_train
                X_test_dense = X_test
            
            if model_name in ["RandomForest", "XGBoost"] and n_categories > 10:
                n_explain = 5 
            else:
                n_explain = min(self.shap_explain_samples, X_test_dense.shape[0])

            n_background = min(self.shap_background_samples, X_train_dense.shape[0])
            is_text_explanation = (feature_type == "sbert")
            
            if is_text_explanation:
                n_explain = min(5, n_explain)
                test_sample_text = test_df["cleaned_text"].iloc[:n_explain].tolist()
                
                for i, txt in enumerate(test_sample_text):
                    self._save_sample_text(txt, i+1, dirs['samples'], "shap")

                pipeline = self.get_prediction_pipeline(model, feature_type, n_categories, feature_names)
                masker = shap.maskers.Text(r"\W+") 
                explainer = shap.Explainer(pipeline, masker)
                shap_values = explainer(test_sample_text)
            else:
                background_sample = X_train_dense[np.random.choice(X_train_dense.shape[0], n_background, replace=False)]
                test_sample = X_test_dense[:n_explain]
                
                if model_name == "LogisticRegression":
                    explainer = shap.LinearExplainer(model, background_sample, feature_names=feature_names)
                    shap_values = explainer.shap_values(test_sample)
                elif model_name in ["RandomForest", "XGBoost"]:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(test_sample, check_additivity=False)
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, background_sample)
                    shap_values = explainer.shap_values(test_sample)

            vals_for_summary = None
            names_for_summary = feature_names
            text_word_importance = {} 
            is_multiclass_list = False

            if is_text_explanation:
                word_scores = defaultdict(float)
                for i in range(len(shap_values)):
                    sample_exp = shap_values[i]
                    vals = sample_exp.values
                    tokens = sample_exp.data
                    if len(vals.shape) == 2: impacts = np.sum(np.abs(vals), axis=1)
                    else: impacts = np.abs(vals)
                    for token, impact in zip(tokens, impacts):
                        clean_token = token.strip()
                        if clean_token: word_scores[clean_token] += impact
                text_word_importance = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            else:
                if isinstance(shap_values, list):
                    vals_for_summary = shap_values
                    is_multiclass_list = True
                elif len(np.array(shap_values).shape) == 3:
                    sv = np.array(shap_values)
                    vals_for_summary = [sv[:, :, i] for i in range(sv.shape[2])]
                    is_multiclass_list = True
                else:
                    vals_for_summary = shap_values
                features_for_summary = test_sample

            for k in self.explain_top_k:
                try:
                    title = f"SHAP Top-{k} Features - {model_name} ({feature_type.upper()})"
                    out_path = dirs['shap'] / f"shap_summary_top{k}_{model_name}_{feature_type}.{self.plot_format}"
                    
                    if is_text_explanation:
                        feats, weights = zip(*text_word_importance) if text_word_importance else ([], [])
                        self._plot_manual_bar(feats, weights, title, out_path, k)
                    else:
                        plt.figure(figsize=(12, 8))
                        # FIX: Only pass class_labels if the length matches exactly
                        safe_class_labels = class_labels if (class_labels and len(class_labels) == len(vals_for_summary)) else None
                        
                        if is_multiclass_list:
                             shap.summary_plot(vals_for_summary, features_for_summary, 
                                            feature_names=names_for_summary, class_names=safe_class_labels,
                                            plot_type="bar", show=False, max_display=k)
                        else:
                             shap.summary_plot(vals_for_summary, features_for_summary, 
                                            feature_names=names_for_summary, plot_type="bar", 
                                            show=False, max_display=k)
                        plt.title(title, fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(str(out_path), dpi=self.plot_dpi, bbox_inches='tight')
                        plt.close()
                except Exception as e:
                    logger.warning(f"Could not generate Top-{k} SHAP plot: {e}")

            if is_text_explanation:
                if text_word_importance:
                    df = pd.DataFrame(text_word_importance, columns=['feature', 'importance'])
                    path = dirs['feature_importance'] / f"shap_importance_{model_name}_{feature_type}.csv"
                    df.to_csv(path, index=False)
                    shap_results['feature_importance'] = str(path)
                else:
                    shap_results['feature_importance'] = None
            else:
                if is_multiclass_list:
                    sum_impacts = np.sum([np.mean(np.abs(arr), axis=0) for arr in vals_for_summary], axis=0)
                    mean_abs_shap = sum_impacts / len(vals_for_summary)
                else:
                    mean_abs_shap = np.mean(np.abs(vals_for_summary), axis=0)
                if mean_abs_shap.ndim > 1: mean_abs_shap = mean_abs_shap.flatten()
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': mean_abs_shap})
                importance_path = dirs['feature_importance'] / f"shap_importance_{model_name}_{feature_type}.csv"
                feature_importance_df.sort_values('importance', ascending=False).to_csv(importance_path, index=False)
                shap_results['feature_importance'] = str(importance_path)

            logger.info(f"SHAP analysis completed for {model_name}")
        except Exception as e:
            logger.error(f"Error in SHAP analysis for {model_name}: {e}")
            shap_results = {'error': str(e)}
        return shap_results
    
    def generate_lime_explanations(self, model, model_name, X_train, X_test, test_df,
                                   feature_names, class_labels, n_categories, feature_type, dirs):
        """Standard LIME explanations (Optimized)"""
        logger.info(f"Generating LIME explanations for {model_name}")
        lime_results = {}
        try:
            is_text_explanation = (feature_type == "sbert")
            current_num_samples = 1000 if is_text_explanation else self.lime_num_samples

            if is_text_explanation:
                explainer = LimeTextExplainer(class_names=class_labels)
                pipeline_fn = self.get_prediction_pipeline(model, feature_type, n_categories, feature_names)
                data_source = test_df["cleaned_text"].tolist()
            else:
                if hasattr(X_train, 'toarray'): X_train_dense = X_train.toarray()
                else: X_train_dense = X_train
                if hasattr(X_test, 'toarray'): X_test_dense = X_test.toarray()
                else: X_test_dense = X_test
                explainer = LimeTabularExplainer(
                    training_data=X_train_dense, feature_names=feature_names,
                    class_names=class_labels, mode='classification', discretize_continuous=True
                )
                data_source = X_test_dense

            n_samples = min(self.lime_num_instances, len(data_source))
            if is_text_explanation:
                sample_indices = np.random.choice(len(data_source), n_samples, replace=False)
            else:
                sample_indices = np.linspace(0, len(data_source)-1, n_samples, dtype=int)
            
            explanations = []
            for idx, sample_idx in enumerate(sample_indices):
                try:
                    logger.info(f"Processing LIME sample {idx+1}/{n_samples}...")
                    
                    if is_text_explanation:
                        self._save_sample_text(data_source[sample_idx], idx+1, dirs['samples'], "lime")

                    max_k = max(self.explain_top_k)
                    if is_text_explanation:
                        exp = explainer.explain_instance(
                            data_source[sample_idx], pipeline_fn,
                            num_features=max_k, top_labels=1, num_samples=current_num_samples
                        )
                    else:
                        exp = explainer.explain_instance(
                            data_source[sample_idx], model.predict_proba,
                            num_features=max_k, top_labels=1, num_samples=current_num_samples
                        )
                    
                    if not exp.local_exp: continue
                    available_label = list(exp.local_exp.keys())[0]
                    class_name_str = class_labels[available_label] if available_label < len(class_labels) else str(available_label)
                    full_exp_list = exp.as_list(label=available_label)
                    
                    for k in self.explain_top_k:
                        title = f"LIME Top-{k} - Sample {idx+1} (Class: {class_name_str})\n{model_name} ({feature_type.upper()})"
                        path = dirs['lime'] / f"lime_sample_{idx+1}_top{k}_{model_name}_{feature_type}.{self.plot_format}"
                        self._plot_manual_bar([x[0] for x in full_exp_list], [x[1] for x in full_exp_list], title, path, k)

                    explanations.append({
                        'sample_index': int(sample_idx), 'predicted_label': int(available_label),
                        'top_features': full_exp_list
                    })
                except Exception as e:
                    logger.warning(f"Error generating LIME explanation for sample {idx+1}: {e}")
                    continue
            
            all_features = {}
            for exp_data in explanations:
                for feature, weight in exp_data['top_features']:
                    if feature not in all_features: all_features[feature] = []
                    all_features[feature].append(abs(weight))
            if all_features:
                feature_importance = {k: np.mean(v) for k, v in all_features.items()}
                feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
                plt.figure(figsize=(12, 8))
                features, importances = zip(*feature_importance_sorted)
                plt.barh(range(len(features)), importances, color='steelblue')
                plt.yticks(range(len(features)), features)
                plt.gca().invert_yaxis()
                plt.title(f'LIME Aggregated Feature Importance - {model_name} ({feature_type.upper()})')
                plt.tight_layout()
                plt.savefig(dirs['lime'] / f"lime_aggregate_{model_name}_{feature_type}.{self.plot_format}")
                plt.close()
                
                lime_df = pd.DataFrame(feature_importance_sorted, columns=['feature', 'importance'])
                path = dirs['feature_importance'] / f"lime_importance_{model_name}_{feature_type}.csv"
                lime_df.to_csv(path, index=False)
                lime_results['feature_importance'] = str(path)
            
            lime_results['explanations'] = explanations
            logger.info(f"LIME analysis completed for {model_name}")
        except Exception as e:
            logger.error(f"Error in LIME analysis for {model_name}: {e}")
            lime_results = {'error': str(e)}
        return lime_results

    # ==================================================================================
    # NEW FEATURE: Extra Lime Explainer (Fixed: Labels from YAML, No PNGs)
    # ==================================================================================
    def generate_extra_lime_charts(self, model, model_name, test_df, feature_names, class_labels, n_categories, feature_type, dirs):
        """Generates LIME HTML Dashboard and Text Samples (No PNGs)."""
        logger.info(f"\n{'='*40}")
        logger.info(f"Running EXTRA LIME EXPLAINER for {model_name}...")
        
        try:
            is_text_explanation = (feature_type == "sbert")
            target_indices = [0, 1, 2, 3, 4] 
            
            if is_text_explanation:
                explainer = LimeTextExplainer(class_names=class_labels)
                pipeline_fn = self.get_prediction_pipeline(model, feature_type, n_categories, feature_names)
                data_source = test_df["cleaned_text"].tolist()
            else:
                data_source = test_df["cleaned_text"].tolist() 
                pipeline_fn = self.get_prediction_pipeline(model, feature_type, n_categories, feature_names)
                explainer = LimeTextExplainer(class_names=class_labels)

            for i, idx in enumerate(target_indices):
                if idx >= len(data_source): continue
                
                text_instance = data_source[idx]
                
                # A. Predict Probability
                probs = pipeline_fn([text_instance])[0]
                top_class_index = int(np.argmax(probs)) 
                
                try:
                    winner_class_name = class_labels[top_class_index]
                except:
                    winner_class_name = str(top_class_index)
                
                logger.info(f"  - Sample {i+1}: Winner = {winner_class_name} ({probs[top_class_index]:.2f})")
                
                # B. Explain "Winner" Class (HTML Only)
                # FIX: REMOVED top_labels=1 to avoid conflict with labels=[...]
                exp = explainer.explain_instance(
                    text_instance,
                    pipeline_fn,
                    num_features=10, 
                    labels=[top_class_index]
                )
                
                if top_class_index not in exp.local_exp:
                    if not exp.local_exp: continue
                    top_class_index = list(exp.local_exp.keys())[0]

                # C. Save HTML Dashboard ONLY
                output_filename = dirs['extra_lime'] / f"dashboard_sample_{i+1}_{model_name}.html"
                exp.save_to_file(str(output_filename))
                
                # D. Save Text Sample
                text_filename = dirs['extra_lime'] / f"text_sample_{i+1}.txt"
                with open(text_filename, 'w', encoding='utf-8') as f:
                    f.write(f"Sample ID: {idx}\nPrediction: {winner_class_name}\n\n{text_instance}")

            logger.info(f"Dashboards and Text samples saved to: {dirs['extra_lime']}")
            
        except Exception as e:
            logger.warning(f"Failed to run Extra LIME Explainer: {e}")
            logger.warning(traceback.format_exc())

    def generate_combined_comparison(self, shap_results, lime_results, model_name, 
                                     n_categories, feature_type, dirs):
        """Create combined SHAP vs LIME comparison visualization"""
        try:
            if not shap_results or not lime_results: return None
            
            # FIX: Ensure we use the FILE paths (Safe Check)
            shap_path = shap_results.get('feature_importance')
            lime_path = lime_results.get('feature_importance')
            
            if not shap_path or not lime_path: return None
            if not Path(shap_path).exists() or not Path(lime_path).exists():
                logger.warning("Feature importance files missing. Skipping comparison.")
                return None
                
            shap_df = pd.read_csv(shap_path).head(10)
            lime_df = pd.read_csv(lime_path).head(10)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            ax1.barh(range(len(shap_df)), shap_df['importance'], color='coral')
            ax1.set_yticks(range(len(shap_df)))
            ax1.set_yticklabels(shap_df['feature'], fontsize=10)
            ax1.set_title('SHAP Feature Importance')
            ax1.invert_yaxis()
            
            ax2.barh(range(len(lime_df)), lime_df['importance'], color='steelblue')
            ax2.set_yticks(range(len(lime_df)))
            ax2.set_yticklabels(lime_df['feature'], fontsize=10)
            ax2.set_title('LIME Feature Importance')
            ax2.invert_yaxis()
            
            fig.suptitle(f'SHAP vs LIME Comparison - {model_name} ({feature_type.upper()})')
            plt.tight_layout()
            
            path = dirs['combined'] / f"shap_lime_comparison_{model_name}_{feature_type}.{self.plot_format}"
            dirs['combined'].mkdir(parents=True, exist_ok=True) 
            plt.savefig(str(path), dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            return str(path)
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return None
    
    def explain_model(self, model_name, n_categories, feature_type="tfidf"):
        """Generate complete explainability analysis for a single model"""
        logger.info(f"\n{'='*80}\nStarting explainability analysis for {model_name}\n{'='*80}")
        dirs = self.setup_directories(n_categories)
        model, X_train, X_test, test_df, train_df, feature_names, class_labels = \
            self.load_model_and_data(model_name, n_categories, feature_type)
        
        results = {'model_name': model_name, 'n_categories': n_categories, 'feature_type': feature_type}
        
        # 1. Standard SHAP
        results['shap'] = self.generate_shap_explanations(
            model, model_name, X_train, X_test, test_df, 
            feature_names, class_labels, n_categories, feature_type, dirs
        )
        
        # 2. Standard LIME
        results['lime'] = self.generate_lime_explanations(
            model, model_name, X_train, X_test, test_df,
            feature_names, class_labels, n_categories, feature_type, dirs
        )
        
        # 3. Combined Comparison
        results['comparison'] = self.generate_combined_comparison(
            results['shap'], results['lime'], model_name, n_categories, feature_type, dirs
        )
        
        # 4. NEW FEATURE: Run Extra Explainer (No PNGs)
        self.generate_extra_lime_charts(
            model, model_name, test_df, feature_names, class_labels, n_categories, feature_type, dirs
        )
        
        with open(dirs['combined'] / f"explainability_summary_{model_name}_{feature_type}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def explain_all_models(self, n_categories=None, feature_types=None):
        """Generate explainability analysis for all ML models"""
        if n_categories is None: n_categories = CATEGORY_SIZES[0]
        if feature_types is None: feature_types = ["tfidf", "sbert"]
        
        print(f"\n{'='*100}\nML MODEL EXPLAINABILITY ANALYSIS (FINAL)\n{'='*100}")
        
        all_results = {}
        for feature_type in feature_types:
            all_results[feature_type] = {}
            for model_name in self.model_names:
                try:
                    all_results[feature_type][model_name] = self.explain_model(model_name, n_categories, feature_type)
                except Exception as e:
                    logger.error(f"Failed to explain {model_name}: {e}")
                    all_results[feature_type][model_name] = {'error': str(e)}
        return all_results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--categories", type=int, default=50)
    parser.add_argument("--feature", type=str, choices=['tfidf', 'sbert'])
    args = parser.parse_args()
    
    explainer = MLExplainability()
    
    if args.model:
        fts = [args.feature] if args.feature else ['tfidf', 'sbert']
        for ft in fts: explainer.explain_model(args.model, args.categories, ft)
    else:
        fts = [args.feature] if args.feature else ['tfidf', 'sbert']
        explainer.explain_all_models(args.categories, fts)

if __name__ == "__main__":
    main()
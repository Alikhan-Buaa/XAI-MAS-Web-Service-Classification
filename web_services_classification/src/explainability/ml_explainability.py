"""
ML Model Explainability Module
Provides comprehensive SHAP and LIME explanations for ML models
Enhanced version with proper config usage
"""

import pandas as pd
import numpy as np
import joblib
import logging
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP imports
import shap

# LIME imports
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

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
    # Default configuration if not in config.py
    EXPLAINABILITY_CONFIG = {
        'plot_dpi': 300,
        'plot_format': 'png',
        'max_features_display': 20,
        'shap_background_samples': 100,
        'shap_explain_samples': 50,
        'lime_num_samples': 5000,
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
        """
        Initialize explainability analyzer
        
        Args:
            config: Optional config dict to override defaults
        """
        self.feature_extractor = FeatureExtractor()
        self.evaluator = ModelEvaluator()
        self.model_names = ML_CONFIG['models']
        
        # Use provided config or default EXPLAINABILITY_CONFIG
        self.config = config if config is not None else EXPLAINABILITY_CONFIG
        
        # Extract configuration values
        self.plot_dpi = self.config.get('plot_dpi', 300)
        self.plot_format = self.config.get('plot_format', 'png')
        self.max_features = self.config.get('max_features_display', 20)
        self.shap_background_samples = self.config.get('shap_background_samples', 100)
        self.shap_explain_samples = self.config.get('shap_explain_samples', 50)
        self.lime_num_samples = self.config.get('lime_num_samples', 5000)
        self.lime_num_features = self.config.get('lime_num_features', 20)
        self.lime_num_instances = self.config.get('lime_num_instances', 5)
        
        logger.info(f"MLExplainability initialized with config:")
        logger.info(f"  - Plot DPI: {self.plot_dpi}")
        logger.info(f"  - Max features display: {self.max_features}")
        logger.info(f"  - SHAP background samples: {self.shap_background_samples}")
        logger.info(f"  - SHAP explain samples: {self.shap_explain_samples}")
        logger.info(f"  - LIME instances: {self.lime_num_instances}")
        
    def setup_directories(self, n_categories):
        """Setup explainability output directories"""
        base_path = RESULTS_CONFIG['ml_results_path'] / f"top_{n_categories}_categories" / "explainability"
        
        dirs = {
            'shap': base_path / "shap",
            'lime': base_path / "lime",
            'combined': base_path / "combined",
            'feature_importance': base_path / "feature_importance"
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    def load_model_and_data(self, model_name, n_categories, feature_type="tfidf"):
        """Load trained model and corresponding data"""
        logger.info(f"Loading {model_name} model for top_{n_categories}_categories with {feature_type} features")
        
        # Load model
        model_dir = SAVED_MODELS_CONFIG["ml_models_path"] / f"top_{n_categories}_categories"
        # Match actual saved model pattern: ModelName_FEATURETYPE_top_N_categories_model.pkl
        feature_type_upper = feature_type.upper()  # TFIDF or SBERT
        model_filename = f"{model_name}_{feature_type_upper}_top_{n_categories}_categories_model.pkl"
        model_path = model_dir / model_filename
        

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
        # Load test data
        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
        test_df = pd.read_csv(splits_dir / "test.csv")
        train_df = pd.read_csv(splits_dir / "train.csv")
        
        # Load features
        if feature_type == "tfidf":
            self.feature_extractor.load_tfidf_vectorizer(n_categories)
            X_test = self.feature_extractor.tfidf_vectorizer.transform(test_df["cleaned_text"])
            X_train = self.feature_extractor.tfidf_vectorizer.transform(train_df["cleaned_text"])
            feature_names = self.feature_extractor.tfidf_vectorizer.get_feature_names_out()
        else:
            X_test = self.feature_extractor.load_sbert_features(n_categories, "test")
            X_train = self.feature_extractor.load_sbert_features(n_categories, "train")
            feature_names = [f"sbert_dim_{i}" for i in range(X_test.shape[1])]
        
        # Load class labels
        class_labels = self.evaluator.load_class_labels(n_categories)
        
        return model, X_train, X_test, test_df, train_df, feature_names, class_labels
    
    def generate_shap_explanations(self, model, model_name, X_train, X_test, test_df, 
                                   feature_names, class_labels, n_categories, feature_type, dirs):
        """Generate comprehensive SHAP explanations using config settings"""
        logger.info(f"Generating SHAP explanations for {model_name}")
        
        shap_results = {}
        
        try:
            # Convert sparse matrices to dense for SHAP
            if hasattr(X_train, 'toarray'):
                X_train_dense = X_train.toarray()
                X_test_dense = X_test.toarray()
            else:
                X_train_dense = X_train
                X_test_dense = X_test
            
            # Use config values for sampling
            n_background = min(self.shap_background_samples, X_train_dense.shape[0])
            background_sample = X_train_dense[np.random.choice(X_train_dense.shape[0], n_background, replace=False)]
            
            n_explain = min(self.shap_explain_samples, X_test_dense.shape[0])
            test_sample = X_test_dense[:n_explain]
            
            logger.info(f"Using {n_background} background samples and {n_explain} test samples")
            
            # Create SHAP explainer based on model type
            if model_name == "LogisticRegression":
                explainer = shap.LinearExplainer(model, background_sample, feature_names=feature_names)
            elif model_name in ["RandomForest", "XGBoost"]:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, background_sample)
            
            # Calculate SHAP values
            logger.info("Calculating SHAP values...")
            shap_values = explainer.shap_values(test_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values_array = np.array(shap_values)
                if len(shap_values_array.shape) == 3:
                    shap_values_avg = np.mean(np.abs(shap_values_array), axis=0)
                else:
                    shap_values_avg = shap_values_array
            else:
                shap_values_avg = shap_values
            
            # 1. Summary Plot (Bar) - Global Feature Importance
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_avg, test_sample, feature_names=feature_names, 
                            plot_type="bar", show=False, max_display=self.max_features)
            plt.title(f"SHAP Feature Importance - {model_name} ({feature_type.upper()})\nTop {n_categories} Categories", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            summary_bar_path = dirs['shap'] / f"shap_summary_bar_{model_name}_{feature_type}_top_{n_categories}.{self.plot_format}"
            plt.savefig(summary_bar_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary bar plot saved: {summary_bar_path}")
            
            # 2. Summary Plot (Beeswarm) - Feature Impact Distribution
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values_avg, test_sample, feature_names=feature_names, 
                            show=False, max_display=self.max_features)
            plt.title(f"SHAP Feature Impact Distribution - {model_name} ({feature_type.upper()})\nTop {n_categories} Categories", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            summary_bee_path = dirs['shap'] / f"shap_summary_beeswarm_{model_name}_{feature_type}_top_{n_categories}.{self.plot_format}"
            plt.savefig(summary_bee_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP beeswarm plot saved: {summary_bee_path}")
            
            # 3. Waterfall Plot for First Prediction
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[0] if len(expected_value) > 0 else 0
            else:
                expected_value = 0
            
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_obj = shap.Explanation(values=shap_values[0][0], 
                                           base_values=expected_value,
                                           data=test_sample[0],
                                           feature_names=feature_names)
            else:
                shap_obj = shap.Explanation(values=shap_values_avg[0],
                                           base_values=expected_value,
                                           data=test_sample[0],
                                           feature_names=feature_names)
            
            shap.waterfall_plot(shap_obj, max_display=self.max_features, show=False)
            plt.title(f"SHAP Waterfall - First Prediction\n{model_name} ({feature_type.upper()})", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            waterfall_path = dirs['shap'] / f"shap_waterfall_{model_name}_{feature_type}_top_{n_categories}.{self.plot_format}"
            plt.savefig(waterfall_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP waterfall plot saved: {waterfall_path}")
            
            # 4. Dependence Plot for Top Feature
            mean_abs_shap = np.mean(np.abs(shap_values_avg), axis=0)
            top_feature_idx = np.argmax(mean_abs_shap)
            top_feature_name = feature_names[top_feature_idx]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(top_feature_idx, shap_values_avg, test_sample, 
                               feature_names=feature_names, show=False)
            plt.title(f"SHAP Dependence - {top_feature_name}\n{model_name} ({feature_type.upper()})", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            dependence_path = dirs['shap'] / f"shap_dependence_{model_name}_{feature_type}_top_{n_categories}.{self.plot_format}"
            plt.savefig(dependence_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP dependence plot saved: {dependence_path}")
            
            # 5. Force Plot for First Prediction
            try:
                plt.figure(figsize=(20, 3))
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    shap.force_plot(expected_value, shap_values[0][0], test_sample[0], 
                                  feature_names=feature_names, matplotlib=True, show=False)
                else:
                    shap.force_plot(expected_value, shap_values_avg[0], test_sample[0],
                                  feature_names=feature_names, matplotlib=True, show=False)
                
                plt.title(f"SHAP Force Plot - First Prediction\n{model_name} ({feature_type.upper()})", 
                         fontsize=12, fontweight='bold')
                force_path = dirs['shap'] / f"shap_force_{model_name}_{feature_type}_top_{n_categories}.{self.plot_format}"
                plt.savefig(force_path, dpi=self.plot_dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP force plot saved: {force_path}")
            except Exception as e:
                logger.warning(f"Could not generate force plot: {e}")
            
            # Save feature importance data
            mean_abs_shap = np.mean(np.abs(shap_values_avg), axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            importance_path = dirs['feature_importance'] / f"shap_importance_{model_name}_{feature_type}_top_{n_categories}.csv"
            feature_importance_df.to_csv(importance_path, index=False)
            logger.info(f"SHAP feature importance saved: {importance_path}")
            
            shap_results = {
                'summary_bar': str(summary_bar_path),
                'summary_beeswarm': str(summary_bee_path),
                'waterfall': str(waterfall_path),
                'dependence': str(dependence_path),
                'feature_importance': str(importance_path),
                'top_features': feature_importance_df.head(10).to_dict('records')
            }
            
            logger.info(f"SHAP analysis completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis for {model_name}: {e}")
            shap_results = {'error': str(e)}
        
        return shap_results
    
    def generate_lime_explanations(self, model, model_name, X_train, X_test, test_df,
                                   feature_names, class_labels, n_categories, feature_type, dirs):
        """Generate comprehensive LIME explanations using config settings"""
        logger.info(f"Generating LIME explanations for {model_name}")
        
        lime_results = {}
        
        try:
            # Convert sparse matrices to dense for LIME
            if hasattr(X_train, 'toarray'):
                X_train_dense = X_train.toarray()
                X_test_dense = X_test.toarray()
            else:
                X_train_dense = X_train
                X_test_dense = X_test
            
            # Create LIME explainer for tabular data
            explainer = LimeTabularExplainer(
                training_data=X_train_dense,
                feature_names=feature_names,
                class_names=class_labels,
                mode='classification',
                discretize_continuous=True
            )
            
            # Use config value for number of instances
            n_samples = min(self.lime_num_instances, X_test_dense.shape[0])
            sample_indices = np.linspace(0, X_test_dense.shape[0]-1, n_samples, dtype=int)
            
            explanations = []
            
            for idx, sample_idx in enumerate(sample_indices):
                instance = X_test_dense[sample_idx]
                
                # Generate explanation with config num_features
                exp = explainer.explain_instance(
                    data_row=instance,
                    predict_fn=model.predict_proba,
                    num_features=self.lime_num_features,
                    top_labels=3
                )
                
                # Save explanation visualization
                fig = exp.as_pyplot_figure()
                fig.suptitle(f"LIME Explanation - Sample {idx+1}\n{model_name} ({feature_type.upper()})", 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                lime_path = dirs['lime'] / f"lime_explanation_{model_name}_{feature_type}_sample_{idx+1}_top_{n_categories}.{self.plot_format}"
                plt.savefig(lime_path, dpi=self.plot_dpi, bbox_inches='tight')
                plt.close()
                
                # Get explanation as list
                exp_list = exp.as_list()
                
                explanations.append({
                    'sample_index': int(sample_idx),
                    'plot_path': str(lime_path),
                    'top_features': exp_list[:10]
                })
                
                logger.info(f"LIME explanation {idx+1}/{n_samples} saved: {lime_path}")
            
            # Create summary visualization
            all_features = {}
            for exp_data in explanations:
                for feature, weight in exp_data['top_features']:
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(abs(weight))
            
            # Calculate average importance
            feature_importance = {k: np.mean(v) for k, v in all_features.items()}
            feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
            
            # Plot aggregated feature importance
            plt.figure(figsize=(12, 8))
            features, importances = zip(*feature_importance_sorted)
            plt.barh(range(len(features)), importances, color='steelblue')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Average Absolute Weight', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title(f'LIME Aggregated Feature Importance - {model_name} ({feature_type.upper()})\nTop {n_categories} Categories',
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            aggregate_path = dirs['lime'] / f"lime_aggregate_{model_name}_{feature_type}_top_{n_categories}.{self.plot_format}"
            plt.savefig(aggregate_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"LIME aggregate plot saved: {aggregate_path}")
            
            # Save feature importance data
            lime_importance_df = pd.DataFrame(feature_importance_sorted, columns=['feature', 'importance'])
            importance_path = dirs['feature_importance'] / f"lime_importance_{model_name}_{feature_type}_top_{n_categories}.csv"
            lime_importance_df.to_csv(importance_path, index=False)
            
            lime_results = {
                'explanations': explanations,
                'aggregate_plot': str(aggregate_path),
                'feature_importance': str(importance_path),
                'top_features': lime_importance_df.head(10).to_dict('records')
            }
            
            logger.info(f"LIME analysis completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Error in LIME analysis for {model_name}: {e}")
            lime_results = {'error': str(e)}
        
        return lime_results
    
    def generate_combined_comparison(self, shap_results, lime_results, model_name, 
                                     n_categories, feature_type, dirs):
        """Create combined SHAP vs LIME comparison visualization"""
        try:
            logger.info(f"Generating combined SHAP-LIME comparison for {model_name}")
            
            # Load importance data
            shap_importance_path = Path(shap_results.get('feature_importance', ''))
            lime_importance_path = Path(lime_results.get('feature_importance', ''))
            
            if not (shap_importance_path.exists() and lime_importance_path.exists()):
                logger.warning("Missing importance data for comparison")
                return None
            
            shap_df = pd.read_csv(shap_importance_path).head(15)
            lime_df = pd.read_csv(lime_importance_path).head(15)
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # SHAP importance
            ax1.barh(range(len(shap_df)), shap_df['importance'], color='coral')
            ax1.set_yticks(range(len(shap_df)))
            ax1.set_yticklabels(shap_df['feature'], fontsize=9)
            ax1.set_xlabel('SHAP Importance', fontsize=12, fontweight='bold')
            ax1.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
            ax1.invert_yaxis()
            ax1.grid(axis='x', alpha=0.3)
            
            # LIME importance
            ax2.barh(range(len(lime_df)), lime_df['importance'], color='steelblue')
            ax2.set_yticks(range(len(lime_df)))
            ax2.set_yticklabels(lime_df['feature'], fontsize=9)
            ax2.set_xlabel('LIME Importance', fontsize=12, fontweight='bold')
            ax2.set_title('LIME Feature Importance', fontsize=13, fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(axis='x', alpha=0.3)
            
            fig.suptitle(f'SHAP vs LIME Feature Importance Comparison\n{model_name} ({feature_type.upper()}) - Top {n_categories} Categories',
                        fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            comparison_path = dirs['combined'] / f"shap_lime_comparison_{model_name}_{feature_type}_top_{n_categories}.{self.plot_format}"
            plt.savefig(comparison_path, dpi=self.plot_dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Combined comparison saved: {comparison_path}")
            return str(comparison_path)
            
        except Exception as e:
            logger.error(f"Error generating combined comparison: {e}")
            return None
    
    def explain_model(self, model_name, n_categories, feature_type="tfidf"):
        """Generate complete explainability analysis for a single model"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting explainability analysis for {model_name}")
        logger.info(f"Categories: {n_categories}, Feature Type: {feature_type}")
        logger.info(f"Using config: SHAP samples={self.shap_background_samples}/{self.shap_explain_samples}, LIME instances={self.lime_num_instances}")
        logger.info(f"{'='*80}\n")
        
        # Setup directories
        dirs = self.setup_directories(n_categories)
        
        # Load model and data
        model, X_train, X_test, test_df, train_df, feature_names, class_labels = \
            self.load_model_and_data(model_name, n_categories, feature_type)
        
        results = {
            'model_name': model_name,
            'n_categories': n_categories,
            'feature_type': feature_type,
            'n_test_samples': X_test.shape[0],
            'n_features': X_test.shape[1],
            'config': {
                'plot_dpi': self.plot_dpi,
                'max_features': self.max_features,
                'shap_background_samples': self.shap_background_samples,
                'shap_explain_samples': self.shap_explain_samples,
                'lime_num_instances': self.lime_num_instances
            }
        }
        
        # Generate SHAP explanations
        logger.info("\n" + "="*60)
        logger.info("SHAP ANALYSIS")
        logger.info("="*60)
        shap_results = self.generate_shap_explanations(
            model, model_name, X_train, X_test, test_df, 
            feature_names, class_labels, n_categories, feature_type, dirs
        )
        results['shap'] = shap_results
        
        # Generate LIME explanations
        logger.info("\n" + "="*60)
        logger.info("LIME ANALYSIS")
        logger.info("="*60)
        lime_results = self.generate_lime_explanations(
            model, model_name, X_train, X_test, test_df,
            feature_names, class_labels, n_categories, feature_type, dirs
        )
        results['lime'] = lime_results
        
        # Generate combined comparison
        logger.info("\n" + "="*60)
        logger.info("COMBINED COMPARISON")
        logger.info("="*60)
        comparison_path = self.generate_combined_comparison(
            shap_results, lime_results, model_name, n_categories, feature_type, dirs
        )
        results['comparison'] = comparison_path
        
        # Save results summary
        summary_path = dirs['combined'] / f"explainability_summary_{model_name}_{feature_type}_top_{n_categories}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nExplainability summary saved: {summary_path}")
        logger.info(f"{'='*80}\n")
        
        return results
    
    def explain_all_models(self, n_categories=None, feature_types=None):
        """Generate explainability analysis for all ML models"""
        if n_categories is None:
            n_categories = CATEGORY_SIZES[0]
        
        if feature_types is None:
            feature_types = ["tfidf", "sbert"]
        
        all_results = {}
        
        print(f"\n{'='*100}")
        print(f"ML MODEL EXPLAINABILITY ANALYSIS")
        print(f"{'='*100}")
        print(f"Category Size: {n_categories}")
        print(f"Feature Types: {feature_types}")
        print(f"Models: {self.model_names}")
        print(f"Config: SHAP bg={self.shap_background_samples}, explain={self.shap_explain_samples}, LIME instances={self.lime_num_instances}")
        print(f"{'='*100}\n")
        
        for feature_type in feature_types:
            all_results[feature_type] = {}
            
            for model_name in self.model_names:
                try:
                    print(f"\n{'#'*80}")
                    print(f"Processing: {model_name} with {feature_type.upper()} features")
                    print(f"{'#'*80}\n")
                    
                    results = self.explain_model(model_name, n_categories, feature_type)
                    all_results[feature_type][model_name] = results
                    
                    print(f"\n✓ Successfully completed explainability for {model_name} ({feature_type})")
                    
                except Exception as e:
                    logger.error(f"Failed to explain {model_name} with {feature_type}: {e}")
                    print(f"\n✗ Failed: {model_name} ({feature_type}) - {e}")
                    all_results[feature_type][model_name] = {'error': str(e)}
        
        # Save overall results
        overall_results_path = RESULTS_CONFIG['ml_results_path'] / f"top_{n_categories}_categories" / "explainability" / "all_models_explainability.json"
        overall_results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(overall_results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n{'='*100}")
        print(f"EXPLAINABILITY ANALYSIS COMPLETED")
        print(f"{'='*100}")
        print(f"Overall results saved: {overall_results_path}")
        print(f"Individual results located in: results/ml/top_{n_categories}_categories/explainability/")
        print(f"{'='*100}\n")
        
        return all_results


def main():
    """Main function to run ML explainability analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Model Explainability Analysis")
    parser.add_argument("--model", type=str, help="Specific model name (e.g., LogisticRegression)")
    parser.add_argument("--categories", type=int, default=50, help="Number of categories")
    parser.add_argument("--feature", type=str, choices=['tfidf', 'sbert'], help="Specific feature type")
    
    args = parser.parse_args()
    
    explainer = MLExplainability()
    
    if args.model:
        # Explain single model
        feature_types = [args.feature] if args.feature else ['tfidf', 'sbert']
        for ft in feature_types:
            explainer.explain_model(args.model, args.categories, ft)
    else:
        # Explain all models
        feature_types = [args.feature] if args.feature else ['tfidf', 'sbert']
        explainer.explain_all_models(args.categories, feature_types)


if __name__ == "__main__":
    main()
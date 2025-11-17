"""
Machine Learning Models for Web Service Classification
Enhanced with standardized naming and proper result handling
"""

import pandas as pd
import numpy as np
import joblib
import logging
import time
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import xgboost as xgb
from math import pi

# Import configuration and utilities
from src.config import (
    ML_CONFIG, PREPROCESSING_CONFIG,
    CATEGORY_SIZES, SAVED_MODELS_CONFIG, RESULTS_CONFIG
)
from src.preprocessing.feature_extraction import FeatureExtractor
from src.evaluation.evaluate import ModelEvaluator
from src.utils.utils import FileNamingStandard

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Enhanced ML model trainer with standardized naming"""

    def __init__(self):
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        self.evaluator = ModelEvaluator()
        
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types and Path objects to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: MLModelTrainer.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [MLModelTrainer.make_json_serializable(item) for item in obj]
        elif isinstance(obj, (Path, type(Path()))):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    def create_models(self):
        """Create ML model instances with configurations"""
        return {
            "LogisticRegression": LogisticRegression(**ML_CONFIG["logistic_regression"]),
            "RandomForest": RandomForestClassifier(**ML_CONFIG["random_forest"]),
            "XGBoost": xgb.XGBClassifier(**ML_CONFIG["xgboost"])
        }

    def evaluate_model(self, model, X_test, y_test, model_name, n_categories, feature_type, class_labels):
        """Evaluate model and generate visualizations with standardized naming"""
        try:
            start = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            inference_time = time.time() - start

            # Calculate basic metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )

            # Calculate top-K accuracies using common evaluator
            top1 = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=1)
            top3 = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=3)
            top5 = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=5)

            # Calculate macro and micro averages
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)
            
            micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='micro', zero_division=0
            )

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Use common evaluator methods with standardized naming
            cm_path = self.evaluator.generate_confusion_heatmap(
                cm, class_labels, model_name, n_categories, feature_type, "ml"
            )
            report_path = self.evaluator.generate_classification_report_csv(
                y_test, y_pred, class_labels, model_name, n_categories, feature_type, "ml"
            )

            # Compile results
            results = {
                "model_name": model_name,
                "feature_type": feature_type,
                "n_categories": int(n_categories),
                "accuracy": float(acc),
                "top1_accuracy": float(top1),
                "top3_accuracy": float(top3),
                "top5_accuracy": float(top5),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "micro_precision": float(micro_precision),
                "micro_recall": float(micro_recall),
                "micro_f1": float(micro_f1),
                "confusion_matrix_plot": cm_path,
                "classification_report_path": str(report_path),
                "inference_time": float(inference_time)
            }

            return results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise

    def train_model_on_category(self, n_categories, feature_type="tfidf"):
        """Train and evaluate ML models for given category size"""
        logger.info(f"Training models for top_{n_categories}_categories ({feature_type})")

        # Load datasets
        splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
        train_df = pd.read_csv(splits_dir / "train.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")

        # Load class labels using common evaluator
        class_labels = self.evaluator.load_class_labels(n_categories)

        # Load features
        if feature_type == "tfidf":
            self.feature_extractor.load_tfidf_vectorizer(n_categories)
            X_train = self.feature_extractor.tfidf_vectorizer.transform(train_df["cleaned_text"])
            X_test = self.feature_extractor.tfidf_vectorizer.transform(test_df["cleaned_text"])
        else:
            X_train = self.feature_extractor.load_sbert_features(n_categories, "train")
            X_test = self.feature_extractor.load_sbert_features(n_categories, "test")

        y_train, y_test = train_df["encoded_label"], test_df["encoded_label"]

        results = {}
        for model_name, model in self.create_models().items():
            print(f"\nTraining {model_name} with {feature_type.upper()} features on top_{n_categories}_categories...")
            
            try:
                start = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start

                # Evaluate model
                res = self.evaluate_model(model, X_test, y_test, model_name, n_categories, feature_type, class_labels)
                res["training_time"] = float(train_time)
                res["n_categories"] = int(n_categories)
                res["feature_type"] = feature_type
                results[model_name] = res

                # Print metrics using common evaluator
                self.evaluator.print_model_metrics(res, model_name, n_categories, feature_type, train_time, "ML")

                # Save performance data using common evaluator
                self.evaluator.save_model_performance_data(res, model_name, n_categories, feature_type, "ml")

                # Save model with standardized filename
                model_dir = SAVED_MODELS_CONFIG["ml_models_path"] / f"top_{n_categories}_categories"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_filename = FileNamingStandard.generate_model_filename(
                    model_name, feature_type, n_categories, 'pkl'
                )
                model_path = model_dir / model_filename
                joblib.dump(model, model_path)
                
                logger.info(f"Model saved: {model_path}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        return results

    def save_results_for_overall_analysis(self, all_results):
        """Save results in the format expected by OverallPerformanceAnalyzer"""
        try:
            comparisons_path = RESULTS_CONFIG['ml_comparisons_path']
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            # Transform results into the format expected by OverallPerformanceAnalyzer
            formatted_results = {}
            
            for feature_type, feature_results in all_results.items():
                for n_categories, models_results in feature_results.items():
                    if n_categories not in formatted_results:
                        formatted_results[n_categories] = {}
                    
                    for model_name, result in models_results.items():
                        # Use consistent naming pattern
                        clean_model_name = FileNamingStandard.standardize_model_name(model_name)
                        result_key = f"{clean_model_name}_{feature_type}"
                        formatted_results[n_categories][result_key] = result
            
            # Save as pickle file
            pickle_file = comparisons_path / "ml_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(formatted_results, f)
            
            logger.info(f"ML results saved for overall analysis: {pickle_file}")
            
            # Also save JSON for debugging
            json_file = comparisons_path / "ml_final_results.json"
            with open(json_file, 'w') as f:
                json_safe_results = self.make_json_serializable(formatted_results)
                json.dump(json_safe_results, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving ML results for overall analysis: {e}")

    def train_all_categories(self, feature_types=["tfidf", "sbert"]):
        """Train across all category sizes with enhanced result handling"""
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING ML MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Category sizes: {CATEGORY_SIZES}")
        print(f"Feature types: {feature_types}")
        print(f"Models: {ML_CONFIG['models']}")
        print(f"{'='*80}")
        
        for feature in feature_types:
            print(f"\n{'-'*60}")
            print(f"TRAINING WITH {feature.upper()} FEATURES")
            print(f"{'-'*60}")
            
            all_results[feature] = {}
            for n in CATEGORY_SIZES:
                try:
                    print(f"\n>>> Processing top_{n}_categories with {feature.upper()} features...")
                    all_results[feature][n] = self.train_model_on_category(n, feature)
                    logger.info(f"Successfully completed training for top_{n}_categories with {feature}")
                except Exception as e:
                    logger.error(f"Error training {feature} on top_{n}_categories: {e}")
                    print(f"ERROR: Failed to train {feature} on top_{n}_categories: {e}")
        
        print(f"\n{'='*80}")
        print(f"ML MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        
        # Save results for overall analysis
        self.save_results_for_overall_analysis(all_results)
        
        # Print summary
        print(f"\nML Final Results Summary:")
        print(f"  Pickle file: {RESULTS_CONFIG['ml_comparisons_path']}/ml_final_results.pkl")
        print(f"  Categories processed: {list(all_results[list(all_results.keys())[0]].keys()) if all_results else []}")
        
        # Generate visualizations
        print(f"\n{'='*80}")
        print(f"GENERATING ML VISUALIZATIONS")
        print(f"{'='*80}")
        try:
            print("Generating comparison plots...")
            self.plot_ml_results_only()
            print("Generating radar plots...")
            self.generate_ml_radar_plots_only()
            print("All ML visualizations completed successfully!")
        except Exception as e:
            logger.error(f"Error generating ML visualizations: {e}")
            print(f"Warning: Some visualizations may not have been generated due to errors.")
        
        return all_results

    def plot_ml_results_only(self):
        """Generate ML comparison plots"""
        results_file_path = RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl"
        charts_dir = RESULTS_CONFIG["ml_comparisons_path"] / "charts"
        self.evaluator.plot_results_comparison(results_file_path, charts_dir, "ml")
    
    def generate_ml_radar_plots_only(self, show_plots=False):
        """Generate ML radar plots"""
        self.evaluator.generate_radar_plots("ml", show_plots)


def main():
    trainer = MLModelTrainer()
    results = trainer.train_all_categories()
    
    # Save final results
    out_file = SAVED_MODELS_CONFIG["ml_models_path"] / "ml_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json_safe_results = trainer.make_json_serializable(results)
        json.dump(json_safe_results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
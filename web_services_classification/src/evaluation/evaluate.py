"""
Enhanced ModelEvaluator with standardized naming and proper model type handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging
from pathlib import Path
from sklearn.metrics import classification_report
from math import pi

# Import standardized naming
from src.utils.utils import FileNamingStandard
from src.config import RESULTS_CONFIG, CATEGORY_SIZES

# Setup logging
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Enhanced evaluator with standardized naming across all model types"""
    
    def __init__(self):
        self.final_results = {}
        
    def calculate_top_k_accuracy(self, y_true, y_proba, k=1):
        """Calculate top-k accuracy"""
        try:
            # Handle different input formats
            if hasattr(y_true, 'shape') and len(y_true.shape) > 1:
                # One-hot encoded
                if y_true.shape[1] > 1:
                    y_true_labels = np.argmax(y_true, axis=1)
                else:
                    y_true_labels = y_true.flatten()
            else:
                # Already label indices
                y_true_labels = y_true
            
            # Get top-k predictions
            if k == 1:
                top_k_preds = np.argmax(y_proba, axis=1)
                return np.mean(top_k_preds == y_true_labels)
            else:
                top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
                return np.mean([label in pred for label, pred in zip(y_true_labels, top_k_preds)])
                
        except Exception as e:
            logger.error(f"Error calculating top-{k} accuracy: {e}")
            return 0.0
    
    def load_class_labels(self, n_categories):
        """Load class labels for a given category size"""
        try:
            from src.config import PREPROCESSING_CONFIG
            splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
            
            # Try to load from train.csv
            train_df = pd.read_csv(splits_dir / 'train.csv')
            
            if 'Service Classification' in train_df.columns and 'encoded_label' in train_df.columns:
                # Create mapping from encoded labels to original labels
                label_mapping = train_df.groupby('encoded_label')['Service Classification'].first().sort_index()
                return label_mapping.tolist()
            else:
                # Fallback: create generic labels
                return [f"Category_{i}" for i in range(n_categories)]
                
        except Exception as e:
            logger.error(f"Error loading class labels for {n_categories} categories: {e}")
            return [f"Category_{i}" for i in range(n_categories)]
    
    def _get_results_path(self, model_type, n_categories):
        """Get the correct results path based on model type"""
        model_type_mapping = {
            'ml': 'ml',
            'dl': 'dl', 
            'bert': 'bert',
            'roberta': 'bert',
            'deepseek': 'deepseek',
            'fusion': 'fusion'
        }
        
        dir_model_type = model_type_mapping.get(model_type, model_type)
        
        if dir_model_type == 'bert':
            return RESULTS_CONFIG['bert_category_paths'][n_categories]
        elif dir_model_type == 'deepseek':
            return RESULTS_CONFIG['deepseek_category_paths'][n_categories]
        elif dir_model_type == 'ml':
            return RESULTS_CONFIG['ml_category_paths'][n_categories]
        elif dir_model_type == 'dl':
            return RESULTS_CONFIG['dl_category_paths'][n_categories]
        elif dir_model_type == 'fusion':
            return RESULTS_CONFIG['fusion_category_paths'][n_categories]
        else:
            fallback_path = Path(f"results/{dir_model_type}/top_{n_categories}_categories")
            fallback_path.mkdir(parents=True, exist_ok=True)
            return fallback_path
    
    def generate_confusion_heatmap(self, cm, class_labels, model_name, n_categories, feature_type, model_type):
        """Generate confusion matrix heatmap with standardized naming"""
        try:
            results_path = self._get_results_path(model_type, n_categories)
            results_path.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels
            )
            plt.title(f'{model_name} - Confusion Matrix\n{n_categories} Categories ({feature_type.upper()})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            filename = FileNamingStandard.generate_confusion_matrix_filename(
                model_name, feature_type, n_categories
            )
            plot_file = results_path / filename
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix for {model_name}: {e}")
            return None
    
    def _calculate_per_category_metrics(self, y_true, y_pred, class_labels):
        """Helper function to calculate per-category accuracy (same as recall)"""
        per_cat_metrics = {}
        
        for idx, label in enumerate(class_labels):
            mask = (y_true == idx)
            if mask.sum() > 0:
                per_cat_accuracy = (y_pred[mask] == idx).sum() / mask.sum()
                per_cat_metrics[label] = per_cat_accuracy
            else:
                per_cat_metrics[label] = 0.0
        
        return per_cat_metrics

    def generate_classification_report_csv(self, y_true, y_pred, class_labels, model_name, n_categories, feature_type, model_type):
        """Generate classification report CSV with standardized naming"""
        try:
            results_path = self._get_results_path(model_type, n_categories)
            results_path.mkdir(parents=True, exist_ok=True)
            
            report = classification_report(
                y_true, 
                y_pred, 
                target_names=class_labels,
                output_dict=True,
                zero_division=0
            )
            
            report_df = pd.DataFrame(report).transpose()
            per_cat_acc = self._calculate_per_category_metrics(y_true, y_pred, class_labels)
            report_df['accuracy'] = report_df.index.map(per_cat_acc)
            
            report_df.reset_index(inplace=True)
            report_df.rename(columns={'index': 'category_name'}, inplace=True)
            
            filename = FileNamingStandard.generate_classification_report_filename(
                model_name, feature_type, n_categories
            )
            report_file = results_path / filename
            report_df.to_csv(report_file, index=False)
            
            logger.info(f"Classification report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating classification report for {model_name}: {e}")
            return None
    
    def print_model_metrics(self, results, model_name, n_categories, feature_type, training_time, model_category):
        """Print standardized model metrics"""
        print(f"\n{'='*60}")
        print(f"{model_category} MODEL EVALUATION: {model_name}")
        print(f"{'='*60}")
        print(f"Categories: {n_categories} | Feature Type: {feature_type.upper()}")
        print(f"Training Time: {training_time:.2f}s | Inference Time: {results.get('inference_time', 0):.4f}s")
        print(f"{'-'*60}")
        print(f"Top-1 Accuracy: {results.get('top1_accuracy', results.get('accuracy', 0)):.4f}")
        print(f"Top-3 Accuracy: {results.get('top3_accuracy', 0):.4f}")
        print(f"Top-5 Accuracy: {results.get('top5_accuracy', 0):.4f}")
        print(f"Macro F1:      {results.get('macro_f1', 0):.4f}")
        print(f"Micro F1:      {results.get('micro_f1', 0):.4f}")
        print(f"{'='*60}")
    
    def save_model_performance_data(self, results, model_name, n_categories, feature_type, model_type):
        """Save model performance data to final results"""
        try:
            if n_categories not in self.final_results:
                self.final_results[n_categories] = {}
            
            clean_model_name = FileNamingStandard.standardize_model_name(model_name)
            result_key = f"{clean_model_name}_{feature_type}"
            
            self.final_results[n_categories][result_key] = results
            self._save_final_results_pickle(model_type)
            
            logger.info(f"Performance data saved for {model_name} ({feature_type})")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _save_final_results_pickle(self, model_type):
        """Save final results as pickle file for overall analysis"""
        try:
            if model_type == 'ml':
                comparisons_path = RESULTS_CONFIG['ml_comparisons_path']
            elif model_type == 'dl':
                comparisons_path = RESULTS_CONFIG['dl_comparisons_path']
            elif model_type in ['bert', 'roberta']:
                comparisons_path = RESULTS_CONFIG['bert_comparisons_path']
            elif model_type == 'deepseek':
                comparisons_path = RESULTS_CONFIG['deepseek_comparisons_path']
            elif model_type == 'fusion':
                comparisons_path = RESULTS_CONFIG['fusion_comparisons_path']
            else:
                return
            
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            pickle_file = comparisons_path / f"{model_type}_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.final_results, f)
                
            logger.info(f"Final results saved: {pickle_file}")
            
        except Exception as e:
            logger.error(f"Error saving final results pickle: {e}")
    
    def plot_results_comparison(self, results_file_path, charts_dir, model_type):
        """Generate comprehensive comparison plots synchronized with save format"""
        try:
            if not results_file_path.exists():
                print(f"No {model_type.upper()} results file found at: {results_file_path}")
                return
            
            charts_dir = Path(charts_dir)
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_file_path, "rb") as f:
                final_results = pickle.load(f)
            
            print(f"\n{'='*80}")
            print(f"GENERATING {model_type.upper()} COMPARISON PLOTS")
            print(f"{'='*80}")
            print(f"Results loaded from: {results_file_path}")
            print(f"Charts will be saved to: {charts_dir}")
            print(f"{'='*80}\n")
            
            model_metrics = {}
            
            for n_categories, model_results_dict in final_results.items():
                for model_key, entry in model_results_dict.items():
                    model_name = entry.get('model_name', 'Unknown')
                    feature_type = entry.get('feature_type', 'unknown')
                    
                    if model_name not in model_metrics:
                        model_metrics[model_name] = {}
                    
                    if feature_type not in model_metrics[model_name]:
                        model_metrics[model_name][feature_type] = {
                            'n': [], 
                            'accuracy': [], 
                            'precision': [], 
                            'recall': [], 
                            'f1_score': [],
                            'top1_accuracy': [], 
                            'top3_accuracy': [], 
                            'top5_accuracy': [],
                            'training_time': [], 
                            'inference_time': []
                        }
                    
                    metrics = model_metrics[model_name][feature_type]
                    
                    n_cat = entry.get('n_categories', int(n_categories) if isinstance(n_categories, str) else n_categories)
                    metrics['n'].append(n_cat)
                    metrics['accuracy'].append(entry.get('accuracy', 0))
                    metrics['precision'].append(entry.get('macro_precision', 0))
                    metrics['recall'].append(entry.get('macro_recall', 0))
                    metrics['f1_score'].append(entry.get('macro_f1', 0))
                    metrics['top1_accuracy'].append(entry.get('top1_accuracy', entry.get('accuracy', 0)))
                    metrics['top3_accuracy'].append(entry.get('top3_accuracy', 0))
                    metrics['top5_accuracy'].append(entry.get('top5_accuracy', 0))
                    metrics['training_time'].append(entry.get('training_time', 0))
                    metrics['inference_time'].append(entry.get('inference_time', 0))
            
            # Sort metrics by n_categories
            for model in model_metrics:
                for feature in model_metrics[model]:
                    indices = sorted(range(len(model_metrics[model][feature]['n'])), 
                                   key=lambda i: model_metrics[model][feature]['n'][i])
                    
                    for metric_name in model_metrics[model][feature]:
                        values = model_metrics[model][feature][metric_name]
                        model_metrics[model][feature][metric_name] = [values[i] for i in indices]
            
            print(f"✓ Parsed {len(model_metrics)} models with results")
            print(f"  Models: {list(model_metrics.keys())}")
            print(f"  Feature types per model: {[list(model_metrics[m].keys()) for m in model_metrics]}\n")
            
            print("Generating visualizations...")
            
            print("  → Generating line plots...")
            self._generate_line_plots(model_metrics, charts_dir, model_type)
            
            print("  → Generating bar plots...")
            self._generate_bar_plots(final_results, charts_dir, model_type)

            print("  → Generating stacked bar plots...")
            self._generate_stacked_bar_plots(final_results, charts_dir, model_type)

            print("  → Generating Grouped bar plots...")
            self._generate_grouped_bar_plots(final_results, charts_dir, model_type)
            
            print("  → Generating summary statistics...")
            self._generate_summary_statistics(final_results, charts_dir, model_type)
            
            print(f"\n{'='*80}")
            print(f"✓ ALL {model_type.upper()} COMPARISON PLOTS GENERATED SUCCESSFULLY")
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Error generating {model_type} plots: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_line_plots(self, model_metrics, charts_dir, model_type):
        """Generate line plots for performance metrics with improved naming"""
        def plot_metric(metric_name, ylabel=None):
            plt.figure(figsize=(12, 6))
            
            for model, features in model_metrics.items():
                for feature_type, data in features.items():
                    label = f"{model} ({feature_type.upper()})"
                    plt.plot(data['n'], data[metric_name], marker='o', label=label, linewidth=2)
            
            plt.title(f'{ylabel or metric_name.replace("_", " ").title()} vs Number of Web Service Categories ({model_type.upper()} Models)')
            plt.xlabel('Number of Web Service Categories')
            plt.ylabel(ylabel or metric_name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            max_category = max([max(data['n']) for model in model_metrics.values() for data in model.values()])
            plot_path = charts_dir / f"{model_type.lower()}_line_{metric_name}_top_{max_category}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ {model_type.upper()} line plot saved: {plot_path}")
            plt.close()

        print(f"\nGenerating {model_type.upper()} line plots...")
        metrics_config = {
            'accuracy': 'Accuracy',
            'precision': 'Precision (Macro)',
            'recall': 'Recall (Macro)',
            'f1_score': 'F1-Score (Macro)',
            'top1_accuracy': 'Top-1 Accuracy',
            'top3_accuracy': 'Top-3 Accuracy', 
            'top5_accuracy': 'Top-5 Accuracy'
        }
        
        if model_type.lower() == "dl":
            metrics_config.update({
                'training_time': 'Training Time (seconds)',
                'inference_time': 'Inference Time (seconds)'
            })
        
        for metric, ylabel in metrics_config.items():
            plot_metric(metric, ylabel)
        
        # Combined top-K accuracy plot
        print(f"  → Generating combined top-K accuracy plot...")
        plt.figure(figsize=(14, 8))
        
        for model, features in model_metrics.items():
            for feature_type, data in features.items():
                label_base = f"{model} ({feature_type.upper()})"
                plt.plot(data['n'], data['top1_accuracy'], marker='o', label=f"{label_base} - Top-1", linewidth=2)
                plt.plot(data['n'], data['top3_accuracy'], marker='s', label=f"{label_base} - Top-3", linewidth=2, linestyle='--')
                plt.plot(data['n'], data['top5_accuracy'], marker='^', label=f"{label_base} - Top-5", linewidth=2, linestyle=':')
        
        plt.title(f'{model_type.upper()} Models: Top-K Accuracy Comparison')
        plt.xlabel('Number of Web Service Categories')
        plt.ylabel('Top-K Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        max_category = max([max(data['n']) for model in model_metrics.values() for data in model.values()])
        plot_path = charts_dir / f"{model_type.lower()}_line_topk_combined_top_{max_category}_categories.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Combined top-K plot saved: {plot_path}")
        plt.close()
    
    def _generate_bar_plots(self, final_results, charts_dir, model_type):
        """Generate bar plots for each category size with improved naming"""
        print(f"\nGenerating enhanced {model_type.upper()} bar plots for each category size...")
        
        for n_categories, model_results_dict in final_results.items():
            if not model_results_dict:
                print(f"Skipping n={n_categories} (no {model_type.upper()} results found)")
                continue
            
            plot_data = {
                'model_label': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'top1_accuracy': [],
                'top3_accuracy': [],
                'top5_accuracy': []
            }
            
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                label = f"{model_name}\n({feature_type.upper()})"
                
                plot_data['model_label'].append(label)
                plot_data['accuracy'].append(result.get('accuracy', 0))
                plot_data['precision'].append(result.get('macro_precision', 0))
                plot_data['recall'].append(result.get('macro_recall', 0))
                plot_data['f1_score'].append(result.get('macro_f1', 0))
                plot_data['top1_accuracy'].append(result.get('top1_accuracy', 0))
                plot_data['top3_accuracy'].append(result.get('top3_accuracy', 0))
                plot_data['top5_accuracy'].append(result.get('top5_accuracy', 0))
            
            df_combined = pd.DataFrame(plot_data)
            df_combined.set_index('model_label', inplace=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Plot 1: Standard metrics
            standard_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            df_combined[standard_metrics].plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title(f'{model_type.upper()} Standard Performance Metrics - Top {n_categories} Categories', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Model (Feature Type)', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, 1.0)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.legend(title='Metric', labels=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
            
            # Plot 2: Top-K accuracy metrics
            topk_metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            df_combined[topk_metrics].plot(kind='bar', ax=ax2, width=0.8)
            ax2.set_title(f'{model_type.upper()} Top-K Accuracy Metrics - Top {n_categories} Categories', 
                         fontsize=14, fontweight='bold')
            ax2.set_ylabel('Top-K Accuracy', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Model (Feature Type)', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1.0)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.legend(title='Top-K Metric', labels=['Top-1', 'Top-3', 'Top-5'])
            
            plt.tight_layout()
            
            plot_path = charts_dir / f"{model_type.lower()}_bar_grouped_top_{n_categories}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Enhanced bar plot saved: {plot_path}")
            plt.close()
            
            self._generate_individual_metric_bars(plot_data, n_categories, charts_dir, model_type)

    def _generate_individual_metric_bars(self, plot_data, n_categories, charts_dir, model_type):
        """Generate individual bar charts with improved naming"""
        print(f"  → Generating individual metric bar charts for top_{n_categories}...")
        
        metrics_config = {
            'accuracy': ('Accuracy', '#3498db'),
            'precision': ('Precision (Macro)', '#e74c3c'),
            'recall': ('Recall (Macro)', '#2ecc71'),
            'f1_score': ('F1-Score (Macro)', '#f39c12'),
            'top1_accuracy': ('Top-1 Accuracy', '#9b59b6'),
            'top3_accuracy': ('Top-3 Accuracy', '#1abc9c'),
            'top5_accuracy': ('Top-5 Accuracy', '#34495e')
        }
        
        for metric_key, (metric_title, color) in metrics_config.items():
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x_pos = np.arange(len(plot_data['model_label']))
            bars = ax.bar(x_pos, plot_data[metric_key], 
                         color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
            
            max_idx = plot_data[metric_key].index(max(plot_data[metric_key]))
            bars[max_idx].set_edgecolor('gold')
            bars[max_idx].set_linewidth(3)
            
            ax.set_xlabel('Model (Feature Type)', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_title, fontsize=13, fontweight='bold')
            ax.set_title(f'{model_type.upper()} Models: {metric_title} - Top {n_categories} Categories',
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_data['model_label'], fontsize=10, rotation=15, ha='right')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            mean_val = np.mean(plot_data[metric_key])
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.4f}', alpha=0.7)
            ax.legend(loc='upper right', fontsize=10)
            
            plt.tight_layout()
            
            plot_path = charts_dir / f"{model_type.lower()}_bar_individual_{metric_key}_top_{n_categories}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Individual metric bar charts generated for top_{n_categories}")
    
    def _generate_summary_statistics(self, final_results, charts_dir, model_type):
        """Generate summary statistics table"""
        print(f"\nGenerating {model_type.upper()} summary statistics...")
        
        summary_data = []
        for n_categories, model_results_dict in final_results.items():
            for model_key, entry in model_results_dict.items():
                summary_entry = {
                    'Categories': entry.get('n_categories', n_categories),
                    'Model': entry.get('model_name', 'Unknown'),
                    'Feature': entry.get('feature_type', 'unknown'),
                    'Accuracy': entry.get('accuracy', 0),
                    'F1-Score': entry.get('macro_f1', 0),
                    'Top-1': entry.get('top1_accuracy', entry.get('accuracy', 0)),
                    'Top-3': entry.get('top3_accuracy', 0),
                    'Top-5': entry.get('top5_accuracy', 0)
                }
                
                if model_type.lower() == "dl":
                    summary_entry.update({
                        'Training Time': entry.get('training_time', 0),
                        'Inference Time': entry.get('inference_time', 0)
                    })
                
                summary_data.append(summary_entry)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.round(4)
            
            summary_path = charts_dir / f"{model_type.lower()}_summary_statistics.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"✓ {model_type.upper()} summary table saved: {summary_path}")
            
            print(f"\nTop performing {model_type.upper()} models by metric:")
            for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
                if metric in summary_df.columns and len(summary_df) > 0:
                    best = summary_df.loc[summary_df[metric].idxmax()]
                    print(f"  {metric}: {best['Model']} ({best['Feature']}) on {best['Categories']} categories = {best[metric]:.4f}")
            
            if len(summary_df) > 0 and 'Top-1' in summary_df.columns:
                best_overall = summary_df.loc[summary_df['Top-1'].idxmax()]
                print(f"\nBest Overall {model_type.upper()} Model:")
                print(f"  {best_overall['Model']} ({best_overall['Feature']}) on {best_overall['Categories']} categories")
                print(f"  Top-1 Accuracy: {best_overall['Top-1']:.4f}")
                print(f"  F1-Score: {best_overall['F1-Score']:.4f}")
                if model_type.lower() == "dl" and 'Training Time' in summary_df.columns:
                    print(f"  Training Time: {best_overall['Training Time']:.2f}s")
    
    def generate_radar_plots(self, model_type, show_plots=False):
        """Generate radar plots for model performance across categories"""
        NAMING_PATTERNS = {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForest",        
            "xgboost": "XGBoost",
            "bilstm": "BiLSTM",
            "deepseek_roberta_fusion_concat": "DeepSeek_RoBERTa_Fusion_Concat",
            "deepseek_roberta_fusion_average": "DeepSeek_RoBERTa_Fusion_Average",
            "deepseek_roberta_fusion_weighted": "DeepSeek_RoBERTa_Fusion_Weighted",
            "deepseek_roberta_fusion_gating": "DeepSeek_RoBERTa_Fusion_Gating"
        }
        # Get model configuration based on type
        if model_type.lower() == "ml":
            try:
                from src.config import ML_CONFIG
                models = ML_CONFIG.get("models", ["LogisticRegression", "RandomForest", "XGBoost"])
                results_paths = RESULTS_CONFIG["ml_category_paths"]
                save_dir = RESULTS_CONFIG["ml_comparisons_path"]
                title_prefix = "ML Models"
                feature_types = ["tfidf", "sbert"]
            except:
                models = ["LogisticRegression", "RandomForest", "XGBoost"]
                results_paths = RESULTS_CONFIG["ml_category_paths"]
                save_dir = RESULTS_CONFIG["ml_comparisons_path"]
                title_prefix = "ML Models"
                feature_types = ["tfidf", "sbert"]
        elif model_type.lower() == "dl":
            try:
                from src.config import DL_CONFIG
                models = DL_CONFIG.get("models", ["BiLSTM"]) 
                results_paths = RESULTS_CONFIG["dl_category_paths"]
                save_dir = RESULTS_CONFIG["dl_comparisons_path"]
                title_prefix = "DL Models"
                feature_types = DL_CONFIG.get("feature_types", ["tfidf", "sbert"])
            except:
                models = ["BiLSTM"]
                results_paths = RESULTS_CONFIG["dl_category_paths"]
                save_dir = RESULTS_CONFIG["dl_comparisons_path"]
                title_prefix = "DL Models"
                feature_types = ["tfidf", "sbert"]
        elif model_type.lower() == "bert":
            models = ["RoBERTa_Base", "RoBERTa_Large"]
            results_paths = RESULTS_CONFIG["bert_category_paths"]
            save_dir = RESULTS_CONFIG["bert_comparisons_path"]
            title_prefix = "BERT Models"
            feature_types = ["raw_text"]
        elif model_type.lower() == "deepseek":
            models = ["DeepSeek_7B_Base"]
            results_paths = RESULTS_CONFIG["deepseek_category_paths"]
            save_dir = RESULTS_CONFIG["deepseek_comparisons_path"]
            title_prefix = "DeepSeek Models"
            feature_types = ["raw_text"]
        elif model_type.lower() == "fusion":
            # Fusion uses single base model name with different feature types (fusion strategies)
            models = ["DeepSeek_RoBERTa_Fusion"]
            results_paths = RESULTS_CONFIG["fusion_category_paths"]
            save_dir = RESULTS_CONFIG["fusion_comparisons_path"]
            title_prefix = "Fusion Models"
            # Feature types are the fusion strategies: Concat, Average, Weighted, Gating
            feature_types = ["Concat", "Average", "Weighted", "Gating"]
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        metrics = ["precision", "recall", "f1-score", "accuracy"]
        
        print(f"\nGenerating {model_type.upper()} radar plots...")
        
        for num_cat in CATEGORY_SIZES:
            print(f"  Processing radar plots for {num_cat} categories...")
            
            data = self._load_radar_data(models, feature_types, results_paths[num_cat], num_cat, NAMING_PATTERNS)
            
            if not data:
                print(f"  No data found for {num_cat} categories, skipping radar plots")
                continue
            
            for metric in metrics:
                self._plot_radar_chart(data, metric, num_cat, title_prefix, save_dir, model_type, show_plots)
        
        print(f"Completed {model_type.upper()} radar plot generation")
    
    def _load_radar_data(self, models, feature_types, category_path, num_cat, naming_patterns):
        """Load classification report data for radar plots"""
        data = {}
        
        for model in models:
            for feature in feature_types:
                model_lower = model.lower()
                model_display_name = naming_patterns.get(model_lower, model)
                filename = FileNamingStandard.generate_classification_report_filename(
                    model_display_name, feature, num_cat
                )
                file_path = category_path / filename
                print(f"  DEBUG: Looking for file: {file_path}")
                if not file_path.exists():
                    logger.warning(f"Missing radar data file: {file_path}")
                    continue

                try:
                    df = pd.read_csv(file_path)
                    
                    if 'category_name' in df.columns:
                        category_rows = df[~df['category_name'].isin(['macro avg', 'micro avg', 'weighted avg'])]
                        category_rows = category_rows[~category_rows['category_name'].isna()]
                        category_rows = category_rows.head(num_cat).set_index("category_name")
                        data[f"{model}_{feature}"] = category_rows
                    else:
                        logger.warning(f"No 'category_name' column found in {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error reading radar data from {file_path}: {e}")

        return data
    
    def _plot_radar_chart(self, data, metric, num_cat, title_prefix, save_dir, model_type, show_plots):
        """Generate and save radar chart with improved naming"""
        if not data:
            return

        first_key = list(data.keys())[0]
        if first_key not in data or data[first_key].empty:
            return
            
        labels = data[first_key].index.tolist()
        num_labels = len(labels)
        
        if num_labels == 0:
            logger.warning(f"No labels found for radar plot with {num_cat} categories")
            return
        
        angles = [n / float(num_labels) * 2 * pi for n in range(num_labels)]
        angles += angles[:1]

        figsize = (8, 8) if num_cat < 40 else (14, 14)
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        for i, (model_name, df) in enumerate(data.items()):
            if metric not in df.columns:
                logger.warning(f"Metric '{metric}' not found in data for {model_name}")
                continue
                
            metric_values = df[metric].fillna(0).tolist()
            
            if len(metric_values) != num_labels:
                logger.warning(f"Metric values length mismatch for {model_name}")
                continue
            
            values = metric_values + metric_values[:1]
            display_name = model_name.replace('_', ' ').title()
            
            ax.plot(angles, values, 'o-', linewidth=2, label=display_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        fontsize = 10 if num_cat < 20 else 8 if num_cat < 40 else 6
        display_labels = [lbl[:15] + "..." if len(lbl) > 18 else lbl for lbl in labels]
        ax.set_xticklabels(display_labels, fontsize=fontsize)
        
        ax.set_title(f"{title_prefix} - {metric.replace('-', ' ').title()} Performance\n(Top {num_cat} Categories)",
                     size=16 if num_cat < 40 else 14, weight="bold", pad=20)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()

        # Improved naming: model_type_radar_metricname_top_N_categories.png
        filename = f"{model_type.lower()}_radar_{metric.replace('-', '_')}_top_{num_cat}_categories.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Radar plot saved: {filepath}")

        if show_plots:
            plt.show()
        plt.close()

    def _generate_stacked_bar_plots(self, final_results, charts_dir, model_type):
        """Generate stacked bar plots for individual category sizes and overall comparison"""
        print(f"\nGenerating {model_type.upper()} stacked bar plots...")
        
        # 1. Top-K Accuracy Stacked Bars (Individual)
        self._generate_topk_stacked_bars_individual(final_results, charts_dir, model_type)
        
        # 2. Performance Metrics Stacked Bars (Individual)
        self._generate_metrics_stacked_bars_individual(final_results, charts_dir, model_type)
        
        # 3. Top-K Accuracy Stacked Bars (Overall)
        self._generate_topk_stacked_bars_overall(final_results, charts_dir, model_type)
        
        # 4. Performance Metrics Stacked Bars (Overall)
        self._generate_metrics_stacked_bars_overall(final_results, charts_dir, model_type)

    def _generate_topk_stacked_bars_individual(self, final_results, charts_dir, model_type):
        """Generate Top-K accuracy stacked bars for each category size"""
        print(f"  → Generating Top-K stacked bars for individual category sizes...")
        
        for n_categories, model_results_dict in final_results.items():
            if not model_results_dict:
                continue
            
            plot_data = {
                'model_label': [],
                'top1_accuracy': [],
                'top3_gain': [],
                'top5_gain': []
            }
            
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                label = f"{model_name}\n({feature_type.upper()})"
                
                top1 = result.get('top1_accuracy', result.get('accuracy', 0))
                top3 = result.get('top3_accuracy', 0)
                top5 = result.get('top5_accuracy', 0)
                
                plot_data['model_label'].append(label)
                plot_data['top1_accuracy'].append(top1)
                plot_data['top3_gain'].append(max(0, top3 - top1))
                plot_data['top5_gain'].append(max(0, top5 - top3))
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x_pos = np.arange(len(plot_data['model_label']))
            width = 0.6
            
            p1 = ax.bar(x_pos, plot_data['top1_accuracy'], width, 
                    label='Top-1 Accuracy', color='#3498db', edgecolor='black', linewidth=1)
            p2 = ax.bar(x_pos, plot_data['top3_gain'], width,
                    bottom=plot_data['top1_accuracy'],
                    label='Top-3 Gain', color='#2ecc71', edgecolor='black', linewidth=1)
            
            bottom_top5 = [plot_data['top1_accuracy'][i] + plot_data['top3_gain'][i] 
                        for i in range(len(plot_data['top1_accuracy']))]
            p3 = ax.bar(x_pos, plot_data['top5_gain'], width,
                    bottom=bottom_top5,
                    label='Top-5 Gain', color='#f39c12', edgecolor='black', linewidth=1)
            
            # Add value labels
            for i, (t1, t3g, t5g) in enumerate(zip(plot_data['top1_accuracy'], 
                                                    plot_data['top3_gain'], 
                                                    plot_data['top5_gain'])):
                if t1 > 0.05:
                    ax.text(i, t1/2, f'{t1:.3f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
                
                if t3g > 0.05:
                    ax.text(i, t1 + t3g/2, f'+{t3g:.3f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
                
                if t5g > 0.05:
                    ax.text(i, t1 + t3g + t5g/2, f'+{t5g:.3f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
                
                total = t1 + t3g + t5g
                ax.text(i, total + 0.02, f'{total:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black')
            
            ax.set_xlabel('Model (Feature Type)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Cumulative Accuracy', fontsize=13, fontweight='bold')
            ax.set_title(f'{model_type.upper()} Models: Stacked Top-K Accuracy - Top {n_categories} Categories',
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_data['model_label'], fontsize=10, rotation=15, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=11)
            
            plt.tight_layout()
            
            plot_path = charts_dir / f"{model_type.lower()}_stacked_bar_topk_top_{n_categories}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Top-K stacked bar saved: {plot_path}")
            plt.close()

    def _generate_metrics_stacked_bars_individual(self, final_results, charts_dir, model_type):
        """Generate performance metrics stacked bars for each category size"""
        print(f"  → Generating performance metrics stacked bars for individual category sizes...")
        
        for n_categories, model_results_dict in final_results.items():
            if not model_results_dict:
                continue
            
            plot_data = {
                'model_label': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
            
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                label = f"{model_name}\n({feature_type.upper()})"
                
                plot_data['model_label'].append(label)
                plot_data['accuracy'].append(result.get('accuracy', 0))
                plot_data['precision'].append(result.get('macro_precision', 0))
                plot_data['recall'].append(result.get('macro_recall', 0))
                plot_data['f1_score'].append(result.get('macro_f1', 0))
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x_pos = np.arange(len(plot_data['model_label']))
            width = 0.6
            
            # Create stacked bars
            p1 = ax.bar(x_pos, plot_data['accuracy'], width, 
                    label='Accuracy', color='#3498db', edgecolor='black', linewidth=1)
            p2 = ax.bar(x_pos, plot_data['precision'], width,
                    bottom=plot_data['accuracy'],
                    label='Precision', color='#e74c3c', edgecolor='black', linewidth=1)
            
            bottom_recall = [plot_data['accuracy'][i] + plot_data['precision'][i] 
                            for i in range(len(plot_data['accuracy']))]
            p3 = ax.bar(x_pos, plot_data['recall'], width,
                    bottom=bottom_recall,
                    label='Recall', color='#2ecc71', edgecolor='black', linewidth=1)
            
            bottom_f1 = [bottom_recall[i] + plot_data['recall'][i] 
                        for i in range(len(bottom_recall))]
            p4 = ax.bar(x_pos, plot_data['f1_score'], width,
                    bottom=bottom_f1,
                    label='F1-Score', color='#f39c12', edgecolor='black', linewidth=1)
            
            # Add value labels on each segment
            for i in range(len(plot_data['model_label'])):
                acc = plot_data['accuracy'][i]
                prec = plot_data['precision'][i]
                rec = plot_data['recall'][i]
                f1 = plot_data['f1_score'][i]
                
                if acc > 0.05:
                    ax.text(i, acc/2, f'{acc:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
                
                if prec > 0.05:
                    ax.text(i, acc + prec/2, f'{prec:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
                
                if rec > 0.05:
                    ax.text(i, acc + prec + rec/2, f'{rec:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
                
                if f1 > 0.05:
                    ax.text(i, acc + prec + rec + f1/2, f'{f1:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
                
                # Total on top
                total = acc + prec + rec + f1
                ax.text(i, total + 0.05, f'{total:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black')
            
            ax.set_xlabel('Model (Feature Type)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Cumulative Metrics Score', fontsize=13, fontweight='bold')
            ax.set_title(f'{model_type.upper()} Models: Stacked Performance Metrics - Top {n_categories} Categories',
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_data['model_label'], fontsize=10, rotation=15, ha='right')
            ax.set_ylim(0, max([plot_data['accuracy'][i] + plot_data['precision'][i] + 
                                plot_data['recall'][i] + plot_data['f1_score'][i] 
                                for i in range(len(plot_data['model_label']))]) + 0.2)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=11)
            
            plt.tight_layout()
            
            plot_path = charts_dir / f"{model_type.lower()}_stacked_bar_metrics_top_{n_categories}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Metrics stacked bar saved: {plot_path}")
            plt.close()

    def _generate_topk_stacked_bars_overall(self, final_results, charts_dir, model_type):
        """Generate overall Top-K accuracy stacked bar comparison across all category sizes"""
        print(f"  → Generating overall Top-K stacked bar comparison...")
        
        overall_data = {}
        
        for n_categories, model_results_dict in final_results.items():
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                combo_key = f"{model_name}_{feature_type}"
                
                if combo_key not in overall_data:
                    overall_data[combo_key] = {
                        'model_name': model_name,
                        'feature_type': feature_type,
                        'categories': [],
                        'top1': [],
                        'top3': [],
                        'top5': []
                    }
                
                overall_data[combo_key]['categories'].append(int(n_categories))
                overall_data[combo_key]['top1'].append(result.get('top1_accuracy', result.get('accuracy', 0)))
                overall_data[combo_key]['top3'].append(result.get('top3_accuracy', 0))
                overall_data[combo_key]['top5'].append(result.get('top5_accuracy', 0))
        
        if not overall_data:
            return
        
        # Sort by categories
        for key in overall_data:
            indices = sorted(range(len(overall_data[key]['categories'])),
                            key=lambda i: overall_data[key]['categories'][i])
            for metric in ['categories', 'top1', 'top3', 'top5']:
                overall_data[key][metric] = [overall_data[key][metric][i] for i in indices]
        
        # Create subplots
        n_combos = len(overall_data)
        n_cols = min(3, n_combos)
        n_rows = (n_combos + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_combos == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, (combo_key, data) in enumerate(overall_data.items()):
            ax = axes[idx]
            
            categories = data['categories']
            top1 = data['top1']
            top3 = data['top3']
            top5 = data['top5']
            
            top3_gain = [max(0, t3 - t1) for t1, t3 in zip(top1, top3)]
            top5_gain = [max(0, t5 - t3) for t3, t5 in zip(top3, top5)]
            
            x_pos = np.arange(len(categories))
            width = 0.6
            
            p1 = ax.bar(x_pos, top1, width, label='Top-1',
                    color='#3498db', edgecolor='black', linewidth=1)
            p2 = ax.bar(x_pos, top3_gain, width, bottom=top1,
                    label='Top-3 Gain', color='#2ecc71', edgecolor='black', linewidth=1)
            
            bottom_t5 = [top1[i] + top3_gain[i] for i in range(len(top1))]
            p3 = ax.bar(x_pos, top5_gain, width, bottom=bottom_t5,
                    label='Top-5 Gain', color='#f39c12', edgecolor='black', linewidth=1)
            
            # Add labels
            for i, (t1, t3g, t5g) in enumerate(zip(top1, top3_gain, top5_gain)):
                total = t1 + t3g + t5g
                ax.text(i, total + 0.02, f'{total:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
            
            model_label = f"{data['model_name']} ({data['feature_type'].upper()})"
            ax.set_title(model_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Number of Categories', fontsize=11, fontweight='bold')
            ax.set_ylabel('Cumulative Accuracy', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='lower left', fontsize=9)
        
        # Hide extra subplots
        for idx in range(len(overall_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{model_type.upper()} Models: Overall Stacked Top-K Accuracy Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = charts_dir / f"{model_type.lower()}_stacked_bar_overall_topk_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Overall Top-K stacked bar saved: {plot_path}")
        plt.close()

    def _generate_metrics_stacked_bars_overall(self, final_results, charts_dir, model_type):
        """Generate overall performance metrics stacked bar comparison across all category sizes"""
        print(f"  → Generating overall performance metrics stacked bar comparison...")
        
        overall_data = {}
        
        for n_categories, model_results_dict in final_results.items():
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                combo_key = f"{model_name}_{feature_type}"
                
                if combo_key not in overall_data:
                    overall_data[combo_key] = {
                        'model_name': model_name,
                        'feature_type': feature_type,
                        'categories': [],
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': []
                    }
                
                overall_data[combo_key]['categories'].append(int(n_categories))
                overall_data[combo_key]['accuracy'].append(result.get('accuracy', 0))
                overall_data[combo_key]['precision'].append(result.get('macro_precision', 0))
                overall_data[combo_key]['recall'].append(result.get('macro_recall', 0))
                overall_data[combo_key]['f1_score'].append(result.get('macro_f1', 0))
        
        if not overall_data:
            return
        
        # Sort by categories
        for key in overall_data:
            indices = sorted(range(len(overall_data[key]['categories'])),
                            key=lambda i: overall_data[key]['categories'][i])
            for metric in ['categories', 'accuracy', 'precision', 'recall', 'f1_score']:
                overall_data[key][metric] = [overall_data[key][metric][i] for i in indices]
        
        # Create subplots
        n_combos = len(overall_data)
        n_cols = min(3, n_combos)
        n_rows = (n_combos + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_combos == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, (combo_key, data) in enumerate(overall_data.items()):
            ax = axes[idx]
            
            categories = data['categories']
            accuracy = data['accuracy']
            precision = data['precision']
            recall = data['recall']
            f1_score = data['f1_score']
            
            x_pos = np.arange(len(categories))
            width = 0.6
            
            p1 = ax.bar(x_pos, accuracy, width, label='Accuracy',
                    color='#3498db', edgecolor='black', linewidth=1)
            p2 = ax.bar(x_pos, precision, width, bottom=accuracy,
                    label='Precision', color='#e74c3c', edgecolor='black', linewidth=1)
            
            bottom_rec = [accuracy[i] + precision[i] for i in range(len(accuracy))]
            p3 = ax.bar(x_pos, recall, width, bottom=bottom_rec,
                    label='Recall', color='#2ecc71', edgecolor='black', linewidth=1)
            
            bottom_f1 = [bottom_rec[i] + recall[i] for i in range(len(bottom_rec))]
            p4 = ax.bar(x_pos, f1_score, width, bottom=bottom_f1,
                    label='F1-Score', color='#f39c12', edgecolor='black', linewidth=1)
            
            # Add total labels
            for i in range(len(categories)):
                total = accuracy[i] + precision[i] + recall[i] + f1_score[i]
                ax.text(i, total + 0.05, f'{total:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
            
            model_label = f"{data['model_name']} ({data['feature_type'].upper()})"
            ax.set_title(model_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Number of Categories', fontsize=11, fontweight='bold')
            ax.set_ylabel('Cumulative Metrics', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, fontsize=10)
            max_total = max([accuracy[i] + precision[i] + recall[i] + f1_score[i] 
                            for i in range(len(categories))])
            ax.set_ylim(0, max_total + 0.3)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=9)
        
        # Hide extra subplots
        for idx in range(len(overall_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{model_type.upper()} Models: Overall Stacked Performance Metrics Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = charts_dir / f"{model_type.lower()}_stacked_bar_overall_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Overall metrics stacked bar saved: {plot_path}")
        plt.close()

    def _generate_grouped_bar_plots(self, final_results, charts_dir, model_type):
        """Generate grouped bar plots for individual category sizes and overall comparison"""
        print(f"\nGenerating {model_type.upper()} grouped bar plots...")
        
        # 1. Top-K Accuracy Grouped Bars (Individual)
        self._generate_topk_grouped_bars_individual(final_results, charts_dir, model_type)
        
        # 2. Performance Metrics Grouped Bars (Individual)
        self._generate_metrics_grouped_bars_individual(final_results, charts_dir, model_type)
        
        # 3. Top-K Accuracy Grouped Bars (Overall)
        self._generate_topk_grouped_bars_overall(final_results, charts_dir, model_type)
        
        # 4. Performance Metrics Grouped Bars (Overall)
        self._generate_metrics_grouped_bars_overall(final_results, charts_dir, model_type)
        
        # 5. Extended: Side-by-side category size comparison
        self._generate_category_size_comparison_grouped(final_results, charts_dir, model_type)
        
        # 6. Extended: Model performance heatmap comparison
        self._generate_model_performance_heatmap(final_results, charts_dir, model_type)

    def _generate_topk_grouped_bars_individual(self, final_results, charts_dir, model_type):
        """Generate Top-K accuracy grouped bars for each category size"""
        print(f"  → Generating Top-K grouped bars for individual category sizes...")
        
        for n_categories, model_results_dict in final_results.items():
            if not model_results_dict:
                continue
            
            plot_data = {
                'model_label': [],
                'top1_accuracy': [],
                'top3_accuracy': [],
                'top5_accuracy': []
            }
            
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                label = f"{model_name}\n({feature_type.upper()})"
                
                top1 = result.get('top1_accuracy', result.get('accuracy', 0))
                top3 = result.get('top3_accuracy', 0)
                top5 = result.get('top5_accuracy', 0)
                
                plot_data['model_label'].append(label)
                plot_data['top1_accuracy'].append(top1)
                plot_data['top3_accuracy'].append(top3)
                plot_data['top5_accuracy'].append(top5)
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            x = np.arange(len(plot_data['model_label']))
            width = 0.25
            
            # Create grouped bars
            bars1 = ax.bar(x - width, plot_data['top1_accuracy'], width, 
                    label='Top-1 Accuracy', color='#3498db', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x, plot_data['top3_accuracy'], width,
                    label='Top-3 Accuracy', color='#2ecc71', edgecolor='black', linewidth=1.5)
            bars3 = ax.bar(x + width, plot_data['top5_accuracy'], width,
                    label='Top-5 Accuracy', color='#f39c12', edgecolor='black', linewidth=1.5)
            
            # Add value labels on each bar
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.02:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold', color='black')
            
            ax.set_xlabel('Model (Feature Type)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
            ax.set_title(f'{model_type.upper()} Models: Grouped Top-K Accuracy - Top {n_categories} Categories',
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_data['model_label'], fontsize=10, rotation=15, ha='right')
            ax.set_ylim(0, 1.15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
            
            plt.tight_layout()
            
            plot_path = charts_dir / f"{model_type.lower()}_grouped_bar_topk_top_{n_categories}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Top-K grouped bar saved: {plot_path}")
            plt.close()

    def _generate_metrics_grouped_bars_individual(self, final_results, charts_dir, model_type):
        """Generate performance metrics grouped bars for each category size"""
        print(f"  → Generating performance metrics grouped bars for individual category sizes...")
        
        for n_categories, model_results_dict in final_results.items():
            if not model_results_dict:
                continue
            
            plot_data = {
                'model_label': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
            
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                label = f"{model_name}\n({feature_type.upper()})"
                
                plot_data['model_label'].append(label)
                plot_data['accuracy'].append(result.get('accuracy', 0))
                plot_data['precision'].append(result.get('macro_precision', 0))
                plot_data['recall'].append(result.get('macro_recall', 0))
                plot_data['f1_score'].append(result.get('macro_f1', 0))
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            x = np.arange(len(plot_data['model_label']))
            width = 0.2
            
            # Create grouped bars
            bars1 = ax.bar(x - 1.5*width, plot_data['accuracy'], width, 
                    label='Accuracy', color='#3498db', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x - 0.5*width, plot_data['precision'], width,
                    label='Precision', color='#e74c3c', edgecolor='black', linewidth=1.5)
            bars3 = ax.bar(x + 0.5*width, plot_data['recall'], width,
                    label='Recall', color='#2ecc71', edgecolor='black', linewidth=1.5)
            bars4 = ax.bar(x + 1.5*width, plot_data['f1_score'], width,
                    label='F1-Score', color='#f39c12', edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bars in [bars1, bars2, bars3, bars4]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.02:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom',
                            fontsize=8, fontweight='bold', color='black')
            
            ax.set_xlabel('Model (Feature Type)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Metric Score', fontsize=13, fontweight='bold')
            ax.set_title(f'{model_type.upper()} Models: Grouped Performance Metrics - Top {n_categories} Categories',
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_data['model_label'], fontsize=10, rotation=15, ha='right')
            ax.set_ylim(0, 1.15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
            
            plt.tight_layout()
            
            plot_path = charts_dir / f"{model_type.lower()}_grouped_bar_metrics_top_{n_categories}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Metrics grouped bar saved: {plot_path}")
            plt.close()

    def _generate_topk_grouped_bars_overall(self, final_results, charts_dir, model_type):
        """Generate overall Top-K accuracy grouped bar comparison across all category sizes"""
        print(f"  → Generating overall Top-K grouped bar comparison...")
        
        overall_data = {}
        
        for n_categories, model_results_dict in final_results.items():
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                combo_key = f"{model_name}_{feature_type}"
                
                if combo_key not in overall_data:
                    overall_data[combo_key] = {
                        'model_name': model_name,
                        'feature_type': feature_type,
                        'categories': [],
                        'top1': [],
                        'top3': [],
                        'top5': []
                    }
                
                overall_data[combo_key]['categories'].append(int(n_categories))
                overall_data[combo_key]['top1'].append(result.get('top1_accuracy', result.get('accuracy', 0)))
                overall_data[combo_key]['top3'].append(result.get('top3_accuracy', 0))
                overall_data[combo_key]['top5'].append(result.get('top5_accuracy', 0))
        
        if not overall_data:
            return
        
        # Sort by categories
        for key in overall_data:
            indices = sorted(range(len(overall_data[key]['categories'])),
                            key=lambda i: overall_data[key]['categories'][i])
            for metric in ['categories', 'top1', 'top3', 'top5']:
                overall_data[key][metric] = [overall_data[key][metric][i] for i in indices]
        
        # Create subplots
        n_combos = len(overall_data)
        n_cols = min(3, n_combos)
        n_rows = (n_combos + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_combos == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, (combo_key, data) in enumerate(overall_data.items()):
            ax = axes[idx]
            
            categories = data['categories']
            top1 = data['top1']
            top3 = data['top3']
            top5 = data['top5']
            
            x = np.arange(len(categories))
            width = 0.25
            
            bars1 = ax.bar(x - width, top1, width, label='Top-1',
                    color='#3498db', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x, top3, width, label='Top-3',
                    color='#2ecc71', edgecolor='black', linewidth=1.5)
            bars3 = ax.bar(x + width, top5, width, label='Top-5',
                    color='#f39c12', edgecolor='black', linewidth=1.5)
            
            # Add labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.02:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
            
            model_label = f"{data['model_name']} ({data['feature_type'].upper()})"
            ax.set_title(model_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Number of Categories', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1.15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.legend(loc='lower right', fontsize=9)
        
        # Hide extra subplots
        for idx in range(len(overall_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{model_type.upper()} Models: Overall Grouped Top-K Accuracy Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = charts_dir / f"{model_type.lower()}_grouped_bar_overall_topk_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Overall Top-K grouped bar saved: {plot_path}")
        plt.close()

    def _generate_metrics_grouped_bars_overall(self, final_results, charts_dir, model_type):
        """Generate overall performance metrics grouped bar comparison across all category sizes"""
        print(f"  → Generating overall performance metrics grouped bar comparison...")
        
        overall_data = {}
        
        for n_categories, model_results_dict in final_results.items():
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                combo_key = f"{model_name}_{feature_type}"
                
                if combo_key not in overall_data:
                    overall_data[combo_key] = {
                        'model_name': model_name,
                        'feature_type': feature_type,
                        'categories': [],
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': []
                    }
                
                overall_data[combo_key]['categories'].append(int(n_categories))
                overall_data[combo_key]['accuracy'].append(result.get('accuracy', 0))
                overall_data[combo_key]['precision'].append(result.get('macro_precision', 0))
                overall_data[combo_key]['recall'].append(result.get('macro_recall', 0))
                overall_data[combo_key]['f1_score'].append(result.get('macro_f1', 0))
        
        if not overall_data:
            return
        
        # Sort by categories
        for key in overall_data:
            indices = sorted(range(len(overall_data[key]['categories'])),
                            key=lambda i: overall_data[key]['categories'][i])
            for metric in ['categories', 'accuracy', 'precision', 'recall', 'f1_score']:
                overall_data[key][metric] = [overall_data[key][metric][i] for i in indices]
        
        # Create subplots
        n_combos = len(overall_data)
        n_cols = min(3, n_combos)
        n_rows = (n_combos + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_combos == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, (combo_key, data) in enumerate(overall_data.items()):
            ax = axes[idx]
            
            categories = data['categories']
            accuracy = data['accuracy']
            precision = data['precision']
            recall = data['recall']
            f1_score = data['f1_score']
            
            x = np.arange(len(categories))
            width = 0.2
            
            bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy',
                    color='#3498db', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision',
                    color='#e74c3c', edgecolor='black', linewidth=1.5)
            bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall',
                    color='#2ecc71', edgecolor='black', linewidth=1.5)
            bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score',
                    color='#f39c12', edgecolor='black', linewidth=1.5)
            
            # Add total labels
            for i in range(len(categories)):
                total = accuracy[i] + precision[i] + recall[i] + f1_score[i]
                ax.text(i, total + 0.05, f'{total:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='darkred')
            
            model_label = f"{data['model_name']} ({data['feature_type'].upper()})"
            ax.set_title(model_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Number of Categories', fontsize=11, fontweight='bold')
            ax.set_ylabel('Metric Score', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1.5)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.legend(loc='upper left', fontsize=9)
        
        # Hide extra subplots
        for idx in range(len(overall_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{model_type.upper()} Models: Overall Grouped Performance Metrics Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = charts_dir / f"{model_type.lower()}_grouped_bar_overall_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Overall metrics grouped bar saved: {plot_path}")
        plt.close()

    def _generate_category_size_comparison_grouped(self, final_results, charts_dir, model_type):
        """Extended: Generate comparison across different category sizes for all models"""
        print(f"  → Generating category size comparison grouped bars...")
        
        # Organize data by category size
        category_comparison = {}
        all_models = set()
        
        for n_categories, model_results_dict in final_results.items():
            category_comparison[n_categories] = {}
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                combo_key = f"{model_name}_{feature_type}"
                all_models.add(combo_key)
                
                category_comparison[n_categories][combo_key] = {
                    'model_name': model_name,
                    'feature_type': feature_type,
                    'accuracy': result.get('accuracy', 0),
                    'macro_f1': result.get('macro_f1', 0)
                }
        
        if not category_comparison or not all_models:
            return
        
        # Create figure for accuracy comparison
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sorted_categories = sorted(category_comparison.keys())
        all_models_list = sorted(list(all_models))
        
        x = np.arange(len(sorted_categories))
        width = 0.8 / len(all_models_list)
        
        colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        
        for model_idx, model_key in enumerate(all_models_list):
            accuracies = []
            for cat in sorted_categories:
                if model_key in category_comparison[cat]:
                    accuracies.append(category_comparison[cat][model_key]['accuracy'])
                else:
                    accuracies.append(0)
            
            color = colors_list[model_idx % len(colors_list)]
            bars = ax.bar(x + width * (model_idx - len(all_models_list)/2), accuracies, width,
                    label=model_key.replace('_', ' '), color=color, edgecolor='black', linewidth=1)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Number of Categories', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_type.upper()} Models: Accuracy Across Category Sizes',
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_categories, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
        
        plt.tight_layout()
        
        plot_path = charts_dir / f"{model_type.lower()}_grouped_bar_category_size_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Category size comparison grouped bar saved: {plot_path}")
        plt.close()

    def _generate_model_performance_heatmap(self, final_results, charts_dir, model_type):
        """Extended: Generate performance heatmap for all models and category sizes"""
        print(f"  → Generating model performance heatmap...")
        
        import pandas as pd
        
        # Organize data for heatmap
        heatmap_data = {}
        all_models = set()
        all_categories = set()
        
        for n_categories, model_results_dict in final_results.items():
            all_categories.add(n_categories)
            for model_key, result in model_results_dict.items():
                model_name = result.get('model_name', 'Unknown')
                feature_type = result.get('feature_type', 'unknown')
                combo_key = f"{model_name}_{feature_type}"
                all_models.add(combo_key)
                
                if combo_key not in heatmap_data:
                    heatmap_data[combo_key] = {}
                
                heatmap_data[combo_key][n_categories] = result.get('macro_f1', 0)
        
        if not heatmap_data:
            return
        
        # Create DataFrame
        sorted_models = sorted(list(all_models))
        sorted_categories = sorted(list(all_categories))
        
        df = pd.DataFrame(index=sorted_models, columns=sorted_categories)
        
        for model in sorted_models:
            for cat in sorted_categories:
                df.loc[model, cat] = heatmap_data.get(model, {}).get(cat, 0)
        
        df = df.astype(float)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(sorted_categories)))
        ax.set_yticks(np.arange(len(sorted_models)))
        ax.set_xticklabels(sorted_categories, fontsize=11)
        ax.set_yticklabels([m.replace('_', ' ') for m in sorted_models], fontsize=10)
        
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add text annotations
        for i in range(len(sorted_models)):
            for j in range(len(sorted_categories)):
                value = df.values[i, j]
                text = ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                            color="black" if value < 0.5 else "white",
                            fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Number of Categories', fontsize=13, fontweight='bold')
        ax.set_ylabel('Model (Feature Type)', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_type.upper()} Models: Macro F1-Score Performance Heatmap',
                    fontsize=15, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Macro F1-Score', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = charts_dir / f"{model_type.lower()}_performance_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Performance heatmap saved: {plot_path}")
        plt.close()



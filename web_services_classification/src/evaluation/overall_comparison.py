"""
Enhanced Overall Performance Comparison Module
Combines ML, DL, BERT, DeepSeek, and Fusion model results for comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
from pathlib import Path
from math import pi

from src.config import (
    CATEGORY_SIZES, RESULTS_CONFIG, ML_CONFIG, DL_CONFIG, PREPROCESSING_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OverallPerformanceAnalyzer:
    """Enhanced analyzer for combined ML, DL, BERT, DeepSeek, and Fusion model performance"""
    
    def __init__(self):
        # Create overall results directory
        self.overall_dir = RESULTS_CONFIG.get("overall_results_path", Path("results") / "overall")
        self.overall_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced model naming patterns
        self.naming_patterns = {
            "logistic_regression": "LogisticRegression",
            "random_forest": "RandomForest", 
            "xgboost": "XGBoost",
            "bilstm": "BiLSTM",
            "roberta_base": "RoBERTa-Base",
            "roberta_large": "RoBERTa-Large",
            "deepseek_7b_base": "DeepSeek-7B-Base",
            # Fusion patterns
            "deepseek_roberta_fusion": "DeepSeek-RoBERTa-Fusion",
            "deepseek_roberta_fusion_concat": "DeepSeek-RoBERTa-Fusion-Concat",
            "deepseek_roberta_fusion_average": "DeepSeek-RoBERTa-Fusion-Average",
            "deepseek_roberta_fusion_weighted": "DeepSeek-RoBERTa-Fusion-Weighted",
            "deepseek_roberta_fusion_gating": "DeepSeek-RoBERTa-Fusion-Gating"
        }
        
    def load_all_results(self):
        """Load ML, DL, BERT, DeepSeek, and Fusion results"""
        results = {
            'ml': None,
            'dl': None, 
            'bert': None,
            'deepseek': None,
            'fusion': None
        }
        
        # Define result files for each model type
        result_files = {
            'ml': RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl",
            'dl': RESULTS_CONFIG["dl_comparisons_path"] / "dl_final_results.pkl",
            'bert': RESULTS_CONFIG["bert_comparisons_path"] / "bert_final_results.pkl",
            'deepseek': RESULTS_CONFIG["deepseek_comparisons_path"] / "deepseek_final_results.pkl",
            'fusion': RESULTS_CONFIG["fusion_comparisons_path"] / "fusion_final_results.pkl"
        }
        
        # Load each result type
        for model_type, file_path in result_files.items():
            try:
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                    results[model_type] = data
                    logger.info(f"Loaded {model_type.upper()} results: {len(data)} category sizes")
                else:
                    logger.warning(f"{model_type.upper()} results file not found: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {model_type.upper()} results: {e}")
        
        return results
    
    def normalize_data_structure(self, data, model_type):
        """Normalize different data structures to a common format"""
        normalized = {}
        
        if not data:
            return normalized
        
        try:
            for n_categories, category_data in data.items():
                normalized[n_categories] = []
                
                if model_type in ['ml', 'dl']:
                    # ML/DL format: list of dictionaries per category
                    if isinstance(category_data, list):
                        for entry in category_data:
                            normalized_entry = self._normalize_entry(entry, model_type)
                            if normalized_entry:
                                normalized[n_categories].append(normalized_entry)
                    elif isinstance(category_data, dict):
                        # Handle nested dictionary format
                        for model_key, model_data in category_data.items():
                            normalized_entry = self._normalize_entry(model_data, model_type, model_key)
                            if normalized_entry:
                                normalized[n_categories].append(normalized_entry)
                
                elif model_type in ['bert', 'deepseek', 'fusion']:
                    # BERT/DeepSeek/Fusion format: dictionary with model_feature keys
                    if isinstance(category_data, dict):
                        for model_key, model_data in category_data.items():
                            normalized_entry = self._normalize_entry(model_data, model_type, model_key)
                            if normalized_entry:
                                normalized[n_categories].append(normalized_entry)
        
        except Exception as e:
            logger.error(f"Error normalizing {model_type} data: {e}")
        
        return normalized
    
    def _normalize_entry(self, entry, model_type, model_key=None):
        """Normalize a single entry to common format"""
        try:
            # Extract model name and feature type
            if model_key:
                # For BERT/DeepSeek/Fusion: model_key like "RoBERTa_Base_raw_text" or "DeepSeek_RoBERTa_Fusion_Concat"
                parts = model_key.split('_')
                if len(parts) >= 2:
                    model_name = '_'.join(parts[:-1])  # Everything except last part
                    feature_type = parts[-1]  # Last part is feature type
                else:
                    model_name = model_key
                    feature_type = entry.get('feature_type', 'raw_text')
            else:
                # For ML/DL: extract from entry
                model_name = entry.get('model', entry.get('model_name', 'Unknown'))
                feature_type = entry.get('feature_type', 'unknown')
            
            # Create normalized entry
            normalized = {
                'model': model_name,
                'model_type': model_type.upper(),
                'feature_type': feature_type,
                'n_categories': entry.get('n_categories', 0),
                'accuracy': entry.get('accuracy', entry.get('top1_accuracy', 0)),
                'precision': entry.get('precision', entry.get('macro_precision', 0)),
                'recall': entry.get('recall', entry.get('macro_recall', 0)),
                'f1_score': entry.get('f1_score', entry.get('macro_f1', 0)),
                'top1_accuracy': entry.get('top1_accuracy', entry.get('accuracy', 0)),
                'top3_accuracy': entry.get('top3_accuracy', 0),
                'top5_accuracy': entry.get('top5_accuracy', 0),
                'training_time': entry.get('training_time', 0),
                'inference_time': entry.get('inference_time', 0)
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing entry: {e}")
            return None
    
    def combine_results_for_plotting(self, all_results):
        """Combine all model results into unified structure for plotting"""
        combined_metrics = {}
        
        # Process each model type
        for model_type, data in all_results.items():
            if not data:
                continue
                
            # Normalize data structure
            normalized_data = self.normalize_data_structure(data, model_type)
            
            # Process normalized data
            for n_categories, results in normalized_data.items():
                for entry in results:
                    model_key = f"{entry['model']} ({entry['model_type']})"
                    feature_type = entry['feature_type']
                    
                    if model_key not in combined_metrics:
                        combined_metrics[model_key] = {}
                    if feature_type not in combined_metrics[model_key]:
                        combined_metrics[model_key][feature_type] = {
                            'n': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                            'top1_accuracy': [], 'top3_accuracy': [], 'top5_accuracy': [],
                            'training_time': [], 'inference_time': [], 'model_type': entry['model_type']
                        }
                    
                    # Append data
                    metrics = combined_metrics[model_key][feature_type]
                    metrics['n'].append(entry['n_categories'])
                    metrics['accuracy'].append(entry['accuracy'])
                    metrics['precision'].append(entry['precision'])
                    metrics['recall'].append(entry['recall'])
                    metrics['f1_score'].append(entry['f1_score'])
                    metrics['top1_accuracy'].append(entry['top1_accuracy'])
                    metrics['top3_accuracy'].append(entry['top3_accuracy'])
                    metrics['top5_accuracy'].append(entry['top5_accuracy'])
                    metrics['training_time'].append(entry['training_time'])
                    metrics['inference_time'].append(entry['inference_time'])
        
        return combined_metrics
    
    def generate_combined_line_plots(self, combined_metrics):
        """Generate line plots comparing all model types"""
        print("\nGenerating combined line plots for all model types...")
        
        # Define colors for different model types
        model_colors = {
            'ML': ['#1f77b4', '#ff7f0e', '#2ca02c'],      # Blue tones
            'DL': ['#d62728', '#9467bd', '#8c564b'],      # Red/purple tones  
            'BERT': ['#e377c2', '#7f7f7f', '#bcbd22'],    # Pink/gray tones
            'DEEPSEEK': ['#17becf', '#ff9896', '#c5b0d5'], # Cyan/light tones
            'FUSION': ['#28a745', '#ffc107', '#dc3545', '#6c757d']  # Green/yellow/red/gray for 4 fusion types
        }
        
        metrics_config = {
            'accuracy': 'Accuracy',
            'precision': 'Precision (Macro)',
            'recall': 'Recall (Macro)', 
            'f1_score': 'F1-Score (Macro)',
            'top1_accuracy': 'Top-1 Accuracy',
            'top3_accuracy': 'Top-3 Accuracy',
            'top5_accuracy': 'Top-5 Accuracy',
            'training_time': 'Training Time (seconds)',
            'inference_time': 'Inference Time (seconds)'
        }
        
        for metric, ylabel in metrics_config.items():
            plt.figure(figsize=(16, 10))
            
            color_indices = {'ML': 0, 'DL': 0, 'BERT': 0, 'DEEPSEEK': 0, 'FUSION': 0}
            
            for model, features in combined_metrics.items():
                for feature_type, data in features.items():
                    if len(data['n']) == 0:
                        continue
                        
                    label = f"{model} ({feature_type.upper()})"
                    model_type = data['model_type']
                    
                    # Choose color and style based on model type
                    colors = model_colors.get(model_type, ['#000000'])
                    color = colors[color_indices[model_type] % len(colors)]
                    color_indices[model_type] += 1
                    
                    # Different line styles for different model types
                    linestyles = {'ML': '-', 'DL': '--', 'BERT': '-.', 'DEEPSEEK': ':', 'FUSION': '-'}
                    linestyle = linestyles.get(model_type, '-')
                    
                    plt.plot(data['n'], data[metric], marker='o', label=label, 
                            linewidth=2.5, color=color, linestyle=linestyle, markersize=6)
            
            plt.title(f'Overall Model Comparison: {ylabel} vs Number of Categories', fontsize=16, fontweight='bold')
            plt.xlabel('Number of Web Service Categories', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()
            
            plot_path = self.overall_dir / f"Overall_Comparison_{metric}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Combined line plot saved: {plot_path}")
            plt.close()


    def generate_combined_bar_plots(self, combined_metrics):
        """Generate bar plots comparing all model types (mean values)"""
        print("\nGenerating combined BAR plots for all model types...")
    
        # Define colors for different model types
        model_colors = {
            'ML': ['#1f77b4', '#ff7f0e', '#2ca02c'],      
            'DL': ['#d62728', '#9467bd', '#8c564b'],      
            'BERT': ['#e377c2', '#7f7f7f', '#bcbd22'],    
            'DEEPSEEK': ['#17becf', '#ff9896', '#c5b0d5'], 
            'FUSION': ['#28a745', '#ffc107', '#dc3545', '#6c757d']  
        }
    
        metrics_config = {
            'accuracy': 'Accuracy',
            'precision': 'Precision (Macro)',
            'recall': 'Recall (Macro)', 
            'f1_score': 'F1-Score (Macro)',
            'top1_accuracy': 'Top-1 Accuracy',
            'top3_accuracy': 'Top-3 Accuracy',
            'top5_accuracy': 'Top-5 Accuracy',
            'training_time': 'Training Time (seconds)',
            'inference_time': 'Inference Time (seconds)'
        }
    
    
        for metric, ylabel in metrics_config.items():
            plt.figure(figsize=(18, 10))
    
            bar_labels = []
            bar_values = []
            bar_colors = []
    
            # Collect bar data
            for model, features in combined_metrics.items():
                for feature_type, data in features.items():
                    if metric not in data or len(data[metric]) == 0:
                        continue
    
                    label = f"{model}-{feature_type.upper()}"
                    bar_labels.append(label)
                    bar_values.append(np.mean(data[metric]))
    
                    model_type = data['model_type']
                    bar_colors.append(model_colors.get(model_type, ['#000000'])[0])
    
            # Create bar chart
            x_positions = np.arange(len(bar_labels))
            plt.bar(x_positions, bar_values, color=bar_colors, alpha=0.85)
    
            plt.xticks(x_positions, bar_labels, rotation=45, ha='right', fontsize=10)
            plt.ylabel(ylabel, fontsize=14)
            plt.title(f'Model Comparison (BAR): {ylabel} (Average)', fontsize=16, fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
    
            bar_plot_path = self.overall_dir / f"Overall_BAR_{metric}.png"
            plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Bar plot saved: {bar_plot_path}")
            plt.close()


    def generate_overall_stacked_bars(self, combined_metrics):
        """Generate overall stacked bar plots for all models"""
        print("\nGenerating overall stacked bar plots...")
        
        # Flatten the nested structure first
        flattened_metrics = self._flatten_combined_metrics(combined_metrics)
        
        if not flattened_metrics:
            print("  No data available for stacked bars")
            return
        
        # 1. Top-K Stacked Bars
        self._generate_overall_topk_stacked_bars(flattened_metrics)
        
        # 2. Performance Metrics Stacked Bars
        self._generate_overall_metrics_stacked_bars(flattened_metrics)
        
        print("✓ Overall stacked bar plots generated")

    def _flatten_combined_metrics(self, combined_metrics):
        """Flatten the nested combined_metrics structure for stacked bars"""
        flattened = {}
        
        for model_key, feature_data in combined_metrics.items():
            for feature_type, metrics in feature_data.items():
                # Create a unique key combining model and feature
                flat_key = f"{model_key}_{feature_type}"
                flattened[flat_key] = metrics.copy()
        
        return flattened

    def _generate_overall_topk_stacked_bars(self, combined_metrics):
        """Generate Top-K accuracy stacked bars across all models"""
        print("  → Generating overall Top-K stacked bars...")
        
        model_colors = {
            'ML': ['#3498db', '#2980b9', '#1f618d'],
            'DL': ['#e74c3c', '#c0392b', '#a93226'],
            'BERT': ['#2ecc71', '#27ae60', '#1e8449'],
            'DEEPSEEK': ['#f39c12', '#e67e22', '#d35400'],
            'FUSION': ['#9b59b6', '#8e44ad', '#7d3c98']
        }
        
        # Get all unique category sizes
        all_categories = set()
        for data in combined_metrics.values():
            if 'n' in data:
                all_categories.update(data['n'])
        
        category_sizes = sorted(list(all_categories))
        
        # Prepare data for each category size
        for n_cat in category_sizes:
            plot_data = {
                'labels': [],
                'top1': [],
                'top3_gain': [],
                'top5_gain': [],
                'colors': []
            }
            
            for model_key, data in combined_metrics.items():
                if 'n' not in data or n_cat not in data['n']:
                    continue
                
                idx = data['n'].index(n_cat)
                model_type = data.get('model_type', 'UNKNOWN')
                
                # Create label
                label = model_key.replace('_', ' ')
                
                top1 = data['top1_accuracy'][idx]
                top3 = data['top3_accuracy'][idx]
                top5 = data['top5_accuracy'][idx]
                
                plot_data['labels'].append(label)
                plot_data['top1'].append(top1)
                plot_data['top3_gain'].append(max(0, top3 - top1))
                plot_data['top5_gain'].append(max(0, top5 - top3))
                plot_data['colors'].append(model_colors.get(model_type, ['#000000'])[0])
            
            if not plot_data['labels']:
                continue
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(16, 8))
            
            x_pos = np.arange(len(plot_data['labels']))
            width = 0.7
            
            # Stack bars
            p1 = ax.bar(x_pos, plot_data['top1'], width,
                    label='Top-1 Accuracy', color='#3498db', edgecolor='black', linewidth=1)
            p2 = ax.bar(x_pos, plot_data['top3_gain'], width,
                    bottom=plot_data['top1'],
                    label='Top-3 Gain', color='#2ecc71', edgecolor='black', linewidth=1)
            
            bottom_top5 = [plot_data['top1'][i] + plot_data['top3_gain'][i] 
                        for i in range(len(plot_data['top1']))]
            p3 = ax.bar(x_pos, plot_data['top5_gain'], width,
                    bottom=bottom_top5,
                    label='Top-5 Gain', color='#f39c12', edgecolor='black', linewidth=1)
            
            # Add value labels
            for i in range(len(plot_data['labels'])):
                t1 = plot_data['top1'][i]
                t3g = plot_data['top3_gain'][i]
                t5g = plot_data['top5_gain'][i]
                total = t1 + t3g + t5g
                
                # Total on top
                ax.text(i, total + 0.02, f'{total:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black')
            
            ax.set_xlabel('Model (Feature Type)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Cumulative Accuracy', fontsize=13, fontweight='bold')
            ax.set_title(f'Overall Comparison: Stacked Top-K Accuracy - Top {n_cat} Categories',
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_data['labels'], fontsize=9, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=11)
            
            plt.tight_layout()
            
            plot_path = self.overall_dir / f"Overall_Stacked_Bar_TopK_top_{n_cat}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Overall Top-K stacked bar saved for {n_cat} categories")
            plt.close()

    def _generate_overall_metrics_stacked_bars(self, combined_metrics):
        """Generate performance metrics stacked bars across all models"""
        print("  → Generating overall performance metrics stacked bars...")
        
        model_colors = {
            'ML': ['#3498db', '#2980b9', '#1f618d'],
            'DL': ['#e74c3c', '#c0392b', '#a93226'],
            'BERT': ['#2ecc71', '#27ae60', '#1e8449'],
            'DEEPSEEK': ['#f39c12', '#e67e22', '#d35400'],
            'FUSION': ['#9b59b6', '#8e44ad', '#7d3c98']
        }
        
        # Get all unique category sizes
        all_categories = set()
        for data in combined_metrics.values():
            if 'n' in data:
                all_categories.update(data['n'])
        
        category_sizes = sorted(list(all_categories))
        
        # Prepare data for each category size
        for n_cat in category_sizes:
            plot_data = {
                'labels': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'colors': []
            }
            
            for model_key, data in combined_metrics.items():
                if 'n' not in data or n_cat not in data['n']:
                    continue
                
                idx = data['n'].index(n_cat)
                model_type = data.get('model_type', 'UNKNOWN')
                
                # Create label
                label = model_key.replace('_', ' ')
                
                plot_data['labels'].append(label)
                plot_data['accuracy'].append(data['accuracy'][idx])
                plot_data['precision'].append(data['precision'][idx])
                plot_data['recall'].append(data['recall'][idx])
                plot_data['f1_score'].append(data['f1_score'][idx])
                plot_data['colors'].append(model_colors.get(model_type, ['#000000'])[0])
            
            if not plot_data['labels']:
                continue
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(16, 8))
            
            x_pos = np.arange(len(plot_data['labels']))
            width = 0.7
            
            # Stack bars
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
            
            # Add value labels
            for i in range(len(plot_data['labels'])):
                acc = plot_data['accuracy'][i]
                prec = plot_data['precision'][i]
                rec = plot_data['recall'][i]
                f1 = plot_data['f1_score'][i]
                total = acc + prec + rec + f1
                
                # Total on top
                ax.text(i, total + 0.05, f'{total:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black')
            
            ax.set_xlabel('Model (Feature Type)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Cumulative Metrics Score', fontsize=13, fontweight='bold')
            ax.set_title(f'Overall Comparison: Stacked Performance Metrics - Top {n_cat} Categories',
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_data['labels'], fontsize=9, rotation=45, ha='right')
            max_total = max([plot_data['accuracy'][i] + plot_data['precision'][i] + 
                            plot_data['recall'][i] + plot_data['f1_score'][i] 
                            for i in range(len(plot_data['labels']))])
            ax.set_ylim(0, max_total + 0.3)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=11)
            
            plt.tight_layout()
            
            plot_path = self.overall_dir / f"Overall_Stacked_Bar_Metrics_top_{n_cat}_categories.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Overall metrics stacked bar saved for {n_cat} categories")
            plt.close()

    def generate_overall_model_wise_stacked_bars(self, combined_metrics):
        """Generate stacked bars showing each model's performance across category sizes"""
        print("\nGenerating model-wise stacked bar comparisons...")
        
        # Flatten the nested structure first
        flattened_metrics = self._flatten_combined_metrics(combined_metrics)
        
        if not flattened_metrics:
            print("  No data available for model-wise stacked bars")
            return
        
        # Group data by model
        model_data = {}
        
        for model_key, data in flattened_metrics.items():
            if 'n' not in data or 'model_type' not in data:
                continue
                
            if model_key not in model_data:
                model_data[model_key] = {
                    'model_name': model_key,
                    'model_type': data['model_type'],
                    'categories': [],
                    'top1': [],
                    'top3': [],
                    'top5': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1_score': []
                }
            
            for i, n_cat in enumerate(data['n']):
                model_data[model_key]['categories'].append(n_cat)
                model_data[model_key]['top1'].append(data['top1_accuracy'][i])
                model_data[model_key]['top3'].append(data['top3_accuracy'][i])
                model_data[model_key]['top5'].append(data['top5_accuracy'][i])
                model_data[model_key]['accuracy'].append(data['accuracy'][i])
                model_data[model_key]['precision'].append(data['precision'][i])
                model_data[model_key]['recall'].append(data['recall'][i])
                model_data[model_key]['f1_score'].append(data['f1_score'][i])
        
        if not model_data:
            print("  No valid model data for model-wise stacked bars")
            return
        
        # Create subplots for Top-K comparison
        n_models = len(model_data)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, (model_key, data) in enumerate(model_data.items()):
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
            for i in range(len(categories)):
                total = top1[i] + top3_gain[i] + top5_gain[i]
                ax.text(i, total + 0.02, f'{total:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
            
            model_label = f"{data['model_name'].replace('_', ' ')} ({data['model_type']})"
            ax.set_title(model_label, fontsize=11, fontweight='bold')
            ax.set_xlabel('Number of Categories', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cumulative Accuracy', fontsize=10, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='lower left', fontsize=8)
        
        # Hide extra subplots
        for idx in range(len(model_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Overall Comparison: Model-wise Stacked Top-K Accuracy',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = self.overall_dir / "Overall_Stacked_Bar_Model_Wise_TopK.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print("✓ Model-wise Top-K stacked bars saved")
        plt.close()
        
        # Create subplots for Metrics comparison
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, (model_key, data) in enumerate(model_data.items()):
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
            
            model_label = f"{data['model_name'].replace('_', ' ')} ({data['model_type']})"
            ax.set_title(model_label, fontsize=11, fontweight='bold')
            ax.set_xlabel('Number of Categories', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cumulative Metrics', fontsize=10, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, fontsize=9)
            max_total = max([accuracy[i] + precision[i] + recall[i] + f1_score[i] 
                            for i in range(len(categories))])
            ax.set_ylim(0, max_total + 0.3)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=8)
        
        # Hide extra subplots
        for idx in range(len(model_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Overall Comparison: Model-wise Stacked Performance Metrics',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = self.overall_dir / "Overall_Stacked_Bar_Model_Wise_Metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print("✓ Model-wise metrics stacked bars saved")
        plt.close()

    
    def generate_summary_comparison(self, all_results):
        """Generate enhanced summary comparison tables"""
        print("\nGenerating comprehensive summary comparison tables...")
        
        # Combine all results into one DataFrame
        all_summary_data = []
        
        for model_type, data in all_results.items():
            if not data:
                continue
                
            normalized_data = self.normalize_data_structure(data, model_type)
            
            for n_categories, results in normalized_data.items():
                for entry in results:
                    summary_entry = {
                        'Categories': n_categories,
                        'Model_Type': entry['model_type'],
                        'Model': entry['model'],
                        'Feature': entry['feature_type'],
                        'Accuracy': entry['accuracy'],
                        'Precision': entry['precision'],
                        'Recall': entry['recall'],
                        'F1-Score': entry['f1_score'],
                        'Top-1': entry['top1_accuracy'],
                        'Top-3': entry['top3_accuracy'],
                        'Top-5': entry['top5_accuracy'],
                        'Training_Time': entry['training_time'],
                        'Inference_Time': entry['inference_time']
                    }
                    all_summary_data.append(summary_entry)
        
        if all_summary_data:
            summary_df = pd.DataFrame(all_summary_data)
            summary_df = summary_df.round(4)
            
            # Save comprehensive summary
            summary_path = self.overall_dir / "Overall_Performance_Summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"✓ Overall summary table saved: {summary_path}")
            
            # Generate enhanced analysis
            print(f"\n{'='*80}")
            print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
            print(f"{'='*80}")
            
            # Best performers by metric
            print("\nBest Overall Performers by Metric:")
            for metric in ['Accuracy', 'F1-Score', 'Top-1', 'Top-3', 'Top-5']:
                if metric in summary_df.columns and summary_df[metric].max() > 0:
                    best = summary_df.loc[summary_df[metric].idxmax()]
                    print(f"  {metric:12}: {best['Model']:25} ({best['Model_Type']:8}, {best['Feature']:8}) "
                          f"on {best['Categories']:2} categories = {best[metric]:.4f}")
            
            # Best by model type
            print(f"\nBest Performer by Model Type:")
            for model_type in ['ML', 'DL', 'BERT', 'DEEPSEEK', 'FUSION']:
                type_data = summary_df[summary_df['Model_Type'] == model_type]
                if len(type_data) > 0:
                    best_row = type_data.loc[type_data['Top-1'].idxmax()]
                    print(f"  {model_type:8}: {best_row['Model']:25} ({best_row['Feature']:8}) "
                          f"on {best_row['Categories']:2} categories")
                    print(f"           Top-1: {best_row['Top-1']:.4f}, F1: {best_row['F1-Score']:.4f}, "
                          f"Training: {best_row['Training_Time']:.2f}s")
            
            # Feature type analysis
            print(f"\nFeature Type Effectiveness:")
            feature_analysis = summary_df.groupby('Feature').agg({
                'Top-1': 'mean',
                'F1-Score': 'mean',
                'Training_Time': 'mean'
            }).round(4)
            
            for feature in feature_analysis.index:
                row = feature_analysis.loc[feature]
                print(f"  {feature:10}: Avg Top-1: {row['Top-1']:.4f}, "
                      f"Avg F1: {row['F1-Score']:.4f}, Avg Training: {row['Training_Time']:.2f}s")
            
            # Model count summary
            print(f"\nModel Coverage Summary:")
            coverage = summary_df.groupby(['Model_Type', 'Categories']).size().unstack(fill_value=0)
            print(coverage)
            
            print(f"{'='*80}")
            
        else:
            print("No valid data found for summary generation.")
            summary_df = None
        
        return summary_df
    
    def generate_all_comparisons(self):
        """Generate all overall comparison visualizations including BERT, DeepSeek, and Fusion"""
        print("Starting Comprehensive Performance Analysis (ML + DL + BERT + DeepSeek + Fusion)...")
        
        # Load all results
        all_results = self.load_all_results()
        
        # Check if any data was loaded
        has_data = any(data is not None for data in all_results.values())
        if not has_data:
            print("No results found. Run training phases first.")
            return
        
        # Report what was loaded
        loaded_types = [model_type for model_type, data in all_results.items() if data is not None]
        print(f"Loaded results for: {', '.join(loaded_types).upper()}")
        
        # Combine data for plotting
        combined_metrics = self.combine_results_for_plotting(all_results)
        
        if combined_metrics:
            print(f"Combined metrics for {len(combined_metrics)} model configurations")
            
            # Generate all visualizations
            try:
                self.generate_combined_line_plots(combined_metrics)
                print("✓ Line plots generated")
            except Exception as e:
                logger.error(f"Error generating line plots: {e}")

            # Generate all visualizations
            try:
                self.generate_combined_bar_plots(combined_metrics)
                print("✓ Bar plots generated")
            except Exception as e:
                logger.error(f"Error generating Bar plots: {e}")

            try:
                self.generate_overall_stacked_bars(combined_metrics)
                print("✓ Overall stacked bar plots generated")
            except Exception as e:
                logger.error(f"Error generating overall stacked bars: {e}")
            
            try:
                self.generate_overall_model_wise_stacked_bars(combined_metrics)
                print("✓ Model-wise stacked bar plots generated")
            except Exception as e:
                logger.error(f"Error generating model-wise stacked bars: {e}")
            
            try:
                self.generate_summary_comparison(all_results)
                print("✓ Summary comparison generated")
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
            
            print(f"\nAll overall comparison visualizations saved to: {self.overall_dir}")
        else:
            print("No valid data found for comparison plots.")
            
        # Debug information
        print(f"\nDEBUG: Result file check:")
        for model_type in ['ml', 'dl', 'bert', 'deepseek', 'fusion']:
            file_path = RESULTS_CONFIG[f"{model_type}_comparisons_path"] / f"{model_type}_final_results.pkl"
            status = "EXISTS" if file_path.exists() else "MISSING"
            print(f"  {model_type.upper():8}: {file_path} - {status}")


def main():
    """Main function to run enhanced overall comparison analysis"""
    analyzer = OverallPerformanceAnalyzer()
    analyzer.generate_all_comparisons()


if __name__ == "__main__":
    main()

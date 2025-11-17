"""
Enhanced RoBERTa Models for Web Service Classification
Enhanced with standardized naming and proper result handling
"""

import pandas as pd
import numpy as np
import logging
import json
import time
import traceback
import random
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
)

# Import configuration and utilities
from src.config import (
    CATEGORY_SIZES, SAVED_MODELS_CONFIG, BERT_CONFIG, 
    PREPROCESSING_CONFIG, RANDOM_SEED, RESULTS_CONFIG
)
from src.evaluation.evaluate import ModelEvaluator
from src.utils.utils import FileNamingStandard

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class RoBERTaModelTrainer:
    """Enhanced RoBERTa model trainer with standardized naming"""
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types and Path objects to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: RoBERTaModelTrainer.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [RoBERTaModelTrainer.make_json_serializable(item) for item in obj]
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

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.config = BERT_CONFIG
        self.evaluator = ModelEvaluator()
        
        # Configure GPU
        self._configure_gpu()
        
        # Create results directories
        self._create_directories()
        
    def _configure_gpu(self):
        """Configure GPU memory and device"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                logger.info("Using CPU")
        except Exception as e:
            logger.warning(f"GPU configuration warning: {e}")
            self.device = torch.device("cpu")
    
    def _create_directories(self):
        """Create necessary directories for results and visualizations"""
        directories = [
            RESULTS_CONFIG['bert_results_path'],
            RESULTS_CONFIG['bert_comparisons_path'],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create category-specific directories
        for n_categories in CATEGORY_SIZES:
            category_dir = RESULTS_CONFIG['bert_category_paths'][n_categories]
            category_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created result directories for RoBERTa models")
    
    def get_model_config(self, model_name):
        """Get model-specific configuration from config"""
        model_config = self.config.copy()
        
        # Adjust batch sizes based on model from config
        if model_name in self.config.get('batch_sizes', {}):
            batch_config = self.config['batch_sizes'][model_name]
            model_config['per_device_train_batch_size'] = batch_config['train_batch_size']
            model_config['per_device_eval_batch_size'] = batch_config['eval_batch_size']
        
        # Adjust learning rate for large model
        if 'large' in model_name:
            model_config['learning_rate'] = 1e-5  # Lower LR for large model
            model_config['warmup_steps'] = 1000   # More warmup steps
        
        return model_config
    
    def load_tokenizer(self, model_name=None):
        """Load RoBERTa tokenizer"""
        try:
            if model_name is None:
                model_name = self.config['model_name']
            
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded RoBERTa tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def tokenize_function(self, batch):
        """Tokenize text batch for RoBERTa"""
        return self.tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.config['max_length']
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)[2]
        }
    
    def prepare_datasets(self, n_categories):
        """Load and prepare datasets for training"""
        try:
            logger.info(f"Loading datasets for top_{n_categories}_categories")
            
            # Load datasets using correct config paths
            splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
            if not splits_dir.exists():
                raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
            
            train_df = pd.read_csv(splits_dir / 'train.csv')
            val_df = pd.read_csv(splits_dir / 'val.csv')
            test_df = pd.read_csv(splits_dir / 'test.csv')
            
            logger.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Use cleaned_text if available, otherwise use original text
            text_column = 'cleaned_text' if 'cleaned_text' in train_df.columns else 'text'
            if text_column not in train_df.columns:
                # Try original column name from the CSV
                text_column = 'Service Description' if 'Service Description' in train_df.columns else train_df.columns[0]
            
            # Create datasets
            train_data = {
                'text': train_df[text_column].astype(str).tolist(),
                'labels': train_df['encoded_label'].tolist()
            }
            val_data = {
                'text': val_df[text_column].astype(str).tolist(),
                'labels': val_df['encoded_label'].tolist()
            }
            test_data = {
                'text': test_df[text_column].astype(str).tolist(),
                'labels': test_df['encoded_label'].tolist()
            }
            
            # Convert to Hugging Face datasets
            train_dataset = Dataset.from_dict(train_data)
            val_dataset = Dataset.from_dict(val_data)
            test_dataset = Dataset.from_dict(test_data)
            
            # Tokenize datasets
            train_dataset = train_dataset.map(self.tokenize_function, batched=True)
            val_dataset = val_dataset.map(self.tokenize_function, batched=True)
            test_dataset = test_dataset.map(self.tokenize_function, batched=True)
            
            # Remove text column
            train_dataset = train_dataset.remove_columns(["text"])
            val_dataset = val_dataset.remove_columns(["text"])
            test_dataset = test_dataset.remove_columns(["text"])
            
            logger.info("Datasets tokenized successfully")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error preparing datasets: {e}")
            raise
    
    def create_model(self, num_labels, model_name=None):
        """Create RoBERTa model for classification"""
        try:
            if model_name is None:
                model_name = self.config['model_name']
            
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels
            )
            logger.info(f"Created {model_name} with {num_labels} labels")
            return self.model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def plot_training_history(self, trainer, model_name, n_categories):
        """Create training history plots with standardized naming"""
        try:
            # Get training history from trainer
            log_history = trainer.state.log_history
            
            if not log_history:
                logger.warning("No training history available")
                return None
            
            # Extract training and validation metrics
            train_loss = []
            val_loss = []
            val_acc = []
            
            for log in log_history:
                if 'loss' in log:  # Training step
                    if 'epoch' in log:
                        train_loss.append(log['loss'])
                elif 'eval_loss' in log:  # Evaluation step
                    val_loss.append(log['eval_loss'])
                    val_acc.append(log.get('eval_accuracy', 0))
            
            if not train_loss or not val_loss:
                logger.warning("Insufficient training history for plotting")
                return None
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot loss
            ax1.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss', linewidth=2)
            ax1.plot(range(1, len(val_loss)+1), val_loss, label='Validation Loss', linewidth=2)
            ax1.set_title(f'{model_name} - Training & Validation Loss\n{n_categories} Categories')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracy
            if val_acc:
                ax2.plot(range(1, len(val_acc)+1), val_acc, label='Validation Accuracy', linewidth=2)
                ax2.set_title(f'{model_name} - Validation Accuracy\n{n_categories} Categories')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot using standardized naming
            plot_dir = RESULTS_CONFIG['bert_category_paths'][n_categories]
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            filename = FileNamingStandard.generate_training_history_filename(model_name, n_categories)
            plot_file = plot_dir / filename
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history plot saved: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Error creating training history plot: {str(e)}")
            return None
    
    def evaluate_roberta_model(self, trainer, test_dataset, model_name, n_categories, class_labels):
        """Comprehensive evaluation of RoBERTa model"""
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            # Get predictions
            start_time = time.time()
            predictions = trainer.predict(test_dataset)
            inference_time = time.time() - start_time
            
            y_true = predictions.label_ids
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_proba = predictions.predictions
            
            # Convert logits to probabilities
            y_proba = torch.softmax(torch.tensor(y_proba), dim=1).numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)
            
            micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='micro', zero_division=0
            )
            
            # Top-K accuracies using common evaluator
            # Convert y_true to one-hot for top-k calculation
            y_true_onehot = np.eye(n_categories)[y_true]
            top1_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=1)
            top3_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=3)
            top5_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=5)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create visualizations using common evaluator with standardized naming
            cm_plot_path = self.evaluator.generate_confusion_heatmap(
                cm, class_labels, model_name, n_categories, "raw_text", "bert"
            )
            report_path = self.evaluator.generate_classification_report_csv(
                y_true, y_pred, class_labels, model_name, n_categories, "raw_text", "bert"
            )
            
            # Compile results
            results = {
                'model_name': model_name,
                'feature_type': 'raw_text',
                'n_categories': int(n_categories),
                'top1_accuracy': float(top1_accuracy),
                'top3_accuracy': float(top3_accuracy),
                'top5_accuracy': float(top5_accuracy),
                'accuracy': float(accuracy),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_f1': float(macro_f1),
                'micro_precision': float(micro_precision),
                'micro_recall': float(micro_recall),
                'micro_f1': float(micro_f1),
                'confusion_matrix_plot': cm_plot_path,
                'classification_report_path': str(report_path),
                'inference_time': float(inference_time)
            }
            
            logger.info(f"{model_name} Evaluation Results:")
            logger.info(f"  Top-1 Accuracy: {top1_accuracy:.4f}")
            logger.info(f"  Top-3 Accuracy: {top3_accuracy:.4f}")
            logger.info(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
            logger.info(f"  Macro F1: {macro_f1:.4f}")
            logger.info(f"  Micro F1: {micro_f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def train_model_on_category(self, n_categories, model_name=None):
        """Train RoBERTa model on a specific category size"""
        try:
            if model_name is None:
                model_name = self.config['model_name']
            
            # Validate model name against config
            if model_name not in self.config['available_models'].values():
                raise ValueError(f"Model {model_name} not in available models: {list(self.config['available_models'].values())}")
            
            logger.info(f"Training {model_name} for top_{n_categories}_categories")
            
            # Clear any existing model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Get model-specific configuration
            model_config = self.get_model_config(model_name)
            
            # Load tokenizer for specific model
            self.load_tokenizer(model_name)
            
            # Prepare datasets
            train_dataset, val_dataset, test_dataset = self.prepare_datasets(n_categories)
            
            # Load class labels using common evaluator
            class_labels = self.evaluator.load_class_labels(n_categories)
            
            # Create model
            model = self.create_model(n_categories, model_name)
            
            # Setup training arguments with model-specific config
            model_dir = SAVED_MODELS_CONFIG['bert_models_path'] / f'top_{n_categories}_categories'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create model-specific output directory with standardized naming
            model_variant = model_name.replace('-', '_')
            clean_model_name = FileNamingStandard.standardize_model_name(model_name)
            output_dir = model_dir / f'{clean_model_name}_top_{n_categories}_categories'
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=model_config['num_train_epochs'],
                eval_strategy=model_config['eval_strategy'],
                per_device_train_batch_size=model_config['per_device_train_batch_size'],
                per_device_eval_batch_size=model_config['per_device_eval_batch_size'],
                logging_dir=str(output_dir / "logs"),
                logging_strategy=model_config['logging_strategy'],
                logging_steps=model_config['logging_steps'],
                save_strategy=model_config['save_strategy'],
                load_best_model_at_end=model_config['load_best_model_at_end'],
                metric_for_best_model=model_config['metric_for_best_model'],
                seed=model_config['seed'],
                learning_rate=model_config['learning_rate'],
                weight_decay=model_config['weight_decay'],
                warmup_steps=model_config['warmup_steps'],
                greater_is_better=model_config['greater_is_better'],
                fp16=model_config.get('fp16', False),
                dataloader_num_workers=model_config.get('dataloader_num_workers', 0),
                remove_unused_columns=model_config.get('remove_unused_columns', True),
                report_to=model_config.get('report_to', []),
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )
            
            # Train model
            print(f"\nTraining {model_name} on top_{n_categories}_categories...")
            print(f"Batch size: Train={model_config['per_device_train_batch_size']}, Eval={model_config['per_device_eval_batch_size']}")
            print(f"Learning rate: {model_config['learning_rate']}")
            
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save model with standardized naming
            model_filename = FileNamingStandard.generate_model_filename(
                model_name, 'raw_text', n_categories, 'model'
            )
            model_path = model_dir / model_filename
            trainer.save_model(str(model_path))
            logger.info(f"Model saved to {model_path}")
            
            # Create training history plot
            display_name = FileNamingStandard.standardize_model_name(model_name)
            history_plot_path = self.plot_training_history(trainer, display_name, n_categories)
            
            # Evaluate model
            eval_results = self.evaluate_roberta_model(trainer, test_dataset, display_name, n_categories, class_labels)
            eval_results['training_time'] = float(training_time)
            eval_results['model_path'] = str(model_path)
            eval_results['model_variant'] = model_name
            eval_results['training_history_plot'] = history_plot_path
            eval_results['batch_size'] = model_config['per_device_train_batch_size']
            eval_results['learning_rate'] = model_config['learning_rate']
            
            # Print metrics using common evaluator
            self.evaluator.print_model_metrics(eval_results, display_name, n_categories, "raw_text", training_time, "BERT")
            
            # Save performance data using common evaluator  
            self.evaluator.save_model_performance_data(eval_results, display_name, n_categories, "raw_text", "bert")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error training {model_name} for {n_categories} categories: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def save_results_for_overall_analysis(self, all_results):
        """Save results in the format expected by OverallPerformanceAnalyzer"""
        try:
            comparisons_path = RESULTS_CONFIG['bert_comparisons_path']
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            # Transform results into the format expected by OverallPerformanceAnalyzer
            formatted_results = {}
            
            for model_key, model_results in all_results.items():
                for n_categories, result in model_results.items():
                    if n_categories not in formatted_results:
                        formatted_results[n_categories] = {}
                    
                    # Create a key that matches the expected format
                    model_name = result.get('model_name', f'bert_model')
                    feature_type = result.get('feature_type', 'raw_text')
                    
                    # Use consistent naming pattern
                    clean_model_name = FileNamingStandard.standardize_model_name(model_name)
                    result_key = f"{clean_model_name}_{feature_type}"
                    formatted_results[n_categories][result_key] = result
            
            # Save as pickle file with the expected name
            pickle_file = comparisons_path / "bert_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(formatted_results, f)
            
            logger.info(f"BERT results saved for overall analysis: {pickle_file}")
            
            # Also save JSON for debugging
            json_file = comparisons_path / "bert_final_results.json"
            with open(json_file, 'w') as f:
                json_safe_results = self.make_json_serializable(formatted_results)
                json.dump(json_safe_results, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving BERT results for overall analysis: {e}")
    
    def train_roberta_models(self, categories=None):
        """Train both RoBERTa-base and RoBERTa-large models"""
        if categories is None:
            categories = CATEGORY_SIZES
        
        logger.info("Training RoBERTa models from config")
        
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING RoBERTa MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Category sizes: {categories}")
        print(f"Models: {list(self.config['available_models'].values())}")
        print(f"{'='*80}")
        
        # Train models from config
        for model_key, model_name in self.config['available_models'].items():
            print(f"\n{'-'*60}")
            print(f"TRAINING {model_name.upper()}")
            print(f"{'-'*60}")
            
            model_results = {}
            
            for n_categories in categories:
                print(f"\n>>> Processing top_{n_categories}_categories with {model_name}...")
                
                try:
                    results = self.train_model_on_category(n_categories, model_name)
                    model_results[n_categories] = results
                    
                    # Save individual results
                    category_dir = SAVED_MODELS_CONFIG['bert_models_path'] / f'top_{n_categories}_categories'
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as JSON with standardized naming
                    model_variant = model_name.replace('-', '_')
                    results_json = category_dir / f'bert_{model_variant}_results.json'
                    with open(results_json, 'w') as f:
                        json_safe_results = self.make_json_serializable(results)
                        json.dump(json_safe_results, f, indent=2)
                    
                    logger.info(f"Results saved to {results_json}")
                    logger.info(f"Training completed successfully for {model_name} on {n_categories} categories")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {n_categories} categories: {str(e)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
                
                # Clear GPU memory after each training
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            all_results[model_key] = model_results
        
        print(f"\n{'='*80}")
        print(f"\n{'='*80}")
        print(f"RoBERTa MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")

        # Print comparison if multiple models trained
        if len(all_results) > 1:
            self._print_roberta_comparison(all_results)

        # Save results for overall analysis
        self.save_results_for_overall_analysis(all_results)

        # Generate visualizations
        print(f"\n{'='*80}")
        print(f"GENERATING BERT VISUALIZATIONS")
        print(f"{'='*80}")
        try:
            print("Generating comparison plots...")
            self.plot_roberta_results_only()
            print("Generating radar plots...")
            self.evaluator.generate_radar_plots("bert", show_plots=False)
            print("All BERT visualizations completed successfully!")
        except Exception as e:
            logger.error(f"Error generating BERT visualizations: {e}")
            import traceback
            traceback.print_exc()
            print(f"Warning: Some visualizations may not have been generated due to errors.")

        return all_results
    
    def _print_roberta_comparison(self, all_results):
        """Print comparison between RoBERTa models"""
        print(f"\n{'='*80}")
        print(f"RoBERTa MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        for n_categories in CATEGORY_SIZES:
            results_for_category = {}
            
            # Collect results for this category
            for model_key, model_results in all_results.items():
                if n_categories in model_results:
                    results_for_category[model_key] = model_results[n_categories]
            
            if results_for_category:
                print(f"\nTop {n_categories} Categories Results:")
                print(f"{'Model':<15} {'Top-1 Acc':<10} {'Top-3 Acc':<10} {'Top-5 Acc':<10} {'Macro F1':<10} {'Training Time':<15}")
                print("-" * 85)
                
                # Sort by F1 score
                model_scores = []
                for model_key, result in results_for_category.items():
                    model_name = self.config['available_models'][model_key]
                    model_scores.append((
                        model_name,
                        result['top1_accuracy'],
                        result['top3_accuracy'],
                        result['top5_accuracy'],
                        result['macro_f1'],
                        result['training_time']
                    ))
                
                model_scores.sort(key=lambda x: x[4], reverse=True)  # Sort by F1
                
                for model_name, top1, top3, top5, f1, time_taken in model_scores:
                    print(f"{model_name:<15} {top1:<10.4f} {top3:<10.4f} {top5:<10.4f} {f1:<10.4f} {time_taken:<15.2f}")
                
                # Performance comparison for two models
                if len(model_scores) == 2:
                    base_f1 = next(score[4] for score in model_scores if 'base' in score[0])
                    large_f1 = next(score[4] for score in model_scores if 'large' in score[0])
                    improvement = large_f1 - base_f1
                    
                    print(f"\nPerformance Analysis:")
                    print(f"  RoBERTa-large vs RoBERTa-base F1 improvement: {improvement:+.4f}")
                    print(f"  Relative improvement: {(improvement/base_f1)*100:+.2f}%")
        
        print(f"{'='*80}")
    
    def train_all_categories(self):
        """Train RoBERTa models on all category sizes (uses config models)"""
        return self.train_roberta_models()
    
    def plot_roberta_results_only(self):
        """Convenience function to plot RoBERTa results with config paths"""
        results_file_path = RESULTS_CONFIG["bert_comparisons_path"] / "bert_final_results.pkl"
        charts_dir = RESULTS_CONFIG["bert_comparisons_path"] / "charts"
        
        self.evaluator.plot_results_comparison(results_file_path, charts_dir, "bert")


def main():
    """Main function to run comprehensive RoBERTa model training and analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RoBERTa Model Training for Web Service Classification")
    parser.add_argument("--model", type=str, default="both", 
                       choices=["roberta_base", "roberta_large", "both"],
                       help="RoBERTa model to train (default: both)")
    parser.add_argument("--categories", nargs="+", type=int, default=CATEGORY_SIZES,
                       help="Category sizes to train")
    
    args = parser.parse_args()
    
    trainer = RoBERTaModelTrainer()
    
    if args.model == "both":
        # Train both RoBERTa models from config
        results = trainer.train_roberta_models(args.categories)
    else:
        # Train single model
        model_name = BERT_CONFIG['available_models'][args.model]
        logger.info(f"Training single model: {model_name}")
        
        results = {}
        for n_categories in args.categories:
            results[n_categories] = trainer.train_model_on_category(n_categories, model_name)
    
    # Save final results
    out_file = SAVED_MODELS_CONFIG["bert_models_path"] / "bert_final_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json_safe_results = trainer.make_json_serializable(results)
        json.dump(json_safe_results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
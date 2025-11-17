"""
Enhanced DeepSeek Models for Web Service Classification
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
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Import configuration and utilities
from src.config import (
    CATEGORY_SIZES, SAVED_MODELS_CONFIG, DEEPSEEK_CONFIG, 
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


class DeepSeekModelTrainer:
    """Enhanced DeepSeek model trainer with standardized naming"""
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types and Path objects to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: DeepSeekModelTrainer.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DeepSeekModelTrainer.make_json_serializable(item) for item in obj]
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
        self.label_encoder = None
        self.config = DEEPSEEK_CONFIG
        self.evaluator = ModelEvaluator()
        
        # Configure GPU
        self._configure_gpu()
        
        # Create results directories
        self._create_directories()
        
        # Setup NLTK components
        self._setup_nltk()
        
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
            RESULTS_CONFIG['deepseek_results_path'],
            RESULTS_CONFIG['deepseek_comparisons_path'],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create category-specific directories
        for n_categories in CATEGORY_SIZES:
            category_dir = RESULTS_CONFIG['deepseek_category_paths'][n_categories]
            category_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created result directories for DeepSeek models")
    
    def _setup_nltk(self):
        """Download and setup NLTK components"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('punkt_tab')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in stemmed_tokens]
        
        return " ".join(lemmatized_tokens)
    
    def get_model_config(self, model_name):
        """Get model-specific configuration from config"""
        model_config = self.config.copy()
        
        # Adjust batch sizes based on model from config
        if model_name in self.config.get('batch_sizes', {}):
            batch_config = self.config['batch_sizes'][model_name]
            model_config['per_device_train_batch_size'] = batch_config['train_batch_size']
            model_config['per_device_eval_batch_size'] = batch_config['eval_batch_size']
        
        return model_config
    
    def load_tokenizer(self, model_name=None):
        """Load DeepSeek tokenizer"""
        try:
            if model_name is None:
                model_name = self.config['model_name']
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=self.config['trust_remote_code']
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Loaded DeepSeek tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def tokenize_function(self, batch):
        """Tokenize text batch for DeepSeek"""
        return self.tokenizer(
            batch["text"], 
            padding=self.config['padding'], 
            truncation=self.config['truncation'], 
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
            
            # Apply text cleaning if enabled in config
            if self.config.get('text_preprocessing', {}).get('clean_text', True):
                train_df[text_column] = train_df[text_column].astype(str).apply(self.clean_text)
                val_df[text_column] = val_df[text_column].astype(str).apply(self.clean_text)
                test_df[text_column] = test_df[text_column].astype(str).apply(self.clean_text)
            
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
    
    def create_peft_model(self, num_labels, model_name=None):
        """Create DeepSeek PEFT model for classification"""
        try:
            if model_name is None:
                model_name = self.config['model_name']
            
            # Setup quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config['quantization']['load_in_4bit'],
                bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
                bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
            )
            
            # Load base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=self.config['trust_remote_code'],
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # Prepare for training
            if self.config['gradient_checkpointing']:
                self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Setup LoRA config
            peft_config = LoraConfig(
                task_type=getattr(TaskType, self.config['lora']['task_type']),
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                lora_dropout=self.config['lora']['lora_dropout'],
                bias=self.config['lora']['bias']
            )
            
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Created PEFT DeepSeek model with {num_labels} labels")
            return self.model
            
        except Exception as e:
            logger.error(f"Error creating PEFT model: {e}")
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
            plot_dir = RESULTS_CONFIG['deepseek_category_paths'][n_categories]
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
    
    def evaluate_deepseek_model(self, trainer, test_dataset, model_name, n_categories, class_labels):
        """Comprehensive evaluation of DeepSeek model"""
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
                cm, class_labels, model_name, n_categories, "raw_text", "deepseek"
            )
            report_path = self.evaluator.generate_classification_report_csv(
                y_true, y_pred, class_labels, model_name, n_categories, "raw_text", "deepseek"
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
        """Train DeepSeek model on a specific category size"""
        try:
            if model_name is None:
                model_name = self.config['model_name']
            
            # Validate model name against config
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
            
            # Create PEFT model
            model = self.create_peft_model(n_categories, model_name)
            
            # Setup training arguments with model-specific config
            model_dir = SAVED_MODELS_CONFIG['deepseek_models_path'] / f'top_{n_categories}_categories'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create model-specific output directory with standardized naming
            model_variant = model_name.replace('-', '_').replace('/', '_')
            clean_model_name = FileNamingStandard.standardize_model_name(model_name)
            output_dir = model_dir / f'{clean_model_name}_top_{n_categories}_categories'
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=model_config['num_train_epochs'],
                eval_strategy=model_config['eval_strategy'],
                per_device_train_batch_size=model_config['per_device_train_batch_size'],
                per_device_eval_batch_size=model_config['per_device_eval_batch_size'],
                gradient_accumulation_steps=model_config['gradient_accumulation_steps'],
                logging_dir=str(output_dir / "logs"),
                logging_strategy="steps",
                logging_steps=model_config['logging_steps'],
                save_strategy=model_config['save_strategy'],
                save_total_limit=model_config['save_total_limit'],
                load_best_model_at_end=model_config['load_best_model_at_end'],
                metric_for_best_model=model_config['metric_for_best_model'],
                seed=model_config['random_state'],
                learning_rate=model_config['learning_rate'],
                greater_is_better=model_config['greater_is_better'],
                fp16=model_config.get('fp16', False),
                dataloader_num_workers=model_config.get('dataloader_num_workers', 0),
                remove_unused_columns=model_config.get('remove_unused_columns', True),
                report_to=model_config.get('report_to', "none"),
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
            
            # Save tokenizer and label encoder
            self.tokenizer.save_pretrained(str(model_path))
            if self.label_encoder:
                with open(model_path / "label_encoder.pkl", "wb") as f:
                    pickle.dump(self.label_encoder, f)
            
            logger.info(f"Model saved to {model_path}")
            
            # Create training history plot
            display_name = FileNamingStandard.standardize_model_name(model_name)
            history_plot_path = self.plot_training_history(trainer, display_name, n_categories)
            
            # Evaluate model
            eval_results = self.evaluate_deepseek_model(trainer, test_dataset, display_name, n_categories, class_labels)
            eval_results['training_time'] = float(training_time)
            eval_results['model_path'] = str(model_path)
            eval_results['model_variant'] = model_name
            eval_results['training_history_plot'] = history_plot_path
            eval_results['batch_size'] = model_config['per_device_train_batch_size']
            eval_results['learning_rate'] = model_config['learning_rate']
            
            # Print metrics using common evaluator
            self.evaluator.print_model_metrics(eval_results, display_name, n_categories, "raw_text", training_time, "DeepSeek")
            
            # Save performance data using common evaluator  
            self.evaluator.save_model_performance_data(eval_results, display_name, n_categories, "raw_text", "deepseek")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error training {model_name} for {n_categories} categories: {str(e)}")
            raise
    
    def save_results_for_overall_analysis(self, all_results):
        """Save results in the format expected by OverallPerformanceAnalyzer"""
        try:
            comparisons_path = RESULTS_CONFIG['deepseek_comparisons_path']
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            # Transform results into the format expected by OverallPerformanceAnalyzer
            formatted_results = {}
            
            for model_key, model_results in all_results.items():
                for n_categories, result in model_results.items():
                    if n_categories not in formatted_results:
                        formatted_results[n_categories] = {}
                    
                    # Create a key that matches the expected format
                    model_name = result.get('model_name', f'deepseek_model')
                    feature_type = result.get('feature_type', 'raw_text')
                    
                    # Use consistent naming pattern
                    clean_model_name = FileNamingStandard.standardize_model_name(model_name)
                    result_key = f"{clean_model_name}_{feature_type}"
                    formatted_results[n_categories][result_key] = result
            
            # Save as pickle file
            pickle_file = comparisons_path / "deepseek_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(formatted_results, f)
            
            logger.info(f"DeepSeek results saved for overall analysis: {pickle_file}")
            
            # Also save JSON for debugging
            json_file = comparisons_path / "deepseek_final_results.json"
            with open(json_file, 'w') as f:
                json_safe_results = self.make_json_serializable(formatted_results)
                json.dump(json_safe_results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving DeepSeek results for overall analysis: {e}")
    
    def train_deepseek_models(self, categories=None):
        """Train DeepSeek models"""
        if categories is None:
            categories = CATEGORY_SIZES
        
        logger.info("Training DeepSeek models from config")
        
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING DEEPSEEK MODEL TRAINING PIPELINE")
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
                    category_dir = SAVED_MODELS_CONFIG['deepseek_models_path'] / f'top_{n_categories}_categories'
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as JSON with standardized naming
                    model_variant = model_name.replace('-', '_').replace('/', '_')
                    results_json = category_dir / f'deepseek_{model_variant}_results.json'
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
        print(f"DEEPSEEK MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        
        # Save results for overall analysis
        self.save_results_for_overall_analysis(all_results)
        
        return all_results
    
    def train_all_categories(self):
        """Train DeepSeek models on all category sizes (uses config models)"""
        return self.train_deepseek_models()
    
    def plot_deepseek_results_only(self):
        """Convenience function to plot DeepSeek results with config paths"""
        results_file_path = RESULTS_CONFIG["deepseek_comparisons_path"] / "deepseek_final_results.pkl"
        charts_dir = RESULTS_CONFIG["deepseek_comparisons_path"] / "charts"
        
        self.evaluator.plot_results_comparison(results_file_path, charts_dir, "deepseek")


def main():
    """Main function to run comprehensive DeepSeek model training and analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek Model Training for Web Service Classification")
    parser.add_argument("--categories", nargs="+", type=int, default=CATEGORY_SIZES,
                       help="Category sizes to train")
    
    args = parser.parse_args()
    
    trainer = DeepSeekModelTrainer()
    
    # Train DeepSeek model from config
    results = trainer.train_deepseek_models(args.categories)
    
    # Save final results
    out_file = SAVED_MODELS_CONFIG["deepseek_models_path"] / "deepseek_final_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json_safe_results = trainer.make_json_serializable(results)
        json.dump(json_safe_results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
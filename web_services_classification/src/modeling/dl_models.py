"""
Enhanced Deep Learning Models for Web Service Classification
Enhanced with standardized naming and proper result handling
"""

import pandas as pd
import numpy as np
import logging
import json
import time
import traceback
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import configuration and utilities
from src.config import (
    CATEGORY_SIZES, SAVED_MODELS_CONFIG, DL_CONFIG, 
    PREPROCESSING_CONFIG, RANDOM_SEED, RESULTS_CONFIG
)
from src.preprocessing.feature_extraction import FeatureExtractor
from src.evaluation.evaluate import ModelEvaluator
from src.utils.utils import FileNamingStandard

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class DLModelTrainer:
    """Enhanced DL model trainer with standardized naming"""
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types and Path objects to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: DLModelTrainer.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DLModelTrainer.make_json_serializable(item) for item in obj]
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
        self.models = {}
        self.feature_extractor = FeatureExtractor()
        self.config = DL_CONFIG['bilstm']
        self.callbacks_config = DL_CONFIG['callbacks']
        self.evaluator = ModelEvaluator()
        
        # Configure GPU memory growth
        self._configure_gpu()
        
        # Create results directories
        self._create_directories()
        
    def _configure_gpu(self):
        """Configure GPU memory growth to prevent OOM errors"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured GPU memory growth for {len(gpus)} GPU(s)")
            else:
                logger.info("No GPUs found, using CPU")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    
    def _create_directories(self):
        """Create necessary directories for results and visualizations"""
        directories = [
            RESULTS_CONFIG['dl_results_path'],
            RESULTS_CONFIG['dl_comparisons_path'],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create category-specific directories
        for n_categories in CATEGORY_SIZES:
            category_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            category_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created result directories for DL models")
    
    def create_bilstm_model_tfidf(self, input_dim, n_classes):
        """Create Dense Neural Network for TF-IDF features"""
        try:
            logger.info(f"Creating TF-IDF model with input_dim={input_dim}, n_classes={n_classes}")
            
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(512, activation='relu', name='dense_1'),
                Dropout(self.config['dropout_rate'], name='dropout_1'),
                Dense(256, activation='relu', name='dense_2'), 
                Dropout(self.config['dropout_rate'], name='dropout_2'),
                Dense(128, activation='relu', name='dense_3'),
                Dropout(self.config['dropout_rate'], name='dropout_3'),
                Dense(n_classes, activation='softmax', name='output')
            ])
            
            optimizer = Adam(learning_rate=self.config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=self.config['loss'],
                metrics=self.config['metrics']
            )
            
            logger.info("TF-IDF model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF model: {str(e)}")
            raise
    
    def create_bilstm_model_sbert(self, input_dim, n_classes):
        """Create model for SBERT embeddings"""
        try:
            logger.info(f"Creating SBERT model with input_dim={input_dim}, n_classes={n_classes}")
            
            if input_dim >= 384:  # Typical SBERT dimension
                model = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(256, activation='relu', name='dense_1'),
                    Dropout(self.config['dropout_rate'], name='dropout_1'),
                    Dense(128, activation='relu', name='dense_2'),
                    Dropout(self.config['dropout_rate'], name='dropout_2'),
                    Dense(64, activation='relu', name='dense_3'),
                    Dropout(self.config['dropout_rate'], name='dropout_3'),
                    Dense(n_classes, activation='softmax', name='output')
                ])
            else:
                model = Sequential([
                    Input(shape=(input_dim,)),
                    Dense(self.config['lstm_units'], activation='relu', name='dense_1'),
                    Dropout(self.config['dropout_rate'], name='dropout_1'),
                    Dense(self.config['lstm_units']//2, activation='relu', name='dense_2'),
                    Dropout(self.config['dropout_rate'], name='dropout_2'),
                    Dense(n_classes, activation='softmax', name='output')
                ])
            
            optimizer = Adam(learning_rate=self.config['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss=self.config['loss'],
                metrics=self.config['metrics']
            )
            
            logger.info("SBERT model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error creating SBERT model: {str(e)}")
            raise
    
    def prepare_data_for_dl(self, X_train, X_val, X_test, y_train, y_val, y_test, n_classes):
        """Prepare data for deep learning models"""
        try:
            logger.info("Preparing data for deep learning...")
            
            # Convert sparse matrices to dense for TF-IDF
            if hasattr(X_train, 'toarray'):
                logger.info("Converting sparse matrices to dense arrays")
                X_train = X_train.toarray()
                X_val = X_val.toarray()
                X_test = X_test.toarray()
            
            # Convert to float32 for better performance
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            X_test = X_test.astype(np.float32)
            
            logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            logger.info(f"Label ranges - Train: {y_train.min()}-{y_train.max()}, Val: {y_val.min()}-{y_val.max()}")
            
            # Validate labels
            max_label = max(y_train.max(), y_val.max(), y_test.max())
            if max_label >= n_classes:
                raise ValueError(f"Label values exceed categories. Max: {max_label}, n_categories: {n_classes}")
            
            # One-hot encode labels
            y_train_encoded = to_categorical(y_train, num_classes=n_classes)
            y_val_encoded = to_categorical(y_val, num_classes=n_classes)
            y_test_encoded = to_categorical(y_test, num_classes=n_classes)
            
            logger.info(f"Encoded labels - Train: {y_train_encoded.shape}, Val: {y_val_encoded.shape}, Test: {y_test_encoded.shape}")
            
            return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def plot_training_history(self, history, model_name, n_categories, feature_type):
        """Create training history plots with standardized naming"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot accuracy
            ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
            ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax1.set_title(f'{model_name} - Training & Validation Accuracy\n{n_categories} Categories ({feature_type.upper()})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot loss
            ax2.plot(history['loss'], label='Training Loss', linewidth=2)
            ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
            ax2.set_title(f'{model_name} - Training & Validation Loss\n{n_categories} Categories ({feature_type.upper()})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot with standardized naming
            plot_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Use standardized filename
            filename = FileNamingStandard.generate_training_history_filename(model_name, n_categories)
            # Modify to include feature type in the filename
            filename = filename.replace('.png', f'_{feature_type}.png')
            plot_file = plot_dir / filename
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history plot saved: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Error creating training history plot: {str(e)}")
            return None
    
    def evaluate_dl_model(self, model, X_test, y_test, model_name, n_categories, feature_type, class_labels):
        """Comprehensive evaluation of deep learning model"""
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            # Get predictions and probabilities
            start_time = time.time()
            y_proba = model.predict(X_test, verbose=0)
            inference_time = time.time() - start_time
            
            y_pred = np.argmax(y_proba, axis=1)
            y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
            
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
            top1_accuracy = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=1)
            top3_accuracy = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=3)
            top5_accuracy = self.evaluator.calculate_top_k_accuracy(y_test, y_proba, k=5)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create visualizations using common evaluator
            cm_plot_path = self.evaluator.generate_confusion_heatmap(
                cm, class_labels, model_name, n_categories, feature_type, "dl"
            )
            report_path = self.evaluator.generate_classification_report_csv(
                y_true, y_pred, class_labels, model_name, n_categories, feature_type, "dl"
            )
            
            # Compile results
            results = {
                'model_name': model_name,
                'feature_type': feature_type,
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
    
    def train_model_on_category(self, n_categories, feature_type='sbert'):
        """Train model on a specific category size"""
        try:
            logger.info(f"Training DL model for top_{n_categories}_categories ({feature_type})")
            
            tf.keras.backend.clear_session()
            
            # Load datasets
            splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
            if not splits_dir.exists():
                raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
            
            train_df = pd.read_csv(splits_dir / 'train.csv')
            val_df = pd.read_csv(splits_dir / 'val.csv')
            test_df = pd.read_csv(splits_dir / 'test.csv')
            
            logger.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Load class labels
            class_labels = self.evaluator.load_class_labels(n_categories)
            
            # Get features
            if feature_type == 'tfidf':
                logger.info("Loading TF-IDF features...")
                self.feature_extractor.load_tfidf_vectorizer(n_categories)
                X_train = self.feature_extractor.tfidf_vectorizer.transform(train_df['cleaned_text'])
                X_val = self.feature_extractor.tfidf_vectorizer.transform(val_df['cleaned_text'])
                X_test = self.feature_extractor.tfidf_vectorizer.transform(test_df['cleaned_text'])
            elif feature_type == 'sbert':
                logger.info("Loading SBERT features...")
                X_train = self.feature_extractor.load_sbert_features(n_categories, 'train')
                X_val = self.feature_extractor.load_sbert_features(n_categories, 'val')
                X_test = self.feature_extractor.load_sbert_features(n_categories, 'test')
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            # Get labels
            y_train = train_df['encoded_label'].values
            y_val = val_df['encoded_label'].values
            y_test = test_df['encoded_label'].values
            
            # Prepare data
            X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded = self.prepare_data_for_dl(
                X_train, X_val, X_test, y_train, y_val, y_test, n_categories
            )
            
            # Create model
            input_dim = X_train.shape[1]
            logger.info(f"Model input dimension: {input_dim}, classes: {n_categories}")
            
            if feature_type == 'tfidf':
                model = self.create_bilstm_model_tfidf(input_dim, n_categories)
            else:
                model = self.create_bilstm_model_sbert(input_dim, n_categories)
            
            # Setup callbacks and paths
            model_dir = SAVED_MODELS_CONFIG['dl_models_path'] / f'top_{n_categories}_categories'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Use standardized filename for model
            model_filename = FileNamingStandard.generate_model_filename(
                'BiLSTM', feature_type, n_categories, 'h5'
            )
            model_path = model_dir / model_filename
            
            callbacks = [
                EarlyStopping(
                    monitor=self.callbacks_config['early_stopping']['monitor'],
                    patience=self.callbacks_config['early_stopping']['patience'],
                    restore_best_weights=self.callbacks_config['early_stopping']['restore_best_weights'],
                    verbose=1
                ),
                ModelCheckpoint(
                    str(model_path),
                    monitor=self.callbacks_config['model_checkpoint']['monitor'],
                    save_best_only=self.callbacks_config['model_checkpoint']['save_best_only'],
                    save_weights_only=self.callbacks_config['model_checkpoint']['save_weights_only'],
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor=self.callbacks_config['reduce_lr']['monitor'],
                    factor=self.callbacks_config['reduce_lr']['factor'],
                    patience=self.callbacks_config['reduce_lr']['patience'],
                    min_lr=self.callbacks_config['reduce_lr']['min_lr'],
                    verbose=1
                )
            ]
            
            # Training parameters
            batch_size = min(self.config['batch_size'], len(X_train) // 10, 32)
            epochs = self.config['epochs']
            
            logger.info(f"Training with batch_size={batch_size}, epochs={epochs}")
            
            # Train model
            print(f"\nTraining BiLSTM with {feature_type.upper()} features on top_{n_categories}_categories...")
            
            start_time = time.time()
            history = model.fit(
                X_train, y_train_encoded,
                validation_data=(X_val, y_val_encoded),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Load best model
            if model_path.exists():
                model.load_weights(str(model_path))
                logger.info("Loaded best model weights")
            
            # Create training history plot
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
            
            model_name = "BiLSTM"
            history_plot_path = self.plot_training_history(history_dict, model_name, n_categories, feature_type)
            
            # Save training history as JSON
            history_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            # Save training history as JSON
            history_dir = RESULTS_CONFIG['dl_category_paths'][n_categories]
            history_json_filename = FileNamingStandard.generate_training_history_filename(
                model_name, n_categories, 'json'
            ).replace('.json', f'_{feature_type}.json')
            history_json_file = history_dir / history_json_filename
            
            with open(history_json_file, 'w') as f:
                json.dump(history_dict, f, indent=2)
            
            # Evaluate model
            eval_results = self.evaluate_dl_model(model, X_test, y_test_encoded, model_name, n_categories, feature_type, class_labels)
            eval_results['training_time'] = float(training_time)
            eval_results['n_categories'] = n_categories
            eval_results['feature_type'] = feature_type
            eval_results['training_history'] = history_dict
            eval_results['training_history_plot'] = history_plot_path
            eval_results['training_history_json'] = str(history_json_file)
            eval_results['model_path'] = str(model_path)
            
            # Print metrics using common evaluator
            self.evaluator.print_model_metrics(eval_results, model_name, n_categories, feature_type, training_time, "DL")
            
            # Save performance data using common evaluator
            self.evaluator.save_model_performance_data(eval_results, model_name, n_categories, feature_type, "dl")
            
            logger.info(f"Model saved to {model_path}")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error training DL model for {n_categories} categories with {feature_type}: {str(e)}")
            raise
    
    def save_results_for_overall_analysis(self, all_results):
        """Save results in the format expected by OverallPerformanceAnalyzer"""
        try:
            comparisons_path = RESULTS_CONFIG['dl_comparisons_path']
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            # Transform results into the format expected by OverallPerformanceAnalyzer
            formatted_results = {}
            
            for feature_type, feature_results in all_results.items():
                for n_categories, result in feature_results.items():
                    if n_categories not in formatted_results:
                        formatted_results[n_categories] = {}
                    
                    # Create a key that matches the expected format
                    model_name = result.get('model_name', 'BiLSTM')
                    clean_model_name = FileNamingStandard.standardize_model_name(model_name)
                    result_key = f"{clean_model_name}_{feature_type}"
                    formatted_results[n_categories][result_key] = result
            
            # Save as pickle file
            pickle_file = comparisons_path / "dl_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(formatted_results, f)
            
            logger.info(f"DL results saved for overall analysis: {pickle_file}")
            
            # Also save JSON for debugging
            json_file = comparisons_path / "dl_final_results.json"
            with open(json_file, 'w') as f:
                json_safe_results = self.make_json_serializable(formatted_results)
                json.dump(json_safe_results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving DL results for overall analysis: {e}")
    
    def train_all_categories(self, feature_types=None):
        """Train models on all category sizes with specified feature types"""
        if feature_types is None:
            feature_types = DL_CONFIG['feature_types']
            
        logger.info("Starting DL model training for all categories...")
        
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING DL MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Category sizes: {CATEGORY_SIZES}")
        print(f"Feature types: {feature_types}")
        print(f"Models: {DL_CONFIG['models']}")
        print(f"{'='*80}")
        
        for feature_type in feature_types:
            all_results[feature_type] = {}
            print(f"\n{'-'*60}")
            print(f"TRAINING WITH {feature_type.upper()} FEATURES")
            print(f"{'-'*60}")
            
            for n_categories in CATEGORY_SIZES:
                print(f"\n>>> Processing top_{n_categories}_categories with {feature_type.upper()} features...")
                
                try:
                    results = self.train_model_on_category(n_categories, feature_type)
                    all_results[feature_type][n_categories] = results
                    
                    # Save individual results
                    category_dir = SAVED_MODELS_CONFIG['dl_models_path'] / f'top_{n_categories}_categories'
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as JSON with standardized naming
                    results_filename = f'dl_results_{feature_type}.json'
                    results_json = category_dir / results_filename
                    with open(results_json, 'w') as f:
                        json_safe_results = self.make_json_serializable(results)
                        json.dump(json_safe_results, f, indent=2)
                    
                    logger.info(f"Results saved to {results_json}")
                    logger.info(f"Training completed successfully for {n_categories} categories")
                
                except Exception as e:
                    logger.error(f"Error training for {n_categories} categories: {str(e)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
                
                # Clear memory after each training
                tf.keras.backend.clear_session()
        
        print(f"\n{'='*80}")
        print(f"DL MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        
        # Save results for overall analysis
        self.save_results_for_overall_analysis(all_results)
        
        # Print summary
        print(f"\nDL Final Results Summary:")
        print(f"  Pickle file: {RESULTS_CONFIG['dl_comparisons_path']}/dl_final_results.pkl")
        print(f"  Categories processed: {list(all_results[list(all_results.keys())[0]].keys()) if all_results else []}")
        
        # Generate visualizations
        print(f"\n{'='*80}")
        print(f"GENERATING DL VISUALIZATIONS")
        print(f"{'='*80}")
        try:
            print("Generating line plots, bar plots, and summary statistics...")
            self.plot_dl_results_only()
            print("Generating radar plots...")
            self.generate_dl_radar_plots_only()
            print("All DL visualizations completed successfully!")
        except Exception as e:
            logger.error(f"Error generating DL visualizations: {e}")
            print(f"Warning: Some visualizations may not have been generated due to errors.")
        
        return all_results
    
    def plot_dl_results_only(self):
        """Generate DL comparison plots"""
        results_file_path = RESULTS_CONFIG["dl_comparisons_path"] / "dl_final_results.pkl"
        charts_dir = RESULTS_CONFIG["dl_comparisons_path"] / "charts"
        self.evaluator.plot_results_comparison(results_file_path, charts_dir, "dl")
    
    def generate_dl_radar_plots_only(self, show_plots=False):
        """Generate DL radar plots"""
        self.evaluator.generate_radar_plots("dl", show_plots)


def main():
    """Main function to run comprehensive DL model training and analysis"""
    trainer = DLModelTrainer()
    results = trainer.train_all_categories()
    
    # Save final results
    out_file = SAVED_MODELS_CONFIG["dl_models_path"] / "dl_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json_safe_results = trainer.make_json_serializable(results)
        json.dump(json_safe_results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
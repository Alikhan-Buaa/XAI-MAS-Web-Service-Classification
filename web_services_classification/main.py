"""
Web Services Classification Project - Enhanced Main Entry Point
Run all phases of the classification pipeline with improved logging and result storage
"""

import argparse
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import (
    CATEGORY_SIZES, LOGGING_CONFIG, RESULTS_CONFIG
)
from src.preprocessing.data_analysis import DataAnalyzer
from src.preprocessing.enhanced_data_analysis import EnhancedDataAnalyzer
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.preprocessing.feature_extraction import FeatureExtractor
from src.modeling.ml_models import MLModelTrainer
from src.modeling.dl_models import DLModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.evaluation.overall_comparison import OverallPerformanceAnalyzer
from src.modeling.bert_models import RoBERTaModelTrainer
from src.modeling.deepseek_models import DeepSeekModelTrainer
from src.modeling.fusion_models import DeepSeekRoBERTaFusionTrainer
from src.utils.utils import setup_logging, get_timestamp
from src.explainability.ml_explainability import MLExplainability
from src.explainability.overall_explainability import generate_overall_charts
from src.explainability.deepseek_explainability import DeepSeekExplainability
from src.explainability.dl_explainability import DLExplainability
from src.explainability.bert_explainability import BERTExplainability
from src.explainability.fusion_explainability import FusionExplainability

class PipelineManager:
    """Enhanced pipeline manager with comprehensive logging and result tracking"""
    
    def __init__(self):
        self.results = {}
        self.execution_log = {
            'start_time': datetime.now().isoformat(),
            'phases_completed': [],
            'phases_failed': [],
            'total_execution_time': 0,
            'phase_timings': {}
        }
        self.logger = None
        
    def setup_project_logging(self, phase_name="main"):
        """Setup enhanced logging for the entire project"""
        log_file = LOGGING_CONFIG['log_files'].get(phase_name, LOGGING_CONFIG['log_files']['data_analysis'])
        
        setup_logging(
            log_file=log_file,
            level=LOGGING_CONFIG['level'],
            format_str=LOGGING_CONFIG['format']
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline Manager initialized for phase: {phase_name}")
    
    def log_phase_start(self, phase_name):
        """Log the start of a pipeline phase"""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"STARTING PHASE: {phase_name.upper()}")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*80}")
        return time.time()
    
    def log_phase_end(self, phase_name, start_time, success=True, error=None):
        """Log the end of a pipeline phase"""
        duration = time.time() - start_time
        self.execution_log['phase_timings'][phase_name] = duration
        
        if success:
            self.execution_log['phases_completed'].append(phase_name)
            self.logger.info(f"PHASE COMPLETED: {phase_name.upper()} ({duration:.2f}s)")
        else:
            self.execution_log['phases_failed'].append({
                'phase': phase_name,
                'error': str(error),
                'duration': duration
            })
            self.logger.error(f"PHASE FAILED: {phase_name.upper()} after {duration:.2f}s - {error}")
        
        self.logger.info(f"{'='*80}")
        return duration
    
    def _clean_for_json(self, obj):
        """Recursively clean objects to make them JSON serializable"""
        try:
            from scipy.sparse import issparse
            import numpy as np
        except ImportError:
            pass
        
        if 'scipy' in sys.modules and issparse(obj):
            return f"<Sparse Matrix {obj.shape}>"
        elif 'numpy' in sys.modules and isinstance(obj, np.ndarray):
            if obj.size > 100:  # Large arrays
                return f"<NumPy Array {obj.shape}>"
            else:
                return obj.tolist()
        elif 'numpy' in sys.modules and isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif 'numpy' in sys.modules and isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                try:
                    cleaned[str(key)] = self._clean_for_json(value)
                except Exception:
                    cleaned[str(key)] = f"<Non-serializable object: {type(value).__name__}>"
            return cleaned
        elif isinstance(obj, (list, tuple)):
            cleaned = []
            for item in obj:
                try:
                    cleaned.append(self._clean_for_json(item))
                except Exception:
                    cleaned.append(f"<Non-serializable object: {type(item).__name__}>")
            return cleaned
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool)):
            # Handle custom objects
            return f"<{type(obj).__name__} object>"
        else:
            try:
                json.dumps(obj)  # Test if it's JSON serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def save_execution_summary(self):
        """Save comprehensive execution summary with sparse matrix handling"""
        try:
            self.execution_log['end_time'] = datetime.now().isoformat()
            self.execution_log['total_execution_time'] = sum(self.execution_log['phase_timings'].values())
            
            # Clean the execution log and results for JSON serialization
            cleaned_execution_log = self._clean_for_json(self.execution_log.copy())
            cleaned_results = self._clean_for_json(self.results.copy())
            
            # Save execution log
            log_file = RESULTS_CONFIG['overall_results_path'] / f"pipeline_execution_{get_timestamp()}.json"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'w') as f:
                json.dump(cleaned_execution_log, f, indent=2)
            
            # Save results summary
            if cleaned_results:
                results_file = RESULTS_CONFIG['overall_results_path'] / f"pipeline_results_{get_timestamp()}.json"
                with open(results_file, 'w') as f:
                    json.dump(cleaned_results, f, indent=2)
            
            self.logger.info(f"Execution summary saved to: {log_file}")
            if cleaned_results:
                self.logger.info(f"Results summary saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save execution summary: {e}")
    
    def run_data_analysis_phase(self):
        """Run data analysis phase with enhanced logging"""
        phase_name = "data_analysis"
        start_time = self.log_phase_start(phase_name)
        
        try:
            analyzer = DataAnalyzer()
            results = analyzer.run_complete_analysis()
            self.results[phase_name] = {
                'status': 'completed',
                'summary': 'Data analysis completed successfully',
                'outputs': 'Analysis completed successfully'  # Don't store actual results
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_preprocessing_phase(self):
        """Run preprocessing phase with enhanced logging"""
        phase_name = "preprocessing"
        start_time = self.log_phase_start(phase_name)
        
        try:
            preprocessor = DataPreprocessor()
            results = preprocessor.process_all_categories()
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Preprocessed data for {len(CATEGORY_SIZES)} category sizes',
                'categories_processed': CATEGORY_SIZES,
                'outputs': 'Preprocessing completed successfully'  # Don't store actual results
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_feature_extraction_phase(self, categories=None, feature_types=None):
        """Run feature extraction with enhanced logging"""
        phase_name = "feature_extraction"
        start_time = self.log_phase_start(phase_name)
        
        try:
            if feature_types is None:
                feature_types = ['tfidf', 'sbert']
            
            extractor = FeatureExtractor()
            results = extractor.extract_features_all_categories(feature_types)
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Extracted {len(feature_types)} feature types for {len(CATEGORY_SIZES)} categories',
                'feature_types': feature_types,
                'categories_processed': CATEGORY_SIZES,
                'outputs': 'Feature extraction completed successfully'  # Don't store actual results
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_ml_training_phase(self):
        """Run ML model training phase with enhanced logging"""
        phase_name = "ml_training"
        start_time = self.log_phase_start(phase_name)
        
        try:
            ml_trainer = MLModelTrainer()
            results = ml_trainer.train_all_categories()
            
            # Extract summary statistics safely
            models_trained = []
            total_models = 0
            feature_types = []
            
            if results and isinstance(results, dict):
                for feature_type, feature_results in results.items():
                    feature_types.append(feature_type)
                    if isinstance(feature_results, dict):
                        for category, model_results in feature_results.items():
                            if isinstance(model_results, dict):
                                total_models += len(model_results)
                                models_trained.extend(list(model_results.keys()))
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Trained {total_models} ML models across {len(set(models_trained))} model types',
                'models_trained': list(set(models_trained)),
                'feature_types': feature_types,
                'categories_processed': CATEGORY_SIZES,
                'pickle_file': str(RESULTS_CONFIG["ml_comparisons_path"] / "ml_final_results.pkl")
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_dl_training_phase(self):
        """Run DL model training phase with enhanced logging"""
        phase_name = "dl_training"
        start_time = self.log_phase_start(phase_name)
        
        try:
            dl_trainer = DLModelTrainer()
            results = dl_trainer.train_all_categories()
            
            # Extract summary statistics safely
            models_trained = []
            total_models = 0
            feature_types = []
            
            if results and isinstance(results, dict):
                for feature_type, feature_results in results.items():
                    feature_types.append(feature_type)
                    if isinstance(feature_results, dict):
                        total_models += len(feature_results)
                        models_trained.append(f"BiLSTM_{feature_type}")
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Trained {total_models} DL models (BiLSTM) with different feature types',
                'models_trained': models_trained,
                'feature_types': feature_types,
                'categories_processed': CATEGORY_SIZES,
                'pickle_file': str(RESULTS_CONFIG["dl_comparisons_path"] / "dl_final_results.pkl")
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_bert_training_phase(self):
        """Run BERT model training phase with enhanced logging"""
        phase_name = "bert_training"
        start_time = self.log_phase_start(phase_name)
        
        try:
            bert_trainer = RoBERTaModelTrainer()
            results = bert_trainer.train_all_categories()
            
            # Extract summary statistics safely
            models_trained = []
            total_models = 0
            
            if results and isinstance(results, dict):
                for model_key, model_results in results.items():
                    models_trained.append(model_key)
                    if isinstance(model_results, dict):
                        total_models += len(model_results)
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Trained {total_models} BERT/RoBERTa models',
                'models_trained': models_trained,
                'categories_processed': CATEGORY_SIZES,
                'pickle_file': str(RESULTS_CONFIG["bert_comparisons_path"] / "bert_final_results.pkl")
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_fusion_training_phase(self):
        """Run RoBERTa Fusion model training phase with enhanced logging"""
        phase_name = "fusion_training"
        start_time = self.log_phase_start(phase_name)
        
        try:
            fusion_trainer = DeepSeekRoBERTaFusionTrainer()
            
            #=================================================
            # Adding these lines to force only concat category
            #=================================================
            # self.logger.info(" FORCING CONFIG : Running only CONCAT category in FUSION")
            # fusion_trainer.config['fusion_types'] = ['concat']
            #=================================================
            
            results = fusion_trainer.train_all_categories()
            
            # Extract summary statistics safely
            fusion_types = []
            total_models = 0
            
            if results and isinstance(results, dict):
                for fusion_key, fusion_results in results.items():
                    fusion_types.append(fusion_key)
                    if isinstance(fusion_results, dict):
                        total_models += len(fusion_results)
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Trained {total_models}  Fusion models',
                'fusion_types': fusion_types,
                'categories_processed': CATEGORY_SIZES,
                'pickle_file': str(RESULTS_CONFIG["fusion_comparisons_path"] / "fusion_final_results.pkl")
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_deepseek_training_phase(self):
        """Run DeepSeek model training phase with enhanced logging"""
        phase_name = "deepseek_training"
        start_time = self.log_phase_start(phase_name)
        
        try:
            trainer = DeepSeekModelTrainer()
            results = trainer.train_deepseek_models()
            
            # Extract summary statistics safely
            models_trained = []
            total_models = 0
            
            if results and isinstance(results, dict):
                for model_key, model_results in results.items():
                    models_trained.append(model_key)
                    if isinstance(model_results, dict):
                        total_models += len(model_results)
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Trained {total_models} DeepSeek models',
                'models_trained': models_trained,
                'categories_processed': CATEGORY_SIZES,
                'pickle_file': str(RESULTS_CONFIG["deepseek_comparisons_path"] / "deepseek_final_results.pkl")
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_evaluation_phase(self):
        """Run comprehensive model evaluation and comparison phase with enhanced logging"""
        phase_name = "evaluation"
        start_time = self.log_phase_start(phase_name)
        
        try:
            evaluator = ModelEvaluator()
            evaluation_results = {
                'ml_analysis': False,
                'dl_analysis': False,
                'bert_analysis': False,
                'fusion_analysis': False,
                'deepseek_analysis': False
            }
            
            # Check and process each model type
            model_types = [
                ('ml', RESULTS_CONFIG["ml_comparisons_path"]),
                ('dl', RESULTS_CONFIG["dl_comparisons_path"]),
                ('bert', RESULTS_CONFIG["bert_comparisons_path"]),
                ('fusion', RESULTS_CONFIG["fusion_comparisons_path"]),
                ('deepseek', RESULTS_CONFIG["deepseek_comparisons_path"])
            ]
            
            for model_type, comparisons_path in model_types:
                try:
                    self.logger.info(f"Generating {model_type.upper()} model analysis...")
                    results_file = comparisons_path / f"{model_type}_final_results.pkl"
                    charts_dir = comparisons_path / "charts"
                    
                    if results_file.exists():
                        evaluator.plot_results_comparison(results_file, charts_dir, model_type)
                        evaluator.generate_radar_plots(model_type)
                        evaluation_results[f'{model_type}_analysis'] = True
                        self.logger.info(f"{model_type.upper()} model analysis completed")
                    else:
                        self.logger.warning(f"{model_type.upper()} results file not found: {results_file}")
                        
                except Exception as e:
                    self.logger.error(f"Error in {model_type.upper()} analysis: {e}")
                    evaluation_results[f'{model_type}_analysis'] = f"Failed: {str(e)}"
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': f'Evaluation completed for available models',
                'analysis_results': evaluation_results,
                'charts_generated': True
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return evaluation_results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise


    def run_visualize_phase(self):
        """Run overall performance visualization phase with enhanced logging"""
        phase_name = "overall_visualization"
        start_time = self.log_phase_start(phase_name)
        
        try:
            analyzer = OverallPerformanceAnalyzer()
            analyzer.generate_all_comparisons()
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': 'Overall performance visualization completed',
                'output_location': str(RESULTS_CONFIG['overall_results_path'])
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return True
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_benchmark_generation_phase(self):
        """Run benchmark generation phase with enhanced logging"""
        phase_name = "benchmark_generation"
        start_time = self.log_phase_start(phase_name)
        
        try:
            # TODO: Implement benchmark generation
            self.logger.info("Benchmark generation phase - to be implemented")
            self.logger.info("Will generate:")
            self.logger.info("- Performance benchmarks across all models")
            self.logger.info("- Comparison tables and reports")
            self.logger.info("- Model ranking and recommendations")
            
            self.results[phase_name] = {
                'status': 'pending_implementation',
                'summary': 'Benchmark generation phase placeholder'
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return True
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def print_final_summary(self):
        """Print comprehensive pipeline execution summary"""
        print(f"\n{'='*100}")
        print(f"PIPELINE EXECUTION SUMMARY")
        print(f"{'='*100}")
        
        print(f"Start Time: {self.execution_log['start_time']}")
        print(f"Total Execution Time: {self.execution_log['total_execution_time']:.2f} seconds")
        print(f"Phases Completed: {len(self.execution_log['phases_completed'])}")
        print(f"Phases Failed: {len(self.execution_log['phases_failed'])}")
        
        if self.execution_log['phases_completed']:
            print(f"\nSUCCESSFUL PHASES:")
            for phase in self.execution_log['phases_completed']:
                duration = self.execution_log['phase_timings'].get(phase, 0)
                print(f"  ✓ {phase}: {duration:.2f}s")
        
        if self.execution_log['phases_failed']:
            print(f"\nFAILED PHASES:")
            for failure in self.execution_log['phases_failed']:
                print(f"  ✗ {failure['phase']}: {failure['error']}")
        
        print(f"\nRESULTS LOCATIONS:")
        print(f"  - ML results: results/ml/comparisons/")
        print(f"  - DL results: results/dl/comparisons/")
        print(f"  - BERT results: results/bert/comparisons/")
        print(f"  - Fusion results: results/fusion/comparisons/")
        print(f"  - DeepSeek results: results/deepseek/comparisons/")
        print(f"  - Overall results: results/overall/")
        print(f"  - Individual category results: results/*/top_*_categories/")
        
        print(f"{'='*100}")

    def run_ml_explainability_phase(self):
        """Run ML explainability analysis phase with SHAP and LIME"""
        phase_name = "ml_explainability"
        start_time = self.log_phase_start(phase_name)
        
        try:
            self.logger.info("Starting ML Model Explainability Analysis (SHAP & LIME)")
            
            explainer = MLExplainability()
            
            # Run explainability for first category size (or all if needed)
            n_categories = CATEGORY_SIZES[0]
            feature_types = ["tfidf", "sbert"]
            
            self.logger.info(f"Analyzing top_{n_categories}_categories")
            self.logger.info(f"Feature types: {feature_types}")
            
            results = explainer.explain_all_models(n_categories, feature_types)
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': 'ML explainability analysis completed successfully',
                'n_categories': n_categories,
                'feature_types': feature_types,
                'models_analyzed': list(results.get('tfidf', {}).keys()) if 'tfidf' in results else []
            }
            
            self.log_phase_end(phase_name, start_time, success=True)
            return results
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise

    def run_dl_explainability_phase(self):
            """Execute Deep Learning Explainability Phase"""
            
            
            self.logger.info("="*80)
            self.logger.info("STARTING PHASE: DL_EXPLAINABILITY")
            self.logger.info("="*80)
            
            try:
                # Initialize the explainer
                explainer = DLExplainability()
                
                # Iterate over all category sizes (e.g., 5, 10) defined in config
                for n_categories in CATEGORY_SIZES:
                    self.logger.info(f"Generating DL explanations for top_{n_categories}_categories...")
                    # This will automatically run for all models (BiLSTM) and features (TFIDF, SBERT)
                    explainer.explain_all_models(n_categories)
                    
                self.logger.info("DL Explainability phase completed successfully.")
                
            except Exception as e:
                self.logger.error(f"DL Explainability phase failed: {e}")
                raise e

    # --- NEW: BERT EXPLAINABILITY PHASE ---
    def run_bert_explainability_phase(self):
        """Execute BERT Explainability Phase"""
        
        
        phase_name = "bert_explainability"
        start_time = self.log_phase_start(phase_name)
        
        try:
            for n_categories in CATEGORY_SIZES:
                self.logger.info(f"Generating BERT explanations for top_{n_categories}_categories...")
                explainer = BERTExplainability(n_categories)
                explainer.explain_all_models()
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': 'BERT explainability analysis completed successfully'
            }
            self.log_phase_end(phase_name, start_time, success=True)
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise e

    # --- NEW: FUSION EXPLAINABILITY PHASE ---
    def run_fusion_explainability_phase(self):
        """Execute Fusion Explainability Phase"""
        
        
        phase_name = "fusion_explainability"
        start_time = self.log_phase_start(phase_name)
        
        try:
            # Define fusion types to analyze
            fusion_types = ['concat', 'average', 'weighted', 'gating']
            
            for n_categories in CATEGORY_SIZES:
                self.logger.info(f"Generating Fusion explanations for top_{n_categories}_categories...")
                self.logger.info(f"Analyzing fusion types: {fusion_types}")
                
                explainer = FusionExplainability(
                    n_categories=n_categories,
                    fusion_types=fusion_types
                )
                explainer.explain_all_models()
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': 'Fusion explainability analysis completed successfully',
                'fusion_types': fusion_types,
                'categories': CATEGORY_SIZES
            }
            self.log_phase_end(phase_name, start_time, success=True)
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise e

    # --- NEW: DEEPSEEK EXPLAINABILITY PHASE ---
    def run_deepseek_explainability_phase(self):
        """Execute DeepSeek Explainability Phase"""
        
        
        phase_name = "deepseek_explainability"
        start_time = self.log_phase_start(phase_name)
        
        try:
            for n_categories in CATEGORY_SIZES:
                self.logger.info(f"Generating DeepSeek explanations for top_{n_categories}_categories...")
                
                explainer = DeepSeekExplainability(n_categories=n_categories)
                explainer.explain()
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': 'DeepSeek explainability analysis completed successfully',
                'categories': CATEGORY_SIZES
            }
            self.log_phase_end(phase_name, start_time, success=True)
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise e
    def run_overall_explainability_phase(self):
        """Run the overall XAI comparison chart generation"""
        phase_name = "overall_explainability"
        start_time = self.log_phase_start(phase_name)
        
        try:
            for n_categories in CATEGORY_SIZES:
                self.logger.info(f"Generating Overall XAI Comparison for top_{n_categories}_categories...")
                generate_overall_charts(n_categories)
            
            self.results[phase_name] = {
                'status': 'completed',
                'summary': 'Overall explainability charts generated successfully',
                'categories': CATEGORY_SIZES
            }
            self.log_phase_end(phase_name, start_time, success=True)
            
        except Exception as e:
            self.results[phase_name] = {
                'status': 'failed',
                'error': str(e)
            }
            self.log_phase_end(phase_name, start_time, success=False, error=e)
            raise e

def main():
    """Enhanced main function with comprehensive pipeline management"""
    parser = argparse.ArgumentParser(description="Web Services Classification Pipeline")
    parser.add_argument(
        "--phase", 
        choices=[
            "all", "analysis", "preprocessing", "features", "ml_training", 
            "dl_training", "bert_training", "fusion_training", "deepseek_training", 
            "evaluation", "visualize", "benchmarks", 
            "ml_explainability", "dl_explainability", "bert_explainability", 
            "fusion_explainability", "deepseek_explainability","overall_explainability"
        ],
        default="all",
        help="Which phase to run"
    )
    parser.add_argument("--categories", type=int, nargs="+", help="Specific category sizes to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Initialize pipeline manager
    pipeline = PipelineManager()
    pipeline.setup_project_logging("main")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        pipeline.logger.info("Starting Web Services Classification Pipeline")
        pipeline.logger.info(f"Phase: {args.phase}")
        
        # Override category sizes if specified
        if args.categories:
            global CATEGORY_SIZES
            CATEGORY_SIZES = args.categories
            pipeline.logger.info(f"Using custom category sizes: {CATEGORY_SIZES}")
        
        # Run specified phase(s)
        if args.phase == "all":
            print("Running complete pipeline...")
            pipeline.run_data_analysis_phase()
            pipeline.run_preprocessing_phase()
            pipeline.run_feature_extraction_phase()
            pipeline.run_ml_training_phase()
            pipeline.run_dl_training_phase()
            pipeline.run_bert_training_phase()
            pipeline.run_fusion_training_phase()
            pipeline.run_deepseek_training_phase()
            pipeline.run_evaluation_phase()
            pipeline.run_ml_explainability_phase()
            pipeline.run_dl_explainability_phase()
            pipeline.run_bert_explainability_phase()
            pipeline.run_fusion_explainability_phase()
            pipeline.run_deepseek_explainability_phase()
            pipeline.run_overall_explainability_phase()
            pipeline.run_visualize_phase()
            pipeline.run_benchmark_generation_phase()
        elif args.phase == "analysis":
            pipeline.run_data_analysis_phase()
            #pipeline.run_enhanced_data_analysis_phase()
        elif args.phase == "preprocessing":
            pipeline.run_preprocessing_phase()
        elif args.phase == "features":
            pipeline.run_feature_extraction_phase()
        elif args.phase == "ml_training":
            pipeline.run_ml_training_phase()
        elif args.phase == "dl_training":
            pipeline.run_dl_training_phase()
        elif args.phase == "bert_training":
            pipeline.run_bert_training_phase()
        elif args.phase == "fusion_training":
            pipeline.run_fusion_training_phase()
        elif args.phase == "deepseek_training":            
            pipeline.run_deepseek_training_phase()
        elif args.phase == "evaluation":
            pipeline.run_evaluation_phase()
        elif args.phase == "visualize":
            pipeline.run_visualize_phase()
        elif args.phase == "benchmarks":
            pipeline.run_benchmark_generation_phase()
        elif args.phase == "ml_explainability":
            pipeline.run_ml_explainability_phase()
        elif args.phase == "dl_explainability":
            pipeline.run_dl_explainability_phase()
        elif args.phase == "bert_explainability":
            pipeline.run_bert_explainability_phase()
        elif args.phase == "fusion_explainability":
            pipeline.run_fusion_explainability_phase()
        elif args.phase == "deepseek_explainability":
            pipeline.run_deepseek_explainability_phase()
        elif args.phase == "overall_explainability":  
            pipeline.run_overall_explainability_phase()
        
        # Save execution summary and print final results
        pipeline.save_execution_summary()
        pipeline.print_final_summary()
        
        pipeline.logger.info("Pipeline completed successfully")
        
    except Exception as e:
        pipeline.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        pipeline.save_execution_summary()  # Save summary even on failure
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
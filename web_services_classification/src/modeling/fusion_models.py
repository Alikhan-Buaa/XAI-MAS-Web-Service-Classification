"""
DeepSeek-RoBERTa Fusion Models for Web Service Classification
Combines embeddings from DeepSeek and RoBERTa with multiple fusion strategies
BASE MODELS ARE FROZEN - Only fusion layers and classifier are trained
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel

# Import configuration and utilities
from src.config import (
    CATEGORY_SIZES, SAVED_MODELS_CONFIG, FUSION_CONFIG,
    PREPROCESSING_CONFIG, RANDOM_SEED, RESULTS_CONFIG
)
from src.evaluation.evaluate import ModelEvaluator
from src.utils.utils import FileNamingStandard

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================================
# DEEPSEEK-ROBERTA FUSION MODEL ARCHITECTURE (FROZEN BASE MODELS)
# ============================================================================

class DeepSeekRoBERTaFusionModel(nn.Module):
    """
    Fusion model combining DeepSeek and RoBERTa embeddings
    Supports: concatenation, averaging, weighted, and gating fusion
    BASE MODELS ARE FROZEN - Only fusion + classifier layers are trained
    """
    
    def __init__(self, config, num_labels):
        super(DeepSeekRoBERTaFusionModel, self).__init__()
        
        self.config = config
        self.num_labels = num_labels
        self.fusion_type = config.get('fusion_type', 'concat')
        dropout = config.get('dropout', 0.3)
        
        # Load DeepSeek model
        deepseek_model_name = config.get('deepseek_model', 'deepseek-ai/deepseek-llm-7b-base')
        logger.info(f"Loading DeepSeek model: {deepseek_model_name}")
        self.deepseek = AutoModel.from_pretrained(
            deepseek_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.deepseek_hidden_size = self.deepseek.config.hidden_size
        
        # Load RoBERTa model
        roberta_model_name = config.get('roberta_model', 'roberta-base')
        logger.info(f"Loading RoBERTa model: {roberta_model_name}")
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size
        
        # ============================================================
        # FREEZE BASE MODELS - Only train fusion + classifier layers
        # ============================================================
        logger.info("Freezing DeepSeek base model...")
        for param in self.deepseek.parameters():
            param.requires_grad = False
        
        logger.info("Freezing RoBERTa base model...")
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # Set models to eval mode to disable dropout in base models
        self.deepseek.eval()
        self.roberta.eval()
        
        logger.info("✓ Base models frozen - only fusion layers and classifier will be trained")
        # ============================================================
        
        # Projection layers to common dimension if sizes differ
        self.common_dim = config.get('common_dim', 768)
        
        if self.deepseek_hidden_size != self.common_dim:
            self.deepseek_proj = nn.Linear(self.deepseek_hidden_size, self.common_dim)
            logger.info(f"DeepSeek projection: {self.deepseek_hidden_size} -> {self.common_dim}")
        else:
            self.deepseek_proj = nn.Identity()
        
        if self.roberta_hidden_size != self.common_dim:
            self.roberta_proj = nn.Linear(self.roberta_hidden_size, self.common_dim)
            logger.info(f"RoBERTa projection: {self.roberta_hidden_size} -> {self.common_dim}")
        else:
            self.roberta_proj = nn.Identity()
        
        # Determine fused dimension based on fusion type
        if self.fusion_type == 'concat':
            fused_dim = self.common_dim * 2
        elif self.fusion_type == 'average':
            fused_dim = self.common_dim
        elif self.fusion_type == 'weighted':
            # Learnable weights for each model
            self.alpha = nn.Parameter(torch.tensor(0.5))
            fused_dim = self.common_dim
        elif self.fusion_type == 'gating':
            # Gating network
            self.gate = nn.Sequential(
                nn.Linear(self.common_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, self.common_dim),
                nn.Sigmoid()
            )
            fused_dim = self.common_dim
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Count trainable vs frozen parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params
        
        logger.info(f"Fusion type: {self.fusion_type}, Fused dimension: {fused_dim}")
        logger.info(f"Parameter breakdown:")
        logger.info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logger.info(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        logger.info(f"  Total: {total_params:,}")
    
    def extract_deepseek_embedding(self, input_ids, attention_mask):
        """Extract DeepSeek embedding with mean pooling"""
        # Use torch.no_grad() since base model is frozen - no gradient computation needed
        with torch.no_grad():
            outputs = self.deepseek(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            # Mean pooling over sequence length
            last_hidden_state = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        # Project to common dimension (this is trainable)
        projected = self.deepseek_proj(pooled.float())
        
        return projected
    
    def extract_roberta_embedding(self, input_ids, attention_mask):
        """Extract RoBERTa embedding (CLS token)"""
        # Use torch.no_grad() since base model is frozen - no gradient computation needed
        with torch.no_grad():
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            # Use CLS token
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to common dimension (this is trainable)
        projected = self.roberta_proj(cls_embedding)
        
        return projected
    
    def fuse_embeddings(self, deepseek_emb, roberta_emb):
        """Fuse embeddings based on fusion type"""
        if self.fusion_type == 'concat':
            # Simple concatenation
            return torch.cat([deepseek_emb, roberta_emb], dim=1)
        
        elif self.fusion_type == 'average':
            # Simple average
            return (deepseek_emb + roberta_emb) / 2.0
        
        elif self.fusion_type == 'weighted':
            # Learnable weighted combination
            alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
            return alpha * deepseek_emb + (1 - alpha) * roberta_emb
        
        elif self.fusion_type == 'gating':
            # Gated fusion - context-dependent weighting
            concat = torch.cat([deepseek_emb, roberta_emb], dim=1)
            gate = self.gate(concat)  # Shape: (batch_size, common_dim)
            return gate * deepseek_emb + (1 - gate) * roberta_emb
    
    def forward(self, deepseek_input_ids, deepseek_attention_mask, 
                roberta_input_ids, roberta_attention_mask, apply_temperature=False):
        """Forward pass through fusion model"""
        # Extract embeddings from both models (frozen - no gradients)
        deepseek_emb = self.extract_deepseek_embedding(deepseek_input_ids, deepseek_attention_mask)
        roberta_emb = self.extract_roberta_embedding(roberta_input_ids, roberta_attention_mask)
        
        # Fuse embeddings (trainable)
        fused_emb = self.fuse_embeddings(deepseek_emb, roberta_emb)
        
        # Classification (trainable)
        logits = self.classifier(fused_emb)
        
        if apply_temperature:
            logits = logits / self.temperature
        
        return logits
    
    def train(self, mode=True):
        """Override train to keep base models in eval mode"""
        super().train(mode)
        # Always keep base models in eval mode
        self.deepseek.eval()
        self.roberta.eval()
        return self


# ============================================================================
# DATASET
# ============================================================================

class DeepSeekRoBERTaFusionDataset(Dataset):
    """Dataset for DeepSeek-RoBERTa fusion model"""
    
    def __init__(self, texts, labels, deepseek_tokenizer, roberta_tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.deepseek_tokenizer = deepseek_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize for DeepSeek
        deepseek_encoding = self.deepseek_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize for RoBERTa
        roberta_encoding = self.roberta_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'deepseek_input_ids': deepseek_encoding['input_ids'].squeeze(0),
            'deepseek_attention_mask': deepseek_encoding['attention_mask'].squeeze(0),
            'roberta_input_ids': roberta_encoding['input_ids'].squeeze(0),
            'roberta_attention_mask': roberta_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# TRAINER
# ============================================================================

class DeepSeekRoBERTaFusionTrainer:
    """Trainer for DeepSeek-RoBERTa Fusion models with frozen base models"""
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert numpy types and Path objects to native Python types"""
        if isinstance(obj, dict):
            return {key: DeepSeekRoBERTaFusionTrainer.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DeepSeekRoBERTaFusionTrainer.make_json_serializable(item) for item in obj]
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
        self.deepseek_tokenizer = None
        self.roberta_tokenizer = None
        self.model = None
        
        # Use FUSION_CONFIG from config.py
        self.config = FUSION_CONFIG.copy()
        
        # Get model_type from config
        self.model_type = self.config.get('model_type', 'fusion')
        
        self.evaluator = ModelEvaluator()
        
        # Configure GPU
        self._configure_gpu()
        
        # Create directories
        self._create_directories()
    
    def _configure_gpu(self):
        """Configure GPU"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                self.scaler = torch.cuda.amp.GradScaler()
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                logger.info("Using CPU")
                self.scaler = None
        except Exception as e:
            logger.warning(f"GPU configuration warning: {e}")
            self.device = torch.device("cpu")
            self.scaler = None
    
    def _create_directories(self):
        """Create result directories"""
        directories = [
            RESULTS_CONFIG['fusion_results_path'],
            RESULTS_CONFIG['fusion_comparisons_path'],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create category-specific directories
        for n_categories in CATEGORY_SIZES:
            category_dir = RESULTS_CONFIG['fusion_category_paths'][n_categories]
            category_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created DeepSeek-RoBERTa Fusion result directories")
    
    def get_model_config(self, fusion_type):
        """Get fusion-specific configuration"""
        model_config = self.config.copy()
        model_config['fusion_type'] = fusion_type
        return model_config
    
    def load_tokenizers(self):
        """Load both tokenizers"""
        try:
            deepseek_model = self.config.get('deepseek_model', 'deepseek-ai/deepseek-llm-7b-base')
            roberta_model = self.config.get('roberta_model', 'roberta-base')
            
            self.deepseek_tokenizer = AutoTokenizer.from_pretrained(
                deepseek_model,
                trust_remote_code=True
            )
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
            
            logger.info(f"Loaded tokenizers: DeepSeek={deepseek_model}, RoBERTa={roberta_model}")
        except Exception as e:
            logger.error(f"Error loading tokenizers: {e}")
            raise
    
    def prepare_datasets(self, n_categories):
        """Prepare datasets"""
        try:
            logger.info(f"Loading datasets for top_{n_categories}_categories")
            
            splits_dir = Path(PREPROCESSING_CONFIG["splits"].format(n=n_categories))
            if not splits_dir.exists():
                raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
            
            train_df = pd.read_csv(splits_dir / 'train.csv')
            val_df = pd.read_csv(splits_dir / 'val.csv')
            test_df = pd.read_csv(splits_dir / 'test.csv')
            
            logger.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            text_column = 'cleaned_text' if 'cleaned_text' in train_df.columns else 'text'
            if text_column not in train_df.columns:
                text_column = 'Service Description' if 'Service Description' in train_df.columns else train_df.columns[0]
            
            logger.info(f"Using text column: {text_column}")
            
            max_length = self.config.get('max_length', 128)
            
            train_dataset = DeepSeekRoBERTaFusionDataset(
                train_df[text_column].astype(str).tolist(),
                train_df['encoded_label'].tolist(),
                self.deepseek_tokenizer,
                self.roberta_tokenizer,
                max_length
            )
            
            val_dataset = DeepSeekRoBERTaFusionDataset(
                val_df[text_column].astype(str).tolist(),
                val_df['encoded_label'].tolist(),
                self.deepseek_tokenizer,
                self.roberta_tokenizer,
                max_length
            )
            
            test_dataset = DeepSeekRoBERTaFusionDataset(
                test_df[text_column].astype(str).tolist(),
                test_df['encoded_label'].tolist(),
                self.deepseek_tokenizer,
                self.roberta_tokenizer,
                max_length
            )
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error preparing datasets: {e}")
            raise
    
    def create_model(self, num_labels, fusion_config):
        """Create fusion model"""
        try:
            self.model = DeepSeekRoBERTaFusionModel(fusion_config, num_labels)
            fusion_type = fusion_config.get('fusion_type', 'concat')
            logger.info(f"Created DeepSeek-RoBERTa Fusion model: {fusion_type}, labels={num_labels}")
            return self.model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train one epoch"""
        model.train()  # This will keep base models in eval mode due to override
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(dataloader):
            deepseek_input_ids = batch['deepseek_input_ids'].to(self.device)
            deepseek_attention_mask = batch['deepseek_attention_mask'].to(self.device)
            roberta_input_ids = batch['roberta_input_ids'].to(self.device)
            roberta_attention_mask = batch['roberta_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(deepseek_input_ids, deepseek_attention_mask,
                                 roberta_input_ids, roberta_attention_mask)
                    loss = criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('gradient_clip', 1.0))
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = model(deepseek_input_ids, deepseek_attention_mask,
                             roberta_input_ids, roberta_attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('gradient_clip', 1.0))
                optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate_epoch(self, model, dataloader, criterion):
        """Evaluate one epoch"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                deepseek_input_ids = batch['deepseek_input_ids'].to(self.device)
                deepseek_attention_mask = batch['deepseek_attention_mask'].to(self.device)
                roberta_input_ids = batch['roberta_input_ids'].to(self.device)
                roberta_attention_mask = batch['roberta_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(deepseek_input_ids, deepseek_attention_mask,
                                     roberta_input_ids, roberta_attention_mask)
                        loss = criterion(logits, labels)
                else:
                    logits = model(deepseek_input_ids, deepseek_attention_mask,
                                 roberta_input_ids, roberta_attention_mask)
                    loss = criterion(logits, labels)
                
                probs = F.softmax(logits, dim=1)
                
                total_loss += loss.item()
                all_probs.append(probs.cpu())
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        all_probs = torch.cat(all_probs, dim=0).numpy()
        
        return avg_loss, accuracy, all_preds, all_labels, all_probs
    
    def calibrate_temperature(self, model, val_loader):
        """Temperature scaling calibration"""
        logger.info("Calibrating temperature...")
        model.eval()
        
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                deepseek_input_ids = batch['deepseek_input_ids'].to(self.device)
                deepseek_attention_mask = batch['deepseek_attention_mask'].to(self.device)
                roberta_input_ids = batch['roberta_input_ids'].to(self.device)
                roberta_attention_mask = batch['roberta_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = model(deepseek_input_ids, deepseek_attention_mask,
                             roberta_input_ids, roberta_attention_mask, apply_temperature=False)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits / model.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        logger.info(f"Optimal temperature: {model.temperature.item():.4f}")
    
    def plot_training_history(self, history, model_name, n_categories):
        """Create training history plots"""
        try:
            if not history['train_loss'] or not history['val_loss']:
                logger.warning("Insufficient training history for plotting")
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Plot loss
            ax1.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2, marker='o')
            ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
            ax1.set_title(f'{model_name} - Training & Validation Loss\n{n_categories} Categories')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracy
            if history['val_acc']:
                ax2.plot(epochs, history['train_acc'], label='Training Accuracy', linewidth=2, marker='o')
                ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2, marker='s')
                ax2.set_title(f'{model_name} - Training & Validation Accuracy\n{n_categories} Categories')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_dir = RESULTS_CONFIG['fusion_category_paths'][n_categories]
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            filename = FileNamingStandard.generate_training_history_filename(model_name, n_categories)
            plot_file = plot_dir / filename
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training history plot saved: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Error creating training history plot: {e}")
            return None
    
    def evaluate_fusion_model(self, model, test_loader, model_name, n_categories, class_labels,fusion_type):
        """Comprehensive evaluation"""
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            start_time = time.time()
            
            with torch.no_grad():
                for batch in test_loader:
                    deepseek_input_ids = batch['deepseek_input_ids'].to(self.device)
                    deepseek_attention_mask = batch['deepseek_attention_mask'].to(self.device)
                    roberta_input_ids = batch['roberta_input_ids'].to(self.device)
                    roberta_attention_mask = batch['roberta_attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    logits = model(deepseek_input_ids, deepseek_attention_mask,
                                 roberta_input_ids, roberta_attention_mask, apply_temperature=True)
                    probs = F.softmax(logits, dim=1)
                    
                    all_probs.append(probs.cpu())
                    preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            inference_time = time.time() - start_time
            
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            y_proba = torch.cat(all_probs, dim=0).numpy()
            
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
            
            # Top-K accuracies
            y_true_onehot = np.eye(n_categories)[y_true]
            top1_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=1)
            top3_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=3)
            top5_accuracy = self.evaluator.calculate_top_k_accuracy(y_true_onehot, y_proba, k=5)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Standardize fusion_type to feature type format (Concat, Average, Weighted, Gating)
            # fusion_type comes as parameter from train_model_on_category()
            from src.config import FEATURE_NAME_MAPPING
            feature_type = FEATURE_NAME_MAPPING.get(fusion_type.lower(), fusion_type.capitalize())
            
            # Use self.model_type from config
            cm_plot_path = self.evaluator.generate_confusion_heatmap(
                cm, class_labels, model_name, n_categories, 
                feature_type, 
                self.model_type
            )
            
            report_path = self.evaluator.generate_classification_report_csv(
                y_true, y_pred, class_labels, model_name, n_categories, 
                feature_type, 
                self.model_type
            )
            
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
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def train_model_on_category(self, n_categories, fusion_type='concat'):
        """Train DeepSeek-RoBERTa fusion model"""
        try:
            model_name = f"DeepSeek-RoBERTa-Fusion-{fusion_type.capitalize()}"
            logger.info(f"Training {model_name} for top_{n_categories}_categories")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get model config
            model_config = self.get_model_config(fusion_type)
            
            # Load tokenizers
            self.load_tokenizers()
            
            # Prepare datasets
            train_dataset, val_dataset, test_dataset = self.prepare_datasets(n_categories)
            
            # Load class labels
            class_labels = self.evaluator.load_class_labels(n_categories)
            
            # Create model
            model = self.create_model(n_categories, model_config).to(self.device)
            
            # Training setup - can use larger batch sizes now that base models are frozen
            batch_size = self.config.get('batch_size', 16)  # Increased from 8
            eval_batch_size = self.config.get('eval_batch_size', 32)  # Increased from 16
            learning_rate = self.config.get('learning_rate', 2e-4)  # Higher LR for frozen base
            weight_decay = self.config.get('weight_decay', 0.01)
            epochs = self.config.get('num_train_epochs', 10)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            
            # Only optimize trainable parameters (fusion + classifier)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            scheduler_config = self.config.get('scheduler', {})
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'max'),
                patience=scheduler_config.get('patience', 2),
                factor=scheduler_config.get('factor', 0.5)
            )
            
            # Training history
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            best_val_acc = 0.0
            best_model_state = None
            
            print(f"\n{'='*80}")
            print(f"Training {model_name} on top_{n_categories}_categories")
            print(f"{'='*80}")
            print(f"Fusion type: {fusion_type}")
            print(f"Base models: FROZEN (only training fusion + classifier layers)")
            print(f"Batch size: Train={batch_size}, Eval={eval_batch_size}")
            print(f"Learning rate: {learning_rate}")
            print(f"Epochs: {epochs}")
            print(f"{'='*80}\n")
            
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, _, _, _ = self.evaluate_epoch(model, val_loader, criterion)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_acc)
                current_lr = optimizer.param_groups[0]['lr']
                
                epoch_time = time.time() - epoch_start
                
                print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"  LR: {current_lr:.2e}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    print(f"  ✓ Best model (Val Acc: {val_acc:.4f})")
                print()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Save model
            model_dir = SAVED_MODELS_CONFIG['fusion_models_path'] / f'top_{n_categories}_categories'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # MODIFIED: Simplified feature_type for model filename
            model_filename = FileNamingStandard.generate_model_filename(
                model_name, fusion_type, n_categories, 'model'
            )
            model_path = model_dir / model_filename
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'fusion_type': fusion_type,
                'n_categories': n_categories,
                'config': model_config,
                'best_val_acc': best_val_acc
            }, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Calibrate temperature
            self.calibrate_temperature(model, val_loader)
            
            # Create training history plot
            display_name = FileNamingStandard.standardize_model_name(model_name)
            history_plot_path = self.plot_training_history(history, display_name, n_categories)
            
            # Evaluate model
            eval_results = self.evaluate_fusion_model(model, test_loader, display_name, n_categories, class_labels,fusion_type)
            eval_results['training_time'] = float(training_time)
            eval_results['model_path'] = str(model_path)
            eval_results['model_variant'] = fusion_type
            eval_results['training_history_plot'] = history_plot_path
            eval_results['batch_size'] = batch_size
            eval_results['learning_rate'] = learning_rate
            eval_results['best_val_acc'] = float(best_val_acc)
            
            # MODIFIED: Simplified feature_type for metrics printing
            self.evaluator.print_model_metrics(
                eval_results, display_name, n_categories, 
                fusion_type, training_time, "DeepSeek-RoBERTa Fusion"
            )
            
            # MODIFIED: Simplified feature_type and model_type for saving
            self.evaluator.save_model_performance_data(
                eval_results, display_name, n_categories, 
                fusion_type, "fusion"
            )
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error training {fusion_type} fusion for {n_categories} categories: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def save_results_for_overall_analysis(self, all_results):
        """Save results for overall analysis - compatible with ML/DL/BERT format"""
        try:
            comparisons_path = RESULTS_CONFIG['fusion_comparisons_path']
            comparisons_path.mkdir(parents=True, exist_ok=True)
            
            # Transform results to match ML/DL/BERT format (list-based structure)
            formatted_results = {}
            
            for fusion_type, fusion_results in all_results.items():
                for n_categories, result in fusion_results.items():
                    if n_categories not in formatted_results:
                        formatted_results[n_categories] = {}  # Use list like other models
                    
                    # Create result entry with 'model' key for compatibility
                    model_name = result.get('model_name', f'DeepSeek_RoBERTa_Fusion_{fusion_type.capitalize()}')
                    clean_model_name = FileNamingStandard.standardize_model_name(model_name)
                    feature_type = result.get('feature_type', fusion_type)
                    result_key = f"{clean_model_name}_{feature_type}"
                    
                    formatted_results[n_categories][result_key] = result
            
            # Save as pickle - same format as ML/DL/BERT
            pickle_file = comparisons_path / "fusion_final_results.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(formatted_results, f)
            
            logger.info(f"DeepSeek-RoBERTa Fusion results saved: {pickle_file}")
            
            # Save JSON for debugging
            json_file = comparisons_path / "fusion_final_results.json"
            with open(json_file, 'w') as f:
                json.dump(self.make_json_serializable(formatted_results), f, indent=2)
            
            logger.info(f"DeepSeek-RoBERTa Fusion results JSON saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def train_fusion_models(self, categories=None):
        """Train all DeepSeek-RoBERTa Fusion models"""
        if categories is None:
            categories = CATEGORY_SIZES
        
        logger.info("Training DeepSeek-RoBERTa Fusion models")
        
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"STARTING DEEPSEEK-ROBERTA FUSION MODEL TRAINING PIPELINE")
        print(f"BASE MODELS FROZEN - Only training fusion + classifier layers")
        print(f"{'='*80}")
        print(f"Category sizes: {categories}")
        print(f"Fusion types: {self.config.get('fusion_types', ['concat', 'average', 'weighted', 'gating'])}")
        print(f"{'='*80}\n")
        
        for fusion_type in self.config.get('fusion_types', ['concat', 'average', 'weighted', 'gating']):
            print(f"\n{'-'*60}")
            print(f"TRAINING DEEPSEEK-ROBERTA-FUSION-{fusion_type.upper()}")
            print(f"{'-'*60}")
            
            fusion_results = {}
            
            for n_categories in categories:
                print(f"\n>>> Processing top_{n_categories}_categories with {fusion_type} fusion...")
                
                try:
                    results = self.train_model_on_category(n_categories, fusion_type)
                    fusion_results[n_categories] = results
                    
                    # Save individual results
                    category_dir = SAVED_MODELS_CONFIG['fusion_models_path'] / f'top_{n_categories}_categories'
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    results_json = category_dir / f'deepseek_roberta_fusion_{fusion_type}_results.json'
                    with open(results_json, 'w') as f:
                        json.dump(self.make_json_serializable(results), f, indent=2)
                    
                    logger.info(f"Results saved to {results_json}")
                    
                except Exception as e:
                    logger.error(f"Error training {fusion_type} for {n_categories} categories: {e}")
                    continue
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_results[fusion_type] = fusion_results
        
        print(f"\n{'='*80}")
        print(f"DEEPSEEK-ROBERTA FUSION MODEL TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")

        # Print comparison
        if len(all_results) > 1:
            self._print_fusion_comparison(all_results)

        # Save results for overall analysis
        self.save_results_for_overall_analysis(all_results)

        # Generate visualizations
        print(f"\n{'='*80}")
        print(f"GENERATING FUSION VISUALIZATIONS")
        print(f"{'='*80}")
        try:
            print("Generating comparison plots...")
            self.plot_fusion_results_only()
            print("Generating radar plots...")
            self.evaluator.generate_radar_plots("fusion", show_plots=False)
            print("All Fusion visualizations completed successfully!")
        except Exception as e:
            logger.error(f"Error generating Fusion visualizations: {e}")
            import traceback
            traceback.print_exc()
            print(f"Warning: Some visualizations may not have been generated due to errors.")

        return all_results
    
    def _print_fusion_comparison(self, all_results):
        """Print comparison between fusion models"""
        print(f"\n{'='*80}")
        print(f"DEEPSEEK-ROBERTA FUSION MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        for n_categories in CATEGORY_SIZES:
            results_for_category = {}
            
            for fusion_key, fusion_results in all_results.items():
                if n_categories in fusion_results:
                    results_for_category[fusion_key] = fusion_results[n_categories]
            
            if results_for_category:
                print(f"\nTop {n_categories} Categories Results:")
                print(f"{'Fusion Type':<15} {'Top-1 Acc':<10} {'Top-3 Acc':<10} {'Top-5 Acc':<10} {'Macro F1':<10} {'Training Time':<15}")
                print("-" * 85)
                
                fusion_scores = []
                for fusion_key, result in results_for_category.items():
                    fusion_scores.append((
                        fusion_key,
                        result['top1_accuracy'],
                        result['top3_accuracy'],
                        result['top5_accuracy'],
                        result['macro_f1'],
                        result['training_time']
                    ))
                
                # Sort by macro F1 score
                fusion_scores.sort(key=lambda x: x[4], reverse=True)
                
                for fusion_type, top1, top3, top5, f1, time_taken in fusion_scores:
                    print(f"{fusion_type:<15} {top1:<10.4f} {top3:<10.4f} {top5:<10.4f} {f1:<10.4f} {time_taken:<15.2f}")
                
                if len(fusion_scores) >= 2:
                    best_f1 = fusion_scores[0][4]
                    worst_f1 = fusion_scores[-1][4]
                    improvement = best_f1 - worst_f1
                    
                    print(f"\nPerformance Analysis:")
                    print(f"  Best fusion: {fusion_scores[0][0]} (F1: {best_f1:.4f})")
                    print(f"  F1 improvement over worst: {improvement:+.4f}")
                    print(f"  Relative improvement: {(improvement/worst_f1)*100:+.2f}%")
        
        print(f"{'='*80}")
        
    
    def train_all_categories(self):
        """Train all fusion models on all categories"""
        self.train_fusion_models()
        # Generate visualizations
        print(f"\n{'='*80}")
        print(f"GENERATING FUSION VISUALIZATIONS")
        print(f"{'='*80}")
        try:
            print("Generating line plots, bar plots, and summary statistics...")
            self.plot_fusion_results_only()
            print("Generating radar plots...")
            self.generate_fusion_radar_plots_only()
            print("All Fusion visualizations completed successfully!")
        except Exception as e:
            logger.error(f"Error generating Fusion visualizations: {e}")
            print(f"Warning: Some visualizations may not have been generated due to errors.")
    
    def plot_fusion_results_only(self):
        """Generate Fusion comparison plots"""
        results_file_path = RESULTS_CONFIG["fusion_comparisons_path"] / "fusion_final_results.pkl"
        charts_dir = RESULTS_CONFIG["fusion_comparisons_path"] / "charts"
        self.evaluator.plot_results_comparison(results_file_path, charts_dir, "fusion")

    def generate_fusion_radar_plots_only(self, show_plots=False):
        """Generate Fusion radar plots"""
        self.evaluator.generate_radar_plots("fusion", show_plots)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run DeepSeek-RoBERTa Fusion training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek-RoBERTa Fusion Model Training (Frozen Base Models)")
    parser.add_argument("--fusion-type", type=str, default="all",
                       choices=["concat", "average", "weighted", "gating", "all"],
                       help="Fusion type to train")
    parser.add_argument("--categories", nargs="+", type=int, default=CATEGORY_SIZES,
                       help="Category sizes to train")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"DeepSeek-RoBERTa Fusion Training")
    print(f"BASE MODELS FROZEN - Only training fusion + classifier layers")
    print(f"{'='*80}\n")
    
    trainer = DeepSeekRoBERTaFusionTrainer()
    
    # Override config if provided
    if args.epochs is not None:
        trainer.config['num_train_epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    if args.batch_size is not None:
        trainer.config['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size: {args.batch_size}")
    if args.lr is not None:
        trainer.config['learning_rate'] = args.lr
        logger.info(f"Overriding learning rate: {args.lr}")
    
    if args.fusion_type == "all":
        results = trainer.train_fusion_models(args.categories)
    else:
        logger.info(f"Training single fusion type: {args.fusion_type}")
        results = {}
        for n_categories in args.categories:
            result = trainer.train_model_on_category(n_categories, args.fusion_type)
            results[n_categories] = result
    
    # Save final results
    out_file = SAVED_MODELS_CONFIG["fusion_models_path"] / "fusion_final_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(trainer.make_json_serializable(results), f, indent=2)
    logger.info(f"Results saved to {out_file}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"Results saved to: {out_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
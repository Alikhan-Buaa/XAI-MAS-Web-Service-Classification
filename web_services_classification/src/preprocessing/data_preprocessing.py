"""
Data Preprocessing Module for Web Service Classification
Handles data cleaning, category filtering, and train/test splits
"""

import numpy as np
import pandas as pd
import re
import yaml
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Import configuration
from src.config import DATA_CONFIG, CATEGORY_SIZES, SPLIT_CONFIG, PREPROCESS_PATH, PREPROCESSING_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main class for data preprocessing operations"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        if PREPROCESSING_CONFIG.get('custom_stopwords'):
            self.stop_words.update(PREPROCESSING_CONFIG['custom_stopwords'])
        self.setup_nltk()
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"NLTK download warning: {e}")
    
    def load_data(self, file_path=None):
        """Load the web services dataset"""
        if file_path is None:
            file_path = DATA_CONFIG['raw_data_path']
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            required_cols = [DATA_CONFIG['text_column'], DATA_CONFIG['target_column']]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        if PREPROCESSING_CONFIG['remove_numbers']:
            text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        filtered_tokens = []
        for token in tokens:
            if len(token) < PREPROCESSING_CONFIG['min_word_length'] or len(token) > PREPROCESSING_CONFIG['max_word_length']:
                continue
            if PREPROCESSING_CONFIG['remove_stopwords'] and token in self.stop_words:
                continue
            if token in PREPROCESSING_CONFIG.get('custom_stopwords', []):
                continue
            if PREPROCESSING_CONFIG['lemmatization']:
                token = self.lemmatizer.lemmatize(token)
            filtered_tokens.append(token)
        return ' '.join(filtered_tokens)
    
    def get_top_categories(self, df, n_categories):
        """Get top N categories by frequency"""
        category_counts = df[DATA_CONFIG['target_column']].value_counts()
        top_categories = category_counts.head(n_categories).index.tolist()
        logger.info(f"Top {n_categories} categories selected:")
        for i, (cat, count) in enumerate(category_counts.head(n_categories).items(), 1):
            logger.info(f"  {i}. {cat}: {count} samples")
        return top_categories
    
    def filter_by_categories(self, df, categories):
        """Filter dataframe by specified categories"""
        filtered_df = df[df[DATA_CONFIG['target_column']].isin(categories)].copy()
        filtered_df = filtered_df.reset_index(drop=True)
        logger.info(f"Filtered dataset shape: {filtered_df.shape}")
        return filtered_df
    
    def create_train_val_test_split(self, df, random_state=None):
        """Create train/validation/test splits"""
        if random_state is None:
            random_state = SPLIT_CONFIG['random_state']
        train_val, test = train_test_split(
            df,
            test_size=SPLIT_CONFIG['test_size'],
            random_state=random_state,
            stratify=df[DATA_CONFIG['target_column']] if SPLIT_CONFIG['stratify'] else None
        )
        val_size_adjusted = SPLIT_CONFIG['val_size'] / (1 - SPLIT_CONFIG['test_size'])
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val[DATA_CONFIG['target_column']] if SPLIT_CONFIG['stratify'] else None
        )
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test
    
    def create_label_encoder(self, categories):
        """Create and fit label encoder"""
        encoder = LabelEncoder()
        encoder.fit(categories)
        return encoder
    
    def save_label_mapping(self, categories, n_categories, encoder):
        """Save label mapping YAML"""
        label_mapping = {
            'categories': categories,
            'label_to_id': {label: int(encoder.transform([label])[0]) for label in categories},
            'id_to_label': {int(encoder.transform([label])[0]): label for label in categories},
            'n_categories': n_categories,
            'random_state': SPLIT_CONFIG['random_state']
        }
        labels_file = Path(str(PREPROCESSING_CONFIG['labels']).format(n=n_categories))
        labels_file.parent.mkdir(parents=True, exist_ok=True)
        with open(labels_file, 'w') as f:
            yaml.dump(label_mapping, f, default_flow_style=False)
        logger.info(f"Label mapping saved to {labels_file}")
        return label_mapping
    
    def save_splits(self, train_df, val_df, test_df, n_categories):
        """Save train/val/test splits and metadata"""
        splits_dir = Path(str(PREPROCESSING_CONFIG['splits']).format(n=n_categories))
        splits_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(splits_dir / 'train.csv', index=False)
        val_df.to_csv(splits_dir / 'val.csv', index=False)
        test_df.to_csv(splits_dir / 'test.csv', index=False)
        splits_data = {
            'train_indices': train_df.index.tolist(),
            'val_indices': val_df.index.tolist(),
            'test_indices': test_df.index.tolist(),
            'n_categories': n_categories,
            'split_config': SPLIT_CONFIG,
            'preprocessing_config': PREPROCESSING_CONFIG,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'total_size': len(train_df) + len(val_df) + len(test_df)
        }
        with open(splits_dir / 'split_info.json', 'w') as f:
            json.dump(splits_data, f, indent=2)
        logger.info(f"Splits saved to {splits_dir}")
    
    def save_full_cleaned_dataset(self, filtered_df, n_categories):
        """Save full cleaned dataset"""
        processed_dir = Path(str(PREPROCESSING_CONFIG['processed_data']).format(n=n_categories))
        processed_dir.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(processed_dir / 'cleaned_dataset.csv', index=False)
        logger.info(f"Full cleaned dataset saved to {processed_dir}")
    
    def process_category_size(self, df, n_categories):
        """Process data for a specific category size"""
        logger.info(f"Processing top {n_categories} categories...")
        top_categories = self.get_top_categories(df, n_categories)
        filtered_df = self.filter_by_categories(df, top_categories)
        
        logger.info("Cleaning text data...")
        filtered_df['cleaned_text'] = filtered_df[DATA_CONFIG['text_column']].apply(self.clean_text)
        
        # Encode labels
        encoder = self.create_label_encoder(top_categories)
        filtered_df['encoded_label'] = encoder.transform(filtered_df[DATA_CONFIG['target_column']])
        
        # Save full cleaned dataset
        self.save_full_cleaned_dataset(filtered_df, n_categories)
        
        # Create train/val/test splits
        train_df, val_df, test_df = self.create_train_val_test_split(filtered_df)
        self.save_splits(train_df, val_df, test_df, n_categories)
        
        # Save label mapping
        self.save_label_mapping(top_categories, n_categories, encoder)
        
        return {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'categories': top_categories
        }
    
    def process_all_categories(self):
        """Process all CATEGORY_SIZES"""
        df = self.load_data()
        results = {}
        for n_categories in CATEGORY_SIZES:
            try:
                result = self.process_category_size(df, n_categories)
                results[n_categories] = result
                logger.info(f" Successfully processed top {n_categories} categories")
            except Exception as e:
                logger.error(f" Error processing top {n_categories} categories: {e}")
        logger.info("All preprocessing completed successfully!")
        return results


# Entry point
def main():
    preprocessor = DataPreprocessor()
    preprocessor.process_all_categories()


if __name__ == "__main__":
    main()

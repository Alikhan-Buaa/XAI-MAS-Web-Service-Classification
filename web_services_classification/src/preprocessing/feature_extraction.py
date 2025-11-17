"""
Feature Extraction Pipeline for Web Service Classification
Handles TF-IDF and SBERT embeddings extraction
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Import configuration
from config import DATA_CONFIG, FEATURES_CONFIG, CATEGORY_SIZES, PREPROCESSING_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.sbert_model = None

    def get_features_dir(self, feature_type, n_categories):
        if feature_type == 'tfidf':
            return Path(FEATURES_CONFIG['tfidf_path'].format(n=n_categories))
        elif feature_type == 'sbert':
            return Path(FEATURES_CONFIG['sbert_path'].format(n=n_categories))
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def load_sbert_model(self):
        if self.sbert_model is None:
            logger.info(f"Loading SBERT model: {FEATURES_CONFIG['sbert']['model_name']}")
            self.sbert_model = SentenceTransformer(FEATURES_CONFIG['sbert']['model_name'])
        return self.sbert_model

    def save_tfidf_features(self, train_features, val_features, test_features, n_categories):
        features_dir = self.get_features_dir('tfidf', n_categories)
        features_dir.mkdir(parents=True, exist_ok=True)
        with open(features_dir / 'train_features.pkl', 'wb') as f: pickle.dump(train_features, f)
        with open(features_dir / 'val_features.pkl', 'wb') as f: pickle.dump(val_features, f)
        with open(features_dir / 'test_features.pkl', 'wb') as f: pickle.dump(test_features, f)
        with open(features_dir / 'vectorizer.pkl', 'wb') as f: pickle.dump(self.tfidf_vectorizer, f)
        metadata = {
            'feature_type': 'tfidf',
            'n_categories': n_categories,
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
            'feature_count': train_features.shape[1],
            'config': FEATURES_CONFIG['tfidf']
        }
        with open(features_dir / 'feature_info.json', 'w') as f: json.dump(metadata, f, indent=2)
        logger.info(f"TF-IDF features saved to {features_dir}")

    def save_sbert_features(self, train_features, val_features, test_features, n_categories):
        features_dir = self.get_features_dir('sbert', n_categories)
        features_dir.mkdir(parents=True, exist_ok=True)
        np.save(features_dir / 'train_embeddings.npy', train_features)
        np.save(features_dir / 'val_embeddings.npy', val_features)
        np.save(features_dir / 'test_embeddings.npy', test_features)
        metadata = {
            'feature_type': 'sbert',
            'n_categories': n_categories,
            'embedding_dimension': train_features.shape[1],
            'model_name': FEATURES_CONFIG['sbert']['model_name'],
            'config': FEATURES_CONFIG['sbert']
        }
        with open(features_dir / 'embedding_config.json', 'w') as f: json.dump(metadata, f, indent=2)
        logger.info(f"SBERT features saved to {features_dir}")

    def create_tfidf_features(self, train_texts, val_texts=None, test_texts=None, n_categories=None):
        self.tfidf_vectorizer = TfidfVectorizer(**FEATURES_CONFIG['tfidf'])
        train_features = self.tfidf_vectorizer.fit_transform(train_texts)
        val_features = self.tfidf_vectorizer.transform(val_texts) if val_texts is not None else None
        test_features = self.tfidf_vectorizer.transform(test_texts) if test_texts is not None else None
        if n_categories: self.save_tfidf_features(train_features, val_features, test_features, n_categories)
        return train_features, val_features, test_features

    def create_sbert_features(self, train_texts, val_texts=None, test_texts=None, n_categories=None):
        model = self.load_sbert_model()
        encode_kwargs = {
            'show_progress_bar': FEATURES_CONFIG['sbert']['show_progress_bar'],
            'batch_size': FEATURES_CONFIG['sbert']['batch_size'],
            'convert_to_numpy': True,
            'normalize_embeddings': FEATURES_CONFIG['sbert']['normalize_embeddings']
        }
        train_features = model.encode(train_texts.tolist(), **encode_kwargs)
        val_features = model.encode(val_texts.tolist(), **encode_kwargs) if val_texts is not None else None
        test_features = model.encode(test_texts.tolist(), **encode_kwargs) if test_texts is not None else None
        if n_categories: self.save_sbert_features(train_features, val_features, test_features, n_categories)
        return train_features, val_features, test_features

    def extract_features_for_category(self, n_categories, feature_types=['tfidf','sbert']):
        split_dir = Path(PREPROCESSING_CONFIG['splits'].format(n=n_categories))
        train_df = pd.read_csv(split_dir / 'train.csv')
        val_df = pd.read_csv(split_dir / 'val.csv')
        test_df = pd.read_csv(split_dir / 'test.csv')
        train_texts, val_texts, test_texts = train_df['cleaned_text'], val_df['cleaned_text'], test_df['cleaned_text']
        features = {}
        if 'tfidf' in feature_types:
            features['tfidf'] = {}
            features['tfidf']['train'], features['tfidf']['val'], features['tfidf']['test'] = self.create_tfidf_features(train_texts, val_texts, test_texts, n_categories)
        if 'sbert' in feature_types:
            features['sbert'] = {}
            features['sbert']['train'], features['sbert']['val'], features['sbert']['test'] = self.create_sbert_features(train_texts, val_texts, test_texts, n_categories)
        return features

    def extract_features_all_categories(self, feature_types=['tfidf','sbert']):
        all_features = {}
        for n_categories in CATEGORY_SIZES:
            try:
                features = self.extract_features_for_category(n_categories, feature_types)
                all_features[n_categories] = features
            except Exception as e:
                logger.error(f"Error for top {n_categories} categories: {e}")
        return all_features

    def create_feature_summary(self):
        summary = {'extraction_config': FEATURES_CONFIG, 'categories': {}}
        for n_categories in CATEGORY_SIZES:
            split_dir = Path(PREPROCESSING_CONFIG['splits'].format(n=n_categories))
            train_df = pd.read_csv(split_dir / 'train.csv')
            val_df = pd.read_csv(split_dir / 'val.csv')
            test_df = pd.read_csv(split_dir / 'test.csv')
            summary['categories'][n_categories] = {
                'dataset_sizes': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
                'features': self.get_feature_info(n_categories)
            }
        summary_file = Path(FEATURES_CONFIG['plots']).parent / 'feature_extraction_summary.json'
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f: json.dump(summary, f, indent=2)
        return summary

    def get_feature_info(self, n_categories):
        info = {}
        tfidf_dir = self.get_features_dir('tfidf', n_categories)
        sbert_dir = self.get_features_dir('sbert', n_categories)
        if (tfidf_dir / 'feature_info.json').exists():
            with open(tfidf_dir / 'feature_info.json', 'r') as f: info['tfidf'] = json.load(f)
        if (sbert_dir / 'embedding_config.json').exists():
            with open(sbert_dir / 'embedding_config.json', 'r') as f: info['sbert'] = json.load(f)
        return info

    def validate_features(self):
        validation = {}
        for n_categories in CATEGORY_SIZES:
            tfidf_dir = self.get_features_dir('tfidf', n_categories)
            sbert_dir = self.get_features_dir('sbert', n_categories)
            validation[f'top_{n_categories}'] = {
                'tfidf_files_exist': all([(tfidf_dir / f).exists() for f in ['train_features.pkl','val_features.pkl','test_features.pkl','vectorizer.pkl','feature_info.json']]),
                'sbert_files_exist': all([(sbert_dir / f).exists() for f in ['train_embeddings.npy','val_embeddings.npy','test_embeddings.npy','embedding_config.json']])
            }
        return validation
    

    def load_tfidf_vectorizer(self, n_categories: int):
        """
        Load TF-IDF vectorizer for a given category size.
        """
        tfidf_dir = self.get_features_dir("tfidf", n_categories)
        vectorizer_path = tfidf_dir / "vectorizer.pkl"

        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f" TF-IDF vectorizer not found at {vectorizer_path}. "
                "Run TF-IDF feature extraction first."
            )

        with open(vectorizer_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        logger.info(f" Loaded TF-IDF vectorizer from {vectorizer_path}")
        return self.tfidf_vectorizer

    def load_tfidf_features(self, n_categories: int, split: str):
        """
        Load TF-IDF feature matrices for a given category size and dataset split.
        """
        tfidf_dir = self.get_features_dir("tfidf", n_categories)
        file_map = {
            "train": "train_features.pkl",
            "val": "val_features.pkl",
            "test": "test_features.pkl",
        }

        if split not in file_map:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(file_map.keys())}")

        feature_path = tfidf_dir / file_map[split]

        if not feature_path.exists():
            raise FileNotFoundError(
                f" TF-IDF {split} features not found at {feature_path}. "
                "Run TF-IDF feature extraction first."
            )

        with open(feature_path, "rb") as f:
            features = pickle.load(f)

        logger.info(f" Loaded TF-IDF {split} features from {feature_path}")
        return features

    def load_sbert_features(self, n_categories: int, split: str):
        """
        Load SBERT features for a given category size and dataset split.
        """
        sbert_dir = self.get_features_dir("sbert", n_categories)
        file_map = {
            "train": "train_embeddings.npy",
            "val": "val_embeddings.npy",
            "test": "test_embeddings.npy",
        }

        if split not in file_map:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(file_map.keys())}")

        feature_path = sbert_dir / file_map[split]

        if not feature_path.exists():
            raise FileNotFoundError(
                f"SBERT {split} features not found at {feature_path}. "
                "Run SBERT feature extraction first."
            )

        features = np.load(feature_path)
        logger.info(f" Loaded SBERT {split} features from {feature_path}")
        return features

def main():
    extractor = FeatureExtractor()
    extractor.extract_features_all_categories()
    summary = extractor.create_feature_summary()
    validation = extractor.validate_features()
    print("Feature extraction summary:", summary)
    print("Validation results:", validation)

if __name__ == "__main__":
    main()

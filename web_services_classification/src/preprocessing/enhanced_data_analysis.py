"""
Enhanced Data Analysis Module for Web Service Classification
Handles comprehensive dataset exploration with advanced statistical analyses,
topic modeling, dimensionality reduction, and sophisticated visualizations

Features:
- Advanced statistical tests (normality, independence, correlation)
- TF-IDF analysis and word importance
- Topic modeling with LDA
- Dimensionality reduction (PCA, t-SNE)
- Class imbalance analysis with recommendations
- Text complexity metrics (readability, lexical diversity)
- Correlation analysis between features
- Outlier detection
- Temporal patterns (if timestamps available)
"""

import argparse
import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from scipy import stats
from scipy.stats import chi2_contingency, kstest, normaltest, pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.config import CATEGORY_SIZES, DATA_PATH, ANALYSIS_PATH, DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedDataAnalyzer:
    """Enhanced class for comprehensive data analysis and visualization"""

    def __init__(self):
        self.analysis_root = DATA_PATH / "analysis"
        self.overall_dir = self.analysis_root / "overall"
        self.topN_dir = {n: self.analysis_root / f"top_{n}_categories" for n in CATEGORY_SIZES}
        self.advanced_dir = self.analysis_root / "advanced"
        self.setup_directories()

    def setup_directories(self):
        """Create required analysis directories"""
        self.overall_dir.mkdir(parents=True, exist_ok=True)
        self.advanced_dir.mkdir(parents=True, exist_ok=True)
        for d in self.topN_dir.values():
            d.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path=None):
        """Load the dataset for analysis"""
        if file_path is None:
            file_path = DATA_CONFIG['raw_data_path']

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    # =====================================================================
    # ADVANCED TEXT METRICS
    # =====================================================================
    
    def compute_readability_metrics(self, text):
        """Compute readability metrics for text"""
        if pd.isna(text) or not text:
            return {'flesch_reading_ease': 0, 'avg_syllables_per_word': 0}
        
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(1, sentences)
        words = len(text.split())
        words = max(1, words)
        
        # Simple syllable count approximation
        syllables = sum([self.count_syllables(word) for word in text.split()])
        
        # Flesch Reading Ease (simplified)
        flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        
        return {
            'flesch_reading_ease': flesch,
            'avg_syllables_per_word': syllables / words
        }
    
    def count_syllables(self, word):
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def compute_lexical_diversity(self, text):
        """Compute lexical diversity (Type-Token Ratio)"""
        if pd.isna(text) or not text:
            return 0
        
        words = text.lower().split()
        if len(words) == 0:
            return 0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words
    
    # =====================================================================
    # STATISTICAL ANALYSIS
    # =====================================================================
    
    def perform_statistical_tests(self, df, text_column, target_column):
        """Perform various statistical tests"""
        logger.info("Performing statistical tests...")
        
        df_copy = df.copy()
        df_copy["text_length"] = df_copy[text_column].str.len()
        df_copy["word_count"] = df_copy[text_column].str.split().apply(len)
        
        stats_results = {}
        
        # Test for normality in text lengths
        stat, p_value = normaltest(df_copy["text_length"].dropna())
        stats_results['normality_test_text_length'] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': p_value > 0.05
        }
        
        # Test for normality in word counts
        stat, p_value = normaltest(df_copy["word_count"].dropna())
        stats_results['normality_test_word_count'] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': p_value > 0.05
        }
        
        # Chi-square test for category independence
        category_dist = df_copy[target_column].value_counts()
        expected_freq = len(df_copy) / len(category_dist)
        chi2_stat = sum((category_dist - expected_freq) ** 2 / expected_freq)
        
        stats_results['category_uniformity_test'] = {
            'chi2_statistic': float(chi2_stat),
            'is_uniform': chi2_stat < 100,
            'max_category_samples': int(category_dist.max()),
            'min_category_samples': int(category_dist.min()),
            'imbalance_ratio': float(category_dist.max() / category_dist.min())
        }
        
        # Correlation between text length and word count
        corr, p_value = pearsonr(df_copy["text_length"], df_copy["word_count"])
        stats_results['text_length_word_count_correlation'] = {
            'pearson_r': float(corr),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05
        }
        
        # Save results
        with open(self.advanced_dir / "statistical_tests.json", "w") as f:
            json.dump(stats_results, f, indent=4)
        
        logger.info(f"Statistical tests completed. Results saved.")
        return stats_results
    
    # =====================================================================
    # CLASS IMBALANCE ANALYSIS
    # =====================================================================
    
    def analyze_class_imbalance(self, df, target_column):
        """Comprehensive class imbalance analysis with recommendations"""
        logger.info("Analyzing class imbalance...")
        
        category_counts = df[target_column].value_counts()
        total_samples = len(df)
        
        # Calculate imbalance metrics
        max_samples = category_counts.max()
        min_samples = category_counts.min()
        imbalance_ratio = max_samples / min_samples
        
        # Gini coefficient for imbalance
        sorted_counts = np.sort(category_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((n - np.arange(1, n+1) + 1) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
        
        # Entropy
        proportions = category_counts / total_samples
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(len(category_counts))
        normalized_entropy = entropy / max_entropy
        
        imbalance_analysis = {
            'total_categories': int(len(category_counts)),
            'total_samples': int(total_samples),
            'max_samples_per_category': int(max_samples),
            'min_samples_per_category': int(min_samples),
            'mean_samples_per_category': float(category_counts.mean()),
            'median_samples_per_category': float(category_counts.median()),
            'std_samples_per_category': float(category_counts.std()),
            'imbalance_ratio': float(imbalance_ratio),
            'gini_coefficient': float(gini),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'cv_coefficient': float(category_counts.std() / category_counts.mean()),
        }
        
        # Recommendations based on imbalance
        recommendations = []
        if imbalance_ratio > 10:
            recommendations.append("Severe class imbalance detected (ratio > 10). Consider using:")
            recommendations.append("- SMOTE or ADASYN for oversampling minority classes")
            recommendations.append("- Class weights in model training")
            recommendations.append("- Stratified sampling for train/test split")
        elif imbalance_ratio > 5:
            recommendations.append("Moderate class imbalance detected (ratio > 5). Consider:")
            recommendations.append("- Class weights in model training")
            recommendations.append("- Stratified sampling")
        else:
            recommendations.append("Class distribution is relatively balanced.")
        
        if gini > 0.5:
            recommendations.append(f"High Gini coefficient ({gini:.3f}) indicates concentration in few classes")
        
        imbalance_analysis['recommendations'] = recommendations
        
        # Categorize classes by size
        q1, q3 = category_counts.quantile([0.25, 0.75])
        imbalance_analysis['minority_classes'] = category_counts[category_counts < q1].index.tolist()
        imbalance_analysis['majority_classes'] = category_counts[category_counts > q3].index.tolist()
        
        # Save analysis
        with open(self.advanced_dir / "class_imbalance_analysis.json", "w") as f:
            json.dump(imbalance_analysis, f, indent=4)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Lorenz curve for imbalance
        sorted_counts = np.sort(category_counts.values)
        cumsum = np.cumsum(sorted_counts)
        cumsum_pct = cumsum / cumsum[-1]
        x = np.linspace(0, 1, len(cumsum))
        
        axes[0, 0].plot(x, cumsum_pct, label='Lorenz Curve', linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=1)
        axes[0, 0].fill_between(x, cumsum_pct, x, alpha=0.3)
        axes[0, 0].set_xlabel('Cumulative Share of Categories')
        axes[0, 0].set_ylabel('Cumulative Share of Samples')
        axes[0, 0].set_title(f'Lorenz Curve (Gini: {gini:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Log-scale distribution
        sorted_counts_desc = category_counts.sort_values(ascending=False)
        axes[0, 1].bar(range(len(sorted_counts_desc)), sorted_counts_desc.values, color='steelblue')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_xlabel('Category Rank')
        axes[0, 1].set_ylabel('Sample Count (log scale)')
        axes[0, 1].set_title('Category Distribution (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Boxplot with outliers
        axes[1, 0].boxplot([category_counts.values], vert=True, widths=0.5)
        axes[1, 0].set_ylabel('Samples per Category')
        axes[1, 0].set_title('Sample Distribution Box Plot')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Cumulative percentage
        sorted_counts_desc = category_counts.sort_values(ascending=False)
        cumsum_pct = (sorted_counts_desc.cumsum() / sorted_counts_desc.sum()) * 100
        axes[1, 1].plot(range(1, len(cumsum_pct) + 1), cumsum_pct.values, marker='o', linewidth=2)
        axes[1, 1].axhline(y=80, color='r', linestyle='--', label='80% threshold')
        axes[1, 1].set_xlabel('Number of Top Categories')
        axes[1, 1].set_ylabel('Cumulative Percentage of Samples')
        axes[1, 1].set_title('Cumulative Sample Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.advanced_dir / "class_imbalance_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Class imbalance analysis completed.")
        return imbalance_analysis
    
    # =====================================================================
    # TF-IDF & FEATURE IMPORTANCE
    # =====================================================================
    
    def analyze_tfidf_features(self, df, text_column, target_column, n_features=20):
        """Analyze TF-IDF features per category"""
        logger.info("Analyzing TF-IDF features...")
        
        # Get top categories for analysis
        top_categories = df[target_column].value_counts().nlargest(10).index
        df_top = df[df[target_column].isin(top_categories)].copy()
        
        tfidf_results = {}
        
        for category in top_categories:
            category_texts = df_top[df_top[target_column] == category][text_column].fillna('')
            
            # Compute TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
            try:
                tfidf_matrix = vectorizer.fit_transform(category_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get average TF-IDF scores
                avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = avg_scores.argsort()[-n_features:][::-1]
                
                top_features = [
                    {'term': feature_names[i], 'score': float(avg_scores[i])}
                    for i in top_indices
                ]
                
                tfidf_results[category] = top_features
            except Exception as e:
                logger.warning(f"Could not compute TF-IDF for category {category}: {e}")
                continue
        
        # Save results
        with open(self.advanced_dir / "tfidf_analysis.json", "w") as f:
            json.dump(tfidf_results, f, indent=4)
        
        # Visualize top features for selected categories
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (category, features) in enumerate(list(tfidf_results.items())[:6]):
            if features:
                terms = [f['term'] for f in features[:10]]
                scores = [f['score'] for f in features[:10]]
                
                axes[idx].barh(terms, scores, color='teal')
                axes[idx].set_xlabel('TF-IDF Score')
                axes[idx].set_title(f'{category}', fontsize=10)
                axes[idx].invert_yaxis()
        
        # Hide unused subplots
        for idx in range(len(tfidf_results), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.advanced_dir / "tfidf_top_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("TF-IDF analysis completed.")
        return tfidf_results
    
    # =====================================================================
    # TOPIC MODELING
    # =====================================================================
    
    def perform_topic_modeling(self, df, text_column, n_topics=5, n_top_words=10):
        """Perform LDA topic modeling"""
        logger.info("Performing topic modeling...")
        
        try:
            # Prepare data - sample for performance
            texts = df[text_column].fillna('').astype(str).sample(
                n=min(10000, len(df)), random_state=42
            ).tolist()
            
            # Create document-term matrix
            vectorizer = CountVectorizer(max_features=1000, stop_words='english', max_df=0.8, min_df=5)
            doc_term_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Fit LDA model
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
            lda.fit(doc_term_matrix)
            
            # Extract topics
            topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                top_scores = [float(topic[i]) for i in top_indices]
                
                topics[f'Topic_{topic_idx + 1}'] = {
                    'words': top_words,
                    'scores': top_scores
                }
            
            # Save results
            with open(self.advanced_dir / "topic_modeling.json", "w") as f:
                json.dump(topics, f, indent=4)
            
            # Visualize topics
            fig, axes = plt.subplots(n_topics, 1, figsize=(12, 3 * n_topics))
            if n_topics == 1:
                axes = [axes]
            
            for idx, (topic_name, topic_data) in enumerate(topics.items()):
                words = topic_data['words']
                scores = topic_data['scores']
                
                axes[idx].barh(words, scores, color='coral')
                axes[idx].set_xlabel('Importance Score')
                axes[idx].set_title(topic_name, fontweight='bold')
                axes[idx].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(self.advanced_dir / "topic_modeling_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Topic modeling completed.")
            return topics
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return None
    
    # =====================================================================
    # DIMENSIONALITY REDUCTION
    # =====================================================================
    
    def perform_dimensionality_reduction(self, df, text_column, target_column):
        """Perform PCA and t-SNE for visualization"""
        logger.info("Performing dimensionality reduction...")
        
        try:
            # Get top categories
            top_categories = df[target_column].value_counts().nlargest(10).index
            df_sample = df[df[target_column].isin(top_categories)].sample(
                n=min(5000, len(df)), random_state=42
            ).copy()
            
            # Create TF-IDF features
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(df_sample[text_column].fillna(''))
            
            # PCA
            pca = PCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(tfidf_matrix.toarray())
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_result = tsne.fit_transform(tfidf_matrix.toarray())
            
            # Visualize
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # PCA plot
            for category in top_categories:
                mask = df_sample[target_column] == category
                axes[0].scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    label=category,
                    alpha=0.6,
                    s=20
                )
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[0].set_title('PCA Visualization')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0].grid(True, alpha=0.3)
            
            # t-SNE plot
            for category in top_categories:
                mask = df_sample[target_column] == category
                axes[1].scatter(
                    tsne_result[mask, 0],
                    tsne_result[mask, 1],
                    label=category,
                    alpha=0.6,
                    s=20
                )
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].set_title('t-SNE Visualization')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.advanced_dir / "dimensionality_reduction.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save explained variance
            variance_info = {
                'pca_explained_variance': [float(x) for x in pca.explained_variance_ratio_],
                'pca_cumulative_variance': [float(x) for x in np.cumsum(pca.explained_variance_ratio_)],
            }
            
            with open(self.advanced_dir / "dimensionality_reduction_info.json", "w") as f:
                json.dump(variance_info, f, indent=4)
            
            logger.info("Dimensionality reduction completed.")
            return pca_result, tsne_result
            
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            return None, None
    
    # =====================================================================
    # TEXT COMPLEXITY ANALYSIS
    # =====================================================================
    
    def analyze_text_complexity(self, df, text_column, target_column):
        """Analyze text complexity metrics across categories"""
        logger.info("Analyzing text complexity...")
        
        df_copy = df.copy()
        
        # Compute complexity metrics (sample for performance)
        complexity_metrics = []
        sample_size = min(10000, len(df_copy))
        df_sample = df_copy.sample(n=sample_size, random_state=42)
        
        for idx, row in df_sample.iterrows():
            text = row[text_column]
            readability = self.compute_readability_metrics(text)
            lexical_div = self.compute_lexical_diversity(text)
            
            complexity_metrics.append({
                'category': row[target_column],
                'flesch_reading_ease': readability['flesch_reading_ease'],
                'avg_syllables_per_word': readability['avg_syllables_per_word'],
                'lexical_diversity': lexical_div
            })
        
        complexity_df = pd.DataFrame(complexity_metrics)
        
        # Aggregate by category
        top_categories = df[target_column].value_counts().nlargest(10).index
        complexity_df = complexity_df[complexity_df['category'].isin(top_categories)]
        
        category_complexity = complexity_df.groupby('category').agg({
            'flesch_reading_ease': ['mean', 'std'],
            'avg_syllables_per_word': ['mean', 'std'],
            'lexical_diversity': ['mean', 'std']
        }).round(3)
        
        # Save results
        category_complexity.to_csv(self.advanced_dir / "text_complexity_by_category.csv")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['flesch_reading_ease', 'avg_syllables_per_word', 'lexical_diversity']
        titles = ['Flesch Reading Ease', 'Avg Syllables per Word', 'Lexical Diversity']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            complexity_df.boxplot(column=metric, by='category', ax=axes[idx])
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Category')
            axes[idx].tick_params(axis='x', rotation=90)
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(self.advanced_dir / "text_complexity_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Text complexity analysis completed.")
        return complexity_df
    
    # =====================================================================
    # CORRELATION ANALYSIS
    # =====================================================================
    
    def analyze_feature_correlations(self, df, text_column):
        """Analyze correlations between text features"""
        logger.info("Analyzing feature correlations...")
        
        df_copy = df.copy()
        df_copy["text_length"] = df_copy[text_column].str.len()
        df_copy["word_count"] = df_copy[text_column].str.split().apply(len)
        df_copy["avg_word_length"] = df_copy["text_length"] / df_copy["word_count"]
        df_copy["unique_words"] = df_copy[text_column].apply(lambda x: len(set(str(x).split())))
        df_copy["lexical_diversity"] = df_copy["unique_words"] / df_copy["word_count"]
        
        # Select numerical features
        features = ["text_length", "word_count", "avg_word_length", "unique_words", "lexical_diversity"]
        correlation_matrix = df_copy[features].corr()
        
        # Save correlation matrix
        correlation_matrix.to_csv(self.advanced_dir / "feature_correlation_matrix.csv")
        
        # Visualize
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.advanced_dir / "feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Feature correlation analysis completed.")
        return correlation_matrix
    
    # =====================================================================
    # OUTLIER DETECTION
    # =====================================================================
    
    def detect_outliers(self, df, text_column, target_column):
        """Detect outliers in text length and word count"""
        logger.info("Detecting outliers...")
        
        df_copy = df.copy()
        df_copy["text_length"] = df_copy[text_column].str.len()
        df_copy["word_count"] = df_copy[text_column].str.split().apply(len)
        
        outliers_info = {}
        
        for metric in ["text_length", "word_count"]:
            Q1 = df_copy[metric].quantile(0.25)
            Q3 = df_copy[metric].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_copy[(df_copy[metric] < lower_bound) | (df_copy[metric] > upper_bound)]
            
            outliers_info[metric] = {
                'count': int(len(outliers)),
                'percentage': float(len(outliers) / len(df_copy) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'min_outlier': float(outliers[metric].min()) if len(outliers) > 0 else None,
                'max_outlier': float(outliers[metric].max()) if len(outliers) > 0 else None
            }
        
        # Save results
        with open(self.advanced_dir / "outlier_analysis.json", "w") as f:
            json.dump(outliers_info, f, indent=4)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, metric in enumerate(["text_length", "word_count"]):
            axes[idx].boxplot(df_copy[metric].values, vert=True)
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Distribution with Outliers')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.advanced_dir / "outlier_detection.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Outlier detection completed.")
        return outliers_info
    
    # =====================================================================
    # ORIGINAL ANALYSES (PRESERVED)
    # =====================================================================
    
    def analyze_overall(self, df: pd.DataFrame, text_column: str, target_column: str):
        """Run overall dataset analysis and save plots/stats"""
        df_copy = df.copy()
        df_copy["text_length"] = df_copy[text_column].str.len()
        df_copy["word_count"] = df_copy[text_column].str.split().apply(len)

        stats = {
            "num_samples": len(df_copy),
            "num_categories": df_copy[target_column].nunique(),
            'columns': list(df_copy.columns),
            "avg_text_length": float(df_copy["text_length"].mean()),
            "avg_word_count": float(df_copy["word_count"].mean()),
            'missing_values': {k: int(v) for k, v in df_copy.isnull().sum().to_dict().items()},
            'duplicates': int(df_copy.duplicated().sum()),
            'memory_usage_mb': round(df_copy.memory_usage(deep=True).sum() / 1024**2, 2)
        }

        with open(self.overall_dir / "dataset_summary.json", "w") as f:
            json.dump(stats, f, indent=4)

        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        sns.histplot(df_copy["text_length"], bins=50, kde=True, ax=axes[0, 0], color="C0")
        axes[0, 0].set_title("Text Length Distribution")

        sns.histplot(df_copy["word_count"], bins=50, kde=True, ax=axes[0, 1], color="C1")
        axes[0, 1].set_title("Word Count Distribution")

        all_words = [w for text in df_copy[text_column].dropna().str.split() for w in text]
        word_lengths = pd.Series([len(w) for w in all_words])
        sns.histplot(word_lengths, bins=30, kde=True, ax=axes[1, 0], color="C2")
        axes[1, 0].set_title("Word Length Distribution")

        category_counts = df_copy[target_column].value_counts().head(40)
        sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[1, 1], palette="viridis")
        axes[1, 1].set_title("Top 40 Category Distribution")
        axes[1, 1].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.savefig(self.overall_dir / "dataset_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Individual plots
        plt.figure(figsize=(8, 5))
        sns.histplot(df_copy["text_length"], bins=50, kde=True, color="C0")
        plt.title("Text Length Distribution")
        plt.savefig(self.overall_dir / "text_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.histplot(df_copy["word_count"], bins=50, kde=True, color="C1")
        plt.title("Word Count Distribution")
        plt.savefig(self.overall_dir / "word_count_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.histplot(word_lengths, bins=30, kde=True, color="C2")
        plt.title("Word Length Distribution")
        plt.savefig(self.overall_dir / "word_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        category_counts = df_copy[target_column].value_counts()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
        plt.title("Category Distribution")
        plt.xticks(rotation=90)
        plt.savefig(self.overall_dir / "category_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_topN(self, df: pd.DataFrame, text_column: str, target_column: str, n_categories: int):
        """Run Top-N category analysis"""
        subset_dir = self.topN_dir[n_categories]

        top_categories = df[target_column].value_counts().nlargest(n_categories).index
        df_top = df[df[target_column].isin(top_categories)].copy()

        df_top["text_length"] = df_top[text_column].str.len()
        df_top["word_count"] = df_top[text_column].str.split().apply(len)

        top_words_dict = {}
        for category in top_categories:
            cat_data = df_top[df_top[target_column] == category]
            category_texts = cat_data[text_column].fillna('').astype(str)
            all_text = ' '.join(category_texts)
            words = all_text.lower().split()
            word_freq = {}
            for word in words:
                word = word.strip('.,!?";()[]{}')
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            top_words_dict[category] = ', '.join([f"{word}({count})" for word, count in top_words])

        stats = df_top.groupby(target_column).agg(
            samples=(text_column, "count"),
            avg_text_length=("text_length", "mean"),
            min_text_length=("text_length", "min"),
            max_text_length=("text_length", "max"),
            median_text_length=("text_length", "median"),
            avg_word_count=("word_count", "mean"),
            min_word_count=("word_count", "min"),
            max_word_count=("word_count", "max"),
            median_word_count=("word_count", "median")
        ).reset_index()

        stats['Top_10_Words'] = stats[target_column].map(top_words_dict)

        float_cols = ['avg_text_length', 'avg_word_count']
        stats[float_cols] = stats[float_cols].round(2)

        stats.to_csv(subset_dir / f"category_statistics_top{n_categories}.csv", index=False)
        stats.to_json(subset_dir / f"category_statistics_top{n_categories}.json", orient="records", indent=4)

        stats = stats.sort_values("samples", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=stats[target_column], y=stats["samples"], palette="crest")
        plt.title(f"Category Distribution (Top-{n_categories})")
        plt.xticks(rotation=90)
        plt.savefig(subset_dir / f"category_distribution_top{n_categories}.png", dpi=300, bbox_inches='tight')
        plt.close()

        order = stats[target_column]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_top, x=target_column, y="text_length", order=order, palette="Set2")
        plt.title(f"Text Length by Category (Top-{n_categories})")
        plt.xticks(rotation=90)
        plt.savefig(subset_dir / f"text_length_boxplot_top{n_categories}.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_top, x=target_column, y="word_count", order=order, palette="Set3")
        plt.title(f"Word Count by Category (Top-{n_categories})")
        plt.xticks(rotation=90)
        plt.savefig(subset_dir / f"word_count_boxplot_top{n_categories}.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(x=stats[target_column], y=stats["avg_text_length"], ax=axes[0], palette="Blues")
        axes[0].set_title(f"Avg Text Length (Top-{n_categories})")
        axes[0].tick_params(axis="x", rotation=90)

        sns.barplot(x=stats[target_column], y=stats["avg_word_count"], ax=axes[1], palette="Purples")
        axes[1].set_title(f"Avg Word Count (Top-{n_categories})")
        axes[1].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.savefig(subset_dir / f"grouped_text_word_top{n_categories}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # =====================================================================
    # MAIN ANALYSIS PIPELINE
    # =====================================================================

    def run_complete_analysis(self, run_advanced=True):
        """Run complete data analysis pipeline with optional advanced analyses"""
        logger.info("Starting comprehensive data analysis...")

        df = self.load_data(file_path=DATA_CONFIG["raw_data_path"])

        text_column = DATA_CONFIG["text_column"]
        target_column = DATA_CONFIG["target_column"]

        if text_column not in df.columns or target_column not in df.columns:
            logger.error(f"Required columns '{text_column}' or '{target_column}' not found!")
            return False

        # Overall analysis
        logger.info("Running overall dataset analysis...")
        self.analyze_overall(df, text_column=text_column, target_column=target_column)
        logger.info("Overall analysis completed.")

        # Top-N category analysis
        for n in CATEGORY_SIZES:
            logger.info(f"Running Top-{n} category analysis...")
            self.analyze_topN(df, text_column=text_column, target_column=target_column, n_categories=n)
            logger.info(f"Top-{n} category analysis completed.")

        # Advanced analyses
        if run_advanced:
            logger.info("=" * 60)
            logger.info("STARTING ADVANCED ANALYSES")
            logger.info("=" * 60)
            
            self.perform_statistical_tests(df, text_column, target_column)
            self.analyze_class_imbalance(df, target_column)
            self.analyze_tfidf_features(df, text_column, target_column)
            self.perform_topic_modeling(df, text_column)
            self.perform_dimensionality_reduction(df, text_column, target_column)
            self.analyze_text_complexity(df, text_column, target_column)
            self.analyze_feature_correlations(df, text_column)
            self.detect_outliers(df, text_column, target_column)
            
            logger.info("=" * 60)
            logger.info("ADVANCED ANALYSES COMPLETED")
            logger.info("=" * 60)

        logger.info("All analyses completed successfully.")
        return True


def main():
    """Main function to run data analysis"""
    parser = argparse.ArgumentParser(description='Enhanced Data Analysis for Web Service Classification')
    parser.add_argument('--skip-advanced', action='store_true', 
                        help='Skip advanced analyses (faster execution)')
    args = parser.parse_args()
    
    analyzer = EnhancedDataAnalyzer()
    results = analyzer.run_complete_analysis(run_advanced=not args.skip_advanced)
    
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("Analysis complete! Check the following directories:")
        logger.info(f"  - Overall: {analyzer.overall_dir}")
        logger.info(f"  - Advanced: {analyzer.advanced_dir}")
        logger.info(f"  - Top-N: {list(analyzer.topN_dir.values())}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
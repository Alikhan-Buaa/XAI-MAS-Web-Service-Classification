"""
Data Analysis Module for Web Service Classification
Handles comprehensive dataset exploration, statistics, and visualizations

Modified to support splitting the dataset into top-N-category subsets (top10, top20, ...),
optionally save those split datasets with labels, and run analysis per-split while
preserving all original config/path usage.
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

from src.config import CATEGORY_SIZES, DATA_PATH,ANALYSIS_PATH, DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Main class for data analysis and visualization"""

    def __init__(self):
        self.analysis_root = DATA_PATH / "analysis"
        self.overall_dir = self.analysis_root / "overall"
        self.topN_dir = {n: self.analysis_root / f"top_{n}_categories" for n in CATEGORY_SIZES}
        self.setup_directories()

    def setup_directories(self):
        """Create required analysis directories"""
        self.overall_dir.mkdir(parents=True, exist_ok=True)
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
        
        
    # ---------------------------------------------------------------------
    # Overall Dataset Analysis
    # ---------------------------------------------------------------------
    def analyze_overall(self, df: pd.DataFrame, text_column: str, target_column: str):
        """Run overall dataset analysis and save plots/stats"""
        # ---- Stats ----
        df["text_length"] = df[text_column].str.len()
        df["word_count"] = df[text_column].str.split().apply(len)

        stats = {
            "num_samples": len(df),
            "num_categories": df[target_column].nunique(),
            'columns': list(df.columns),
            "avg_text_length": df["text_length"].mean(),
            "avg_word_count": df["word_count"].mean(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }

        with open(self.overall_dir / "dataset_summary.json", "w") as f:
            json.dump(stats, f, indent=4)

        # ---- Plots ----
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Text length distribution
        sns.histplot(df["text_length"], bins=50, kde=True, ax=axes[0, 0], color="C0")
        axes[0, 0].set_title("Text Length Distribution")

        # Word count distribution
        sns.histplot(df["word_count"], bins=50, kde=True, ax=axes[0, 1], color="C1")
        axes[0, 1].set_title("Word Count Distribution")

        # Word length distribution
        all_words = [w for text in df[text_column].dropna().str.split() for w in text]
        word_lengths = pd.Series([len(w) for w in all_words])
        sns.histplot(word_lengths, bins=30, kde=True, ax=axes[1, 0], color="C2")
        axes[1, 0].set_title("Word Length Distribution")

        # Category distribution
        category_counts = df[target_column].value_counts().head(40)
        sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[1, 1], palette="viridis")
        axes[1, 1].set_title("Top 40 Category Distribution")
        axes[1, 1].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.savefig(self.overall_dir / "dataset_overview.png", dpi=300)
        plt.close()

        # ---- Individual Plots ----
        # Text length distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df["text_length"], bins=50, kde=True, color="C0")
        plt.title("Text Length Distribution")
        plt.savefig(self.overall_dir / "text_length_distribution.png", dpi=300)
        plt.close()

        # Word count distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df["word_count"], bins=50, kde=True, color="C1")
        plt.title("Word Count Distribution")
        plt.savefig(self.overall_dir / "word_count_distribution.png", dpi=300)
        plt.close()

        # Word length distribution
        all_words = [w for text in df[text_column].dropna().str.split() for w in text]
        word_lengths = pd.Series([len(w) for w in all_words])
        plt.figure(figsize=(8, 5))
        sns.histplot(word_lengths, bins=30, kde=True, color="C2")
        plt.title("Word Length Distribution")
        plt.savefig(self.overall_dir / "word_length_distribution.png", dpi=300)
        plt.close()

        # Category distribution (descending order)
        category_counts = df[target_column].value_counts()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
        plt.title("Category Distribution")
        plt.xticks(rotation=90)
        plt.savefig(self.overall_dir / "category_distribution.png", dpi=300)
        plt.close()

    # ---------------------------------------------------------------------
    # Category-wise Analysis for Top-N
    # ---------------------------------------------------------------------
    def analyze_topN(self, df: pd.DataFrame, text_column: str, target_column: str, n_categories: int):
        """Run Top-N category analysis"""
        subset_dir = self.topN_dir[n_categories]

        # Select top-N categories
        top_categories = df[target_column].value_counts().nlargest(n_categories).index
        df_top = df[df[target_column].isin(top_categories)].copy()

        # ---- Stats ----
        df_top["text_length"] = df_top[text_column].str.len()
        df_top["word_count"] = df_top[text_column].str.split().apply(len)

        # Compute per-category top 10 words
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

        # Grouped statistics
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

        # Add top words per category
        stats['Top_10_Words'] = stats[target_column].map(top_words_dict)

        # Round float columns
        float_cols = ['avg_text_length', 'avg_word_count']
        stats[float_cols] = stats[float_cols].round(2)

        # Save CSV and JSON
        stats.to_csv(subset_dir / f"category_statistics_top{n_categories}.csv", index=False)
        stats.to_json(subset_dir / f"category_statistics_top{n_categories}.json", orient="records", indent=4)

        # ---- Individual Plots ----
        # Category distribution
        stats = stats.sort_values("samples", ascending=False)

        # ---- Individual Plots ----
        plt.figure(figsize=(10, 6))
        sns.barplot(x=stats[target_column], y=stats["samples"], palette="crest")
        plt.title(f"Category Distribution (Top-{n_categories})")
        plt.xticks(rotation=90)
        plt.savefig(subset_dir / f"category_distribution_top{n_categories}.png", dpi=300)
        plt.close()

        order = stats[target_column]  # already sorted by samples descending

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_top, x=target_column, y="text_length", order=order, palette="Set2")
        plt.title(f"Text Length by Category (Top-{n_categories})")
        plt.xticks(rotation=90)
        plt.savefig(subset_dir / f"text_length_boxplot_top{n_categories}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_top, x=target_column, y="word_count", order=order, palette="Set3")
        plt.title(f"Word Count by Category (Top-{n_categories})")
        plt.xticks(rotation=90)
        plt.savefig(subset_dir / f"word_count_boxplot_top{n_categories}.png", dpi=300)
        plt.close()

        # ---- Grouped Multi-Plot ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(x=stats[target_column], y=stats["avg_text_length"], ax=axes[0], palette="Blues")
        axes[0].set_title(f"Avg Text Length (Top-{n_categories})")
        axes[0].tick_params(axis="x", rotation=90)

        sns.barplot(x=stats[target_column], y=stats["avg_word_count"], ax=axes[1], palette="Purples")
        axes[1].set_title(f"Avg Word Count (Top-{n_categories})")
        axes[1].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.savefig(subset_dir / f"grouped_text_word_top{n_categories}.png", dpi=300)
        plt.close()

   

    def run_complete_analysis(self):
        """Run complete data analysis pipeline"""
        logger.info("Starting comprehensive data analysis...")

        # Load data
        df = self.load_data(file_path=DATA_CONFIG["raw_data_path"])

        # Use column names from DATA_CONFIG
        text_column = DATA_CONFIG["text_column"]
        target_column = DATA_CONFIG["target_column"]

        # Check for essential columns
        if text_column not in df.columns or target_column not in df.columns:
            logger.error(f"Required columns '{text_column}' or '{target_column}' not found in dataset!")
            return

        # ---- Overall analysis ----
        logger.info("Running overall dataset analysis...")
        self.analyze_overall(df, text_column=text_column, target_column=target_column)
        logger.info("Overall analysis completed.")

        # ---- Top-N category analysis ----
        for n in CATEGORY_SIZES:
            logger.info(f"Running Top-{n} category analysis...")
            self.analyze_topN(df, text_column=text_column, target_column=target_column, n_categories=n)
            logger.info(f"Top-{n} category analysis completed.")

        logger.info("All analyses completed successfully.")
        return True


def main():
    """Main function to run data analysis"""
    analyzer = DataAnalyzer()
    results = analyzer.run_complete_analysis()
    

if __name__ == "__main__":
    main()


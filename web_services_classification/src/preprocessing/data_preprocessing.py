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
import matplotlib.pyplot as plt
from pathlib import Path


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
    
    def plot_top_words_from_json(self,json_path, output_dir, max_categories=None):
        """Generate plots for top words per category"""

        with open(json_path, 'r') as f:
            data = json.load(f)

        if max_categories:
            data = data[:max_categories]

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for item in data:
            category = item[list(item.keys())[0]]
            top_words_str = item.get("Top_10_Words", "")

            if not top_words_str:
                continue

            words, counts = [], []

            for wc in top_words_str.split(", "):
                try:
                    word, count = wc.rsplit("(", 1)
                    words.append(word)
                    counts.append(int(count.strip(")")))
                except:
                    continue

            # Sort ascending so highest bar is at the top
            words_counts = sorted(zip(words, counts), key=lambda x: x[1])
            words, counts = zip(*words_counts)

            # ── Chart setup ───────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(11, 6))
            fig.patch.set_facecolor("#F8F9FA")
            ax.set_facecolor("#F8F9FA")

            # Gradient colours — deepest blue for top word, lightest for last
            import matplotlib.colors as mcolors
            cmap = plt.cm.Blues
            norm = mcolors.Normalize(vmin=0, vmax=len(counts) - 1)
            bar_colors = [cmap(norm(i)) for i in range(len(counts))]

            bars = ax.barh(
                words, counts,
                color=bar_colors,
                height=0.65,
                edgecolor="white",
                linewidth=0.6,
            )

            # Value labels — white inside if bar wide enough, dark outside otherwise
            max_count = max(counts)
            for bar in bars:
                width = bar.get_width()
                label = f"{int(width):,}"
                threshold = max_count * 0.18
                if width > threshold:
                    ax.text(
                        width - max_count * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        va="center", ha="right",
                        fontsize=9, fontweight="bold", color="white",
                    )
                else:
                    ax.text(
                        width + max_count * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        va="center", ha="left",
                        fontsize=9, fontweight="bold", color="#333333",
                    )

            # Titles & labels
            ax.set_title(
                f"Top Words — {category}",
                fontsize=14, fontweight="bold", color="#1A1A2E", pad=14,
            )
            ax.set_xlabel("Frequency", fontsize=11, color="#444444", labelpad=8)
            ax.tick_params(axis="y", labelsize=10, colors="#333333")
            ax.tick_params(axis="x", labelsize=9,  colors="#555555")

            # Clean spines
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#CCCCCC")
            ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#CCCCCC")
            ax.set_axisbelow(True)

            # x-axis starts at 0
            ax.set_xlim(left=0, right=max_count * 1.12)

            plt.tight_layout()

            # Safe filename
            safe_category = str(category).replace(" ", "_").replace("/", "_")
            plt.savefig(
                Path(output_dir) / f"{safe_category}_top_words.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(),
            )
            plt.close(fig)

    def save_cleaned_category_statistics_from_file(self, text_column: str, target_column: str, n_categories: int):
        """Load cleaned dataset from file and compute category statistics"""

        # 🔹 Path
        processed_dir = Path(str(PREPROCESSING_CONFIG['processed_data']).format(n=n_categories))
        input_file = processed_dir / "cleaned_dataset.csv"

        if not input_file.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {input_file}")

        # 🔹 Load dataset
        df = pd.read_csv(input_file)

        # 🔹 Top-N categories
        top_categories = df[target_column].value_counts().nlargest(n_categories).index
        df_top = df[df[target_column].isin(top_categories)].copy()

        # 🔹 Stats
        df_top["text_length"] = df_top[text_column].astype(str).str.len()
        df_top["word_count"] = df_top[text_column].astype(str).str.split().apply(len)

        # 🔹 Top words
        top_words_dict = {}
        for category in top_categories:
            cat_data = df_top[df_top[target_column] == category]
            texts = cat_data[text_column].fillna('').astype(str)

            all_text = ' '.join(texts)
            words = all_text.lower().split()

            word_freq = {}
            for word in words:
                word = word.strip('.,!?";()[]{}')
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1

            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            top_words_dict[category] = ', '.join([f"{w}({c})" for w, c in top_words])

        # 🔹 Aggregation
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

        stats["Top_10_Words"] = stats[target_column].map(top_words_dict)
        stats[["avg_text_length", "avg_word_count"]] = stats[["avg_text_length", "avg_word_count"]].round(2)

        # 🔹 Save output
        stats.to_csv(processed_dir / f"cleaned_category_statistics_top{n_categories}.csv", index=False)
        stats.to_json(processed_dir / f"cleaned_category_statistics_top{n_categories}.json", orient="records", indent=4)

        logger.info(f"Category statistics generated from cleaned dataset at {processed_dir}")

        plots_dir = processed_dir / "top_words_plots"
        json_path = processed_dir / f"cleaned_category_statistics_top{n_categories}.json"

        self.plot_top_words_from_json(
            json_path=json_path,
            output_dir=plots_dir,
            max_categories=n_categories
        )

        logger.info(f"Top word plots saved at {plots_dir}")

    def save_explainability_samples(self, test_df: pd.DataFrame, n_categories: int) -> None:
        """
        Select 1 representative row per category from EXPLAINABILITY_CONFIG['expl_categories']
        and save to:
            data/processed/top_{n}_categories/explainability_test_samples.csv
            data/processed/top_{n}_categories/explainability_test_samples.json

        Category list comes from config — single source of truth, no hardcoded names.
        No hardcoded row indices — scans test_df live by Service Classification column.

        CSV columns: category | encoded_label | row_index |
                     Service Classification | cleaned_text | text_preview
        """
        # Single source of truth: category list from config
        try:
            from src.config import EXPLAINABILITY_CONFIG as _EC
            FIXED_CATEGORIES = _EC.get("expl_categories", [
                "Payments", "Messaging", "Social", "Storage", "eCommerce",
            ])
        except ImportError as _ie:
            logger.warning(f"save_explainability_samples: cannot import config ({_ie}) — skipping.")
            return

        df = test_df.reset_index(drop=True)

        if "encoded_label" not in df.columns:
            logger.warning("save_explainability_samples: 'encoded_label' not in test_df — skipping.")
            return

        target_col = DATA_CONFIG["target_column"]
        if target_col not in df.columns:
            logger.warning(f"save_explainability_samples: '{target_col}' not in test_df — skipping.")
            return

        rows_out: list = []

        for cat in FIXED_CATEGORIES:
            cat_rows = df[df[target_col] == cat]
            if cat_rows.empty:
                logger.warning(f"  save_explainability_samples: '{cat}' not found in test split — skipped.")
                continue

            pos_idx = int(cat_rows.index[0])
            row     = df.iloc[pos_idx]
            enc_lbl = int(row["encoded_label"])

            rows_out.append({
                "category":               cat,
                "encoded_label":          enc_lbl,
                "row_index":              pos_idx,
                "Service Classification": str(row.get(target_col, "")),
                "cleaned_text":           str(row.get("cleaned_text", "")),
                "text_preview":           str(row.get("cleaned_text", ""))[:80],
            })
            logger.info(f"  save_explainability_samples: '{cat}' → row {pos_idx} (label={enc_lbl})")

        if not rows_out:
            logger.warning("save_explainability_samples: no rows collected — skipping.")
            return

        out_dir = Path(str(PREPROCESSING_CONFIG["processed_data"]).format(n=n_categories))
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "explainability_test_samples.csv"
        pd.DataFrame(rows_out).to_csv(csv_path, index=False)
        logger.info(f"  Explainability samples CSV  ({len(rows_out)} rows) → {csv_path}")

        import json as _json
        json_path = out_dir / "explainability_test_samples.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            _json.dump(
                {
                    "n_categories":   n_categories,
                    "n_samples":      len(rows_out),
                    "n_per_category": 1,
                    "categories":     FIXED_CATEGORIES,
                    "samples":        {r["category"]: r for r in rows_out},
                },
                fh, indent=2, ensure_ascii=False,
            )
        logger.info(f"  Explainability samples JSON ({len(rows_out)} rows) → {json_path}")

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
        
        self.save_cleaned_category_statistics_from_file(
            text_column='cleaned_text',
            target_column=DATA_CONFIG['target_column'],
            n_categories=n_categories
        )
        # Create train/val/test splits
        train_df, val_df, test_df = self.create_train_val_test_split(filtered_df)
        self.save_splits(train_df, val_df, test_df, n_categories)
        
        # Save the 15 fixed explainability samples used by all 5 XAI modules
        self.save_explainability_samples(test_df, n_categories)
        
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

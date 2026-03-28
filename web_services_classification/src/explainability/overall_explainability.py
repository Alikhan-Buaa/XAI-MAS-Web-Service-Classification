"""
Overall Explainability Visualization & Token Consolidation
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
from collections import Counter, defaultdict
import sys

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the new config section
from src.config import RESULTS_PATH, OVERALL_EXPLAINABILITY_CONFIG

# Canonical 5-category shared explainability set — same as all 5 model modules
from src.utils.explainability_utils import EXPL_TARGET_CATEGORIES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

# Use constants from config
MODEL_COLORS = OVERALL_EXPLAINABILITY_CONFIG['model_colors']
METRIC_MAPPING = OVERALL_EXPLAINABILITY_CONFIG['metric_mapping']

def load_all_metrics(n_categories):
    """Crawls result directories to find metrics CSVs."""
    combined_data = []
    
    # Dynamic path construction using config
    filenames = OVERALL_EXPLAINABILITY_CONFIG['metrics_files']
    base_res = RESULTS_PATH
    
    paths_to_check = [
        base_res / "ml" / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['ml'],
        base_res / "dl" / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['dl'],
        base_res / "bert" / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['bert'],
        base_res / "deepseek" / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['deepseek'],
        base_res / "fusion" / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['fusion']
    ]

    for path in paths_to_check:
        if path.exists():
            try:
                df = pd.read_csv(path)
                cols = [c.lower() for c in df.columns]
                
                # Handling raw sample files
                is_summary = any('mean' in c for c in cols)
                
                if 'model' in cols or 'model_name' in cols:
                    model_col = 'model' if 'model' in df.columns else 'model_name'
                    
                    if not is_summary:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        df_grouped = df.groupby(model_col)[numeric_cols].mean().reset_index()
                    else:
                        df_grouped = df

                    for _, row in df_grouped.iterrows():
                        model_name = row[model_col]
                        clean_name = normalize_model_name(model_name)
                        metrics = extract_metrics_from_row(row)
                        if metrics:
                            metrics['Model'] = clean_name
                            combined_data.append(metrics)
                            
                else:
                    # Fallback for DeepSeek/Fusion
                    if "deepseek" in str(path).lower(): current_model = "DeepSeek_7B"
                    elif "fusion" in str(path).lower(): current_model = "DeepSeek_RoBERTa_Fusion"
                    else: current_model = "Unknown"

                    metrics = extract_metrics_from_row(df.mean(numeric_only=True))
                    if metrics:
                        metrics['Model'] = current_model
                        combined_data.append(metrics)

            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
        else:
            logger.warning(f"File not found: {path}")

    if not combined_data:
        return pd.DataFrame()

    return pd.DataFrame(combined_data)

def normalize_model_name(name):
    name = str(name).lower()
    if 'logistic' in name: return 'LogisticRegression'
    if 'forest' in name: return 'RandomForest'
    if 'xgb' in name: return 'XGBoost'
    if 'lstm' in name or 'bilstm' in name: return 'BiLSTM'
    if 'roberta' in name and 'large' in name: return 'RoBERTa_Large'
    if 'roberta' in name: return 'RoBERTa_Base'
    if 'deepseek' in name and 'fusion' not in name: return 'DeepSeek_7B'
    if 'fusion' in name: return 'DeepSeek_RoBERTa_Fusion'
    return "Unknown"

def extract_metrics_from_row(row):
    metrics = {}
    row_dict = {k.lower(): v for k, v in row.items()}
    
    def get_val(keys):
        for k in keys:
            if k in row_dict:
                try: return float(row_dict[k])
                except: return 0.0
        return None

    j_val = get_val(['jaccard', 'jaccard_mean'])
    if j_val is not None: metrics['Jaccard'] = j_val
    
    f_val = get_val(['fidelity', 'fidelity_mean'])
    if f_val is not None: metrics['Fidelity'] = f_val
    
    s_val = get_val(['stability', 'stability_mean'])
    if s_val is not None: metrics['Stability'] = s_val
    
    return metrics if metrics else None

def plot_comparison_chart(df, output_path, title_suffix=""):
    df_long = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    desired_order = ['Jaccard', 'Fidelity', 'Stability']
    df_long = df_long[df_long['Metric'].isin(desired_order)]
    
    metrics = desired_order
    models = list(MODEL_COLORS.keys())
    existing_models = [m for m in models if m in df['Model'].unique()]
    
    x = np.arange(len(metrics))
    width = 0.10
    multiplier = 0

    fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')

    for model in existing_models:
        model_data = df_long[df_long['Model'] == model]
        scores = []
        for m in metrics:
            val = model_data[model_data['Metric'] == m]['Score'].values
            scores.append(np.mean(val) if len(val) > 0 else 0)
        
        offset = width * multiplier
        rects = ax.bar(x + offset, scores, width, label=model, color=MODEL_COLORS.get(model, 'grey'))
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9, fontweight='bold')
        multiplier += 1

    if len(existing_models) > 0:
        center_offset = (width * len(existing_models)) / 2 - (width / 2)
        ax.set_xticks(x + center_offset)
    else:
        ax.set_xticks(x)
        
    ax.set_xticklabels([METRIC_MAPPING[m] for m in metrics], fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'XAI Metrics Comparison across Models {title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.15)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Models", borderaxespad=0.)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved to {output_path}")
    plt.close()

# --- TOKEN CONSOLIDATION ---

def clean_token_string(token_str):
    if not isinstance(token_str, str): return []
    token_str = token_str.replace('[', '').replace(']', '').replace("'", "")
    return [t.strip() for t in token_str.split(',') if t.strip()]

def consolidate_tokens(n_categories):
    logger.info("Starting Token Consolidation...")
    
    base_path = RESULTS_PATH
    filenames = OVERALL_EXPLAINABILITY_CONFIG['token_files']
    
    file_map = {
        'BERT': base_path / "bert" / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['bert'],
        'DeepSeek': base_path / "deepseek" / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['deepseek'],
        'DL': base_path / "dl" / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['dl'],
        'Fusion': base_path / "fusion" / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['fusion'],
        'ML': base_path / "ml" / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['ml']
    }

    cat_tokens_map = defaultdict(list)
    
    for model_name, path in file_map.items():
        if not path.exists():
            logger.warning(f"Missing token file for {model_name}: {path}")
            continue
            
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            
            # Robust column finding
            cat_col = next((c for c in df.columns if 'category' in c), None)
            
            # Flexible Token Column Search
            tok_col = None
            possible_token_cols = ['consolidated_top_15_tokens', 'tokens', 'words', 'features']
            
            # 1. Exact match check (from your config)
            if 'consolidated_top_15_tokens' in df.columns:
                tok_col = 'consolidated_top_15_tokens'
            # 2. Heuristic check
            else:
                for cand in df.columns:
                    if any(x in cand for x in ['token', 'word', 'feature']):
                        tok_col = cand
                        break
            
            if cat_col and tok_col:
                for _, row in df.iterrows():
                    cat = row[cat_col]
                    tokens = clean_token_string(str(row[tok_col]))
                    cat_tokens_map[cat].extend(tokens)
            else:
                logger.warning(f"Could not find Category/Token columns in {model_name}. Cols: {df.columns}")
                
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")

    final_rows = []
    # Use the same 5 canonical categories as all model modules — not an
    # alphabetical slice of whatever happens to be in the CSVs.
    # Falls back to the top-15 alphabetical list when a category is missing.
    all_categories  = sorted(cat_tokens_map.keys())
    canonical       = [c for c in EXPL_TARGET_CATEGORIES if c in cat_tokens_map]
    remaining       = [c for c in all_categories if c not in canonical]
    target_categories = (canonical + remaining)[:15]
    
    for cat in target_categories:
        all_words = cat_tokens_map[cat]
        if not all_words: continue
            
        counts = Counter(all_words)
        top_15_consensus = [word for word, count in counts.most_common(15)]
        
        final_rows.append({
            'Category': cat,
            'Consolidated_Top_15_Tokens': ", ".join(top_15_consensus),
            'Token_Source_Count': len(all_words)
        })
        
    final_df = pd.DataFrame(final_rows)
    
    out_dir = RESULTS_PATH / "overall_explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Overall_Consolidated_Tokens_Top15.csv"
    
    final_df.to_csv(out_file, index=False)
    logger.info(f"Consolidated Token Report saved: {out_file}")

def generate_overall_charts(n_categories=50):
    logger.info(f"Running Overall Explainability Pipeline for top_{n_categories}_categories...")
    
    df = load_all_metrics(n_categories)
    if not df.empty:
        output_dir = RESULTS_PATH / "overall_explainability"
        output_dir.mkdir(parents=True, exist_ok=True)
        title_suffix = "(Top 15 Categories)" if n_categories == 50 else f"(Top {n_categories})"
        output_file = output_dir / "Overall_XAI_Comparison_Top15.png"
        plot_comparison_chart(df, output_file, title_suffix)
    else:
        logger.error("No metrics found for chart.")

    consolidate_tokens(n_categories)

if __name__ == "__main__":
    n = 50
    if len(sys.argv) > 1:
        try: n = int(sys.argv[1])
        except: pass
    generate_overall_charts(n)
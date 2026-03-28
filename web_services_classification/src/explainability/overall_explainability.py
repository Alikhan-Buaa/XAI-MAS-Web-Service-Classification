"""
Overall Explainability Visualization & Token Consolidation
==========================================================
Aggregates metrics and tokens across all 5 explainability modules
(ML, DL, BERT, DeepSeek, Fusion) and produces:

  1. Overall_XAI_Comparison_Top15.png  — grouped bar chart of
     Fidelity / Jaccard / Stability per model
  2. Overall_Consolidated_Tokens_Top15.csv — consensus top-15 tokens
     per category, voted across all 5 models

Fixes applied vs original
--------------------------
FIX #1  Token column name mismatch
        original: exact-match only on 'consolidated_top_15_tokens' (never fired)
        fixed:    exact-match list now includes the names our explainability
                  modules actually write:
                    ML / DL  → 'Top_15_Tokens'   (lowercased: 'top_15_tokens')
                    BERT / DeepSeek / Fusion → 'Consolidated_Top_Words'
                              (lowercased: 'consolidated_top_words')
        fallback heuristic ('token' / 'word') kept as final safety net.

FIX #2  target_categories used alphabetical slice [:15] of whatever categories
        appeared in the CSV — could differ from our fixed 15 categories.
        fixed:    import TARGET_CATEGORIES from src.utils.utils and filter
                  explicitly so the output always covers the same 15 categories.
"""

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import OVERALL_EXPLAINABILITY_CONFIG, RESULTS_PATH
# TARGET_CATEGORIES lives in utils alongside all other shared explainability constants
from src.utils.utils import TARGET_CATEGORIES
try:
    from src.explainability.xai_comparison import XAIComparison
    _XAI_CMP_AVAILABLE = True
except ImportError:
    _XAI_CMP_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

# Pull constants from config — single source of truth
MODEL_COLORS   = OVERALL_EXPLAINABILITY_CONFIG['model_colors']
METRIC_MAPPING = OVERALL_EXPLAINABILITY_CONFIG['metric_mapping']


# ==============================================================================
#  METRICS LOADING
# ==============================================================================

def load_all_metrics(n_categories: int) -> pd.DataFrame:
    """
    Crawl result directories to find per-model metrics CSVs, aggregate
    per-sample rows to per-model means, and return a tidy DataFrame with
    columns [Model, Fidelity, Jaccard, Stability].
    """
    combined_data = []

    filenames  = OVERALL_EXPLAINABILITY_CONFIG['metrics_files']
    base_res   = RESULTS_PATH

    paths_to_check = [
        base_res / "ml"       / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['ml'],
        base_res / "dl"       / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['dl'],
        base_res / "bert"     / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['bert'],
        base_res / "deepseek" / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['deepseek'],
        base_res / "fusion"   / f"top_{n_categories}_categories" / "explainability" / "metrics" / filenames['fusion'],
    ]

    for path in paths_to_check:
        if not path.exists():
            logger.warning(f"Metrics file not found: {path}")
            continue

        try:
            df   = pd.read_csv(path)
            cols = [c.lower() for c in df.columns]

            # Our CSVs always have a 'model' column written by save_reports()
            if 'model' in cols or 'model_name' in cols:
                model_col = 'model' if 'model' in df.columns else 'model_name'

                # Per-sample rows (is_summary=False) → group to per-model mean
                is_summary = any('mean' in c for c in cols)
                if not is_summary:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df_grouped   = df.groupby(model_col)[numeric_cols].mean().reset_index()
                else:
                    df_grouped = df

                for _, row in df_grouped.iterrows():
                    metrics = extract_metrics_from_row(row)
                    if metrics:
                        metrics['Model'] = normalize_model_name(str(row[model_col]))
                        combined_data.append(metrics)

            else:
                # Fallback for files that omit the model column
                if   "deepseek" in str(path).lower(): current_model = "DeepSeek_7B"
                elif "fusion"   in str(path).lower(): current_model = "DeepSeek_RoBERTa_Fusion"
                else:                                  current_model = "Unknown"

                metrics = extract_metrics_from_row(df.mean(numeric_only=True))
                if metrics:
                    metrics['Model'] = current_model
                    combined_data.append(metrics)

        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    if not combined_data:
        logger.error("No metrics data found across any model.")
        return pd.DataFrame()

    return pd.DataFrame(combined_data)


def normalize_model_name(name: str) -> str:
    """
    Map raw model name strings (as written by save_reports) to the
    canonical names used in MODEL_COLORS and the comparison chart.

    Handles:
      ML:      'LogisticRegression', 'RandomForest', 'XGBoost'
      DL:      'BiLSTM'
      BERT:    'roberta-base', 'roberta-large'
      DeepSeek:'DeepSeek_7B'
      Fusion:  'concat_fusion', 'average_fusion', 'weighted_fusion',
               'gating_fusion'  → all collapse to 'DeepSeek_RoBERTa_Fusion'
               (chart shows one averaged Fusion bar; strategies are compared
                in the per-model ablation plot generated by FusionExplainability)
    """
    n = name.lower()
    if 'logistic'                   in n: return 'LogisticRegression'
    if 'forest'                     in n: return 'RandomForest'
    if 'xgb'                        in n: return 'XGBoost'
    if 'lstm' or 'bilstm'           in n: return 'BiLSTM'
    if 'roberta' in n and 'large'   in n: return 'RoBERTa_Large'
    if 'roberta'                    in n: return 'RoBERTa_Base'
    if 'deepseek' in n and 'fusion' not in n: return 'DeepSeek_7B'
    if 'fusion'                     in n: return 'DeepSeek_RoBERTa_Fusion'
    return name  # return original if no rule matches


def extract_metrics_from_row(row) -> dict | None:
    """
    Extract Fidelity, Jaccard, Stability from a DataFrame row.
    Handles both exact column names and _mean suffixes.
    Returns None if no metric columns are found.
    """
    row_dict = {k.lower(): v for k, v in row.items()}

    def get_val(keys):
        for k in keys:
            if k in row_dict:
                try:
                    return float(row_dict[k])
                except (ValueError, TypeError):
                    return 0.0
        return None

    metrics = {}
    j = get_val(['jaccard',   'jaccard_mean'])
    f = get_val(['fidelity',  'fidelity_mean'])
    s = get_val(['stability', 'stability_mean'])

    if j is not None: metrics['Jaccard']   = j
    if f is not None: metrics['Fidelity']  = f
    if s is not None: metrics['Stability'] = s

    return metrics if metrics else None


# ==============================================================================
#  COMPARISON CHART
# ==============================================================================

def plot_comparison_chart(df: pd.DataFrame, output_path: Path,
                          title_suffix: str = "") -> None:
    """
    Grouped bar chart of Fidelity / Jaccard / Stability across all models.
    Model order and colours come from OVERALL_EXPLAINABILITY_CONFIG['model_colors'].
    """
    desired_metrics = ['Jaccard', 'Fidelity', 'Stability']
    df_long = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    df_long = df_long[df_long['Metric'].isin(desired_metrics)]

    existing_models = [m for m in MODEL_COLORS if m in df['Model'].unique()]
    x     = np.arange(len(desired_metrics))
    width = 0.10

    fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')

    for multiplier, model in enumerate(existing_models):
        model_data = df_long[df_long['Model'] == model]
        scores = []
        for metric in desired_metrics:
            vals = model_data[model_data['Metric'] == metric]['Score'].values
            scores.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)

        offset = width * multiplier
        rects  = ax.bar(x + offset, scores, width,
                        label=model,
                        color=MODEL_COLORS.get(model, '#888888'))
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9, fontweight='bold')

    if existing_models:
        center_offset = (width * len(existing_models)) / 2 - width / 2
        ax.set_xticks(x + center_offset)
    else:
        ax.set_xticks(x)

    ax.set_xticklabels(
        [METRIC_MAPPING.get(m, m) for m in desired_metrics],
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        f'XAI Metrics Comparison across Models {title_suffix}',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
              title="Models", borderaxespad=0.)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison chart saved → {output_path}")
    plt.close()


# ==============================================================================
#  TOKEN CONSOLIDATION
# ==============================================================================

def _find_token_column(df: pd.DataFrame) -> str | None:
    """
    Find the token column in a report DataFrame.

    Exact-match priority list (lowercased column names):
      1. 'top_15_tokens'          — written by ML / DL save_reports()
      2. 'consolidated_top_words' — written by BERT / DeepSeek / Fusion
      3. 'consolidated_top_15_tokens' — legacy / alternative name
    Heuristic fallback: any column name containing 'token', 'word', 'feature'.
    """
    cols = list(df.columns)  # already lowercased by caller

    exact_matches = [
        'top_15_tokens',           # ML, DL
        'consolidated_top_words',  # BERT, DeepSeek, Fusion
        'consolidated_top_15_tokens',  # legacy
    ]
    for name in exact_matches:
        if name in cols:
            return name

    # Heuristic fallback
    for col in cols:
        if any(kw in col for kw in ('token', 'word', 'feature')):
            return col

    return None


def clean_token_string(token_str: str) -> list[str]:
    """Parse a comma-separated token string into a clean list."""
    if not isinstance(token_str, str):
        return []
    token_str = token_str.replace('[', '').replace(']', '').replace("'", "")
    return [t.strip() for t in token_str.split(',') if t.strip()]


def consolidate_tokens(n_categories: int) -> None:
    """
    Load per-model token CSVs, merge token lists per category, rank by
    cross-model frequency, and save Overall_Consolidated_Tokens_Top15.csv.

    Uses TARGET_CATEGORIES (15 fixed categories from src.utils.utils) so the
    output always covers exactly the same categories as the individual models.
    """
    logger.info("Starting Token Consolidation…")

    filenames = OVERALL_EXPLAINABILITY_CONFIG['token_files']
    base_path = RESULTS_PATH

    file_map = {
        'ML':       base_path / "ml"       / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['ml'],
        'DL':       base_path / "dl"       / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['dl'],
        'BERT':     base_path / "bert"     / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['bert'],
        'DeepSeek': base_path / "deepseek" / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['deepseek'],
        'Fusion':   base_path / "fusion"   / f"top_{n_categories}_categories" / "explainability" / "reports" / filenames['fusion'],
    }

    # cat → [token, token, ...] aggregated across all 5 models
    cat_tokens_map: dict[str, list[str]] = defaultdict(list)

    for model_name, path in file_map.items():
        if not path.exists():
            logger.warning(f"  Missing token file for {model_name}: {path}")
            continue

        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]   # normalise to lowercase

            cat_col = next((c for c in df.columns if 'category' in c), None)
            tok_col = _find_token_column(df)

            if cat_col and tok_col:
                logger.info(f"  {model_name}: using columns '{cat_col}' / '{tok_col}'")
                for _, row in df.iterrows():
                    cat    = str(row[cat_col]).strip()
                    tokens = clean_token_string(str(row[tok_col]))
                    cat_tokens_map[cat].extend(tokens)
            else:
                logger.warning(
                    f"  {model_name}: could not find category/token columns. "
                    f"Available: {list(df.columns)}"
                )

        except Exception as e:
            logger.error(f"  Error processing {model_name} tokens: {e}")

    # Build output — use TARGET_CATEGORIES so we always cover the fixed 15
    final_rows = []
    for cat in TARGET_CATEGORIES:
        all_words = cat_tokens_map.get(cat, [])
        if not all_words:
            logger.warning(f"  No tokens found for category: {cat}")
            final_rows.append({
                'Category': cat,
                'Consolidated_Top_15_Tokens': 'N/A',
                'Token_Source_Count': 0,
            })
            continue

        top_15 = [word for word, _ in Counter(all_words).most_common(15)]
        final_rows.append({
            'Category':                  cat,
            'Consolidated_Top_15_Tokens': ', '.join(top_15),
            'Token_Source_Count':         len(all_words),
        })

    out_dir  = RESULTS_PATH / "overall_explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Overall_Consolidated_Tokens_Top15.csv"

    pd.DataFrame(final_rows).to_csv(out_file, index=False)
    logger.info(f"Consolidated Token Report saved → {out_file}")


# ==============================================================================
#  MAIN PIPELINE
# ==============================================================================

def generate_overall_charts(n_categories: int = 50) -> None:
    """Run the full overall explainability pipeline."""
    logger.info(f"Overall Explainability Pipeline — top_{n_categories}_categories")

    # 1. Metrics comparison chart
    df = load_all_metrics(n_categories)
    if not df.empty:
        out_dir = RESULTS_PATH / "overall_explainability"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix   = "(Top 15 Categories)" if n_categories == 50 else f"(Top {n_categories})"
        out_file = out_dir / "Overall_XAI_Comparison_Top15.png"
        plot_comparison_chart(df, out_file, suffix)
    else:
        logger.error("No metrics loaded — chart skipped.")

    # 2. Token consolidation
    consolidate_tokens(n_categories)

    # 3. Cross-model XAI comparison charts
    if _XAI_CMP_AVAILABLE:
        logger.info("Running cross-model XAI comparison charts…")
        try:
            XAIComparison(n_categories=n_categories).run_all()
        except Exception as _e:
            logger.warning(f"XAI comparison charts failed: {_e}")
    else:
        logger.warning("xai_comparison module not available — skipping comparison charts.")

    logger.info("Overall Explainability Pipeline complete.")


if __name__ == "__main__":
    n = 50
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    generate_overall_charts(n)

"""
shared_samples.py  —  src/explainability/shared_samples.py
===========================================================
Single reproducible set of (row_index, category_name) pairs used by ALL five
explainability modules (ML · DL · BERT · DeepSeek · Fusion) to explain the
exact same 5 test-set rows → SHAP / LIME outputs are directly comparable.

Design
------
NO hardcoded row indices. get_shared_samples() SCANS the live test_df at
call time, finding the FIRST available row for each of the 5 fixed categories.
This is robust to any data-split change — no stale index maintenance needed.

Category list is defined ONCE in config.py → EXPLAINABILITY_CONFIG['expl_categories'].
shared_samples.py reads it from there. No duplication anywhere.

5 categories × 1 sample = 5 total rows.

Public API
----------
    indices = get_shared_samples(
        test_df      = test_df,
        class_labels = class_labels,
        n_categories = 50,
        results_root = results_root,
    )
    # returns list[tuple[int, str]] — [(row_idx, category_name), ...], length = 5

    FIXED_CATEGORIES        # list[str] — the 5 category names (from config)
    N_SAMPLES_PER_CATEGORY  # int — 1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Single source of truth: category list comes from config ───────────────────
try:
    from src.config import EXPLAINABILITY_CONFIG, PREPROCESSING_CONFIG, DATA_CONFIG
    _cfg              = EXPLAINABILITY_CONFIG
    _PROC_DATA_TEMPLATE = str(PREPROCESSING_CONFIG.get("processed_data", ""))
    _TARGET_COL       = DATA_CONFIG.get("target_column", "Service Classification")
except ImportError:
    _cfg              = {}
    _PROC_DATA_TEMPLATE = ""
    _TARGET_COL       = "Service Classification"

logger = logging.getLogger(__name__)


# ==============================================================================
#  PUBLIC CONSTANTS  —  derived from config, not hardcoded here
# ==============================================================================

FIXED_CATEGORIES: List[str] = _cfg.get("expl_categories", [
    # Fallback if config cannot be imported (should not happen in normal use)
    "Payments", "Messaging", "Social", "Storage", "eCommerce",
])

N_SAMPLES_PER_CATEGORY: int = _cfg.get("n_samples_per_category", 1)


# ==============================================================================
#  PUBLIC FUNCTION
# ==============================================================================

def get_shared_samples(
    test_df: pd.DataFrame,
    class_labels: List[str],
    n_categories: int,
    results_root: Path,
) -> List[Tuple[int, str]]:
    """
    Return N_SAMPLES_PER_CATEGORY × FIXED_CATEGORIES (row_index, category_name) tuples.

    Scans test_df live: finds the first row whose 'Service Classification'
    column matches each category in FIXED_CATEGORIES. Falls back to
    encoded_label scan if the text column is absent. No hardcoded indices —
    always correct regardless of data-split changes.

    Parameters
    ----------
    test_df      : test-split DataFrame
    class_labels : list[str] — used to resolve encoded_label → name if needed
    n_categories : int — used for companion JSON/CSV filename only
    results_root : Path — companion JSON + CSV written here for traceability

    Returns
    -------
    list of (row_index: int, category_name: str)
    Order follows FIXED_CATEGORIES: Payments → Messaging → Social → Storage → eCommerce
    """
    df = test_df.reset_index(drop=True)

    if "encoded_label" not in df.columns:
        raise RuntimeError(
            "[shared_samples] 'encoded_label' column not found in test_df. "
            f"Available: {list(df.columns)}"
        )

    # Build encoded_label lookup from class_labels
    label_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(class_labels)}
    has_target_col = _TARGET_COL in df.columns

    indices:  List[Tuple[int, str]] = []
    rows_out: List[dict]            = []

    for cat in FIXED_CATEGORIES:
        row_i: Optional[int] = None

        # Primary: match by Service Classification text column
        if has_target_col:
            matches = df[df[_TARGET_COL] == cat]
            if not matches.empty:
                row_i = int(matches.index[0])

        # Fallback: match by encoded_label
        if row_i is None:
            enc_lbl = label_to_id.get(cat)
            if enc_lbl is not None:
                matches = df[df["encoded_label"] == enc_lbl]
                if not matches.empty:
                    row_i = int(matches.index[0])

        if row_i is None:
            logger.warning(f"  [shared_samples] '{cat}' not found in test_df — skipped.")
            continue

        enc_lbl_actual = int(df.iloc[row_i]["encoded_label"])
        indices.append((row_i, cat))
        rows_out.append({
            "category":      cat,
            "encoded_label": enc_lbl_actual,
            "row_index":     row_i,
            "text_preview":  str(df.iloc[row_i].get("cleaned_text", ""))[:80],
        })
        logger.info(f"  [shared_samples] '{cat}' → row {row_i} (label={enc_lbl_actual})")

    logger.info(
        f"  [shared_samples] {len(indices)}/{len(FIXED_CATEGORIES)} categories "
        f"selected ({N_SAMPLES_PER_CATEGORY} sample each)"
    )

    # ── Write companion files ──────────────────────────────────────────────────
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    csv_path = results_root / f"shared_sample_index_top{n_categories}.csv"
    try:
        pd.DataFrame(rows_out).to_csv(csv_path, index=False)
        logger.info(f"  [shared_samples] CSV  → {csv_path}")
    except Exception as exc:
        logger.warning(f"  [shared_samples] CSV write failed: {exc}")

    json_path = results_root / f"shared_sample_index_top{n_categories}.json"
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({r["category"]: r for r in rows_out}, fh, indent=2, ensure_ascii=False)
        logger.info(f"  [shared_samples] JSON → {json_path}")
    except Exception as exc:
        logger.warning(f"  [shared_samples] JSON write failed: {exc}")

    return indices

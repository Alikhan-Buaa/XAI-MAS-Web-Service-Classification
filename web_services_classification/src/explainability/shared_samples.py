"""
shared_samples.py  —  src/explainability/shared_samples.py
===========================================================
Single reproducible set of (row_index, category_name) pairs used by ALL five
explainability modules (ML · DL · BERT · DeepSeek · Fusion) to explain the
exact same 15 test-set rows → SHAP / LIME outputs are directly comparable.

Design
------
Row indices are HARDCODED from the real test.csv
(data/splits/top_50_categories/test.csv, random_state=42 split).

5 categories × 3 samples = 15 total rows.

Categories — chosen for maximum semantic diversity + large test counts:
  Payments   (label 30) — financial transactions        60 test rows
  Messaging  (label 25) — communication                 60 test rows
  Social     (label 39) — social networks               60 test rows
  Storage    (label 42) — data / cloud storage          40 test rows
  eCommerce  (label 49) — commerce / retail             60 test rows

Hardcoded index (every row_index validated against encoded_label in test.csv):
  Category    label   row_indices
  ─────────   ─────   ───────────
  Payments      30    28, 64, 100
  Messaging     25    49, 141, 263
  Social        39    14, 19, 25
  Storage       42    36, 132, 139
  eCommerce     49    70, 113, 161

Runtime validation
------------------
get_shared_samples() checks encoded_label for every hardcoded row at call
time. Logs ERROR if a mismatch is found (signals data-split change). Raises
RuntimeError only if more than half the rows fail validation.

Public API
----------
    indices = get_shared_samples(
        test_df      = test_df,       # pd.DataFrame with encoded_label column
        class_labels = class_labels,  # list[str] — used only for logging
        n_categories = 50,
        results_root = RESULTS_PATH,  # pathlib.Path
    )
    # returns list[tuple[int, str]]  →  [(row_idx, category_name), ...]
    # length = 15  (5 categories × 3 samples)

    FIXED_CATEGORIES        # list[str] — the 5 category names in order
    N_SAMPLES_PER_CATEGORY  # int — 3
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from src.config import PREPROCESSING_CONFIG
    _PROC_DATA_TEMPLATE = str(PREPROCESSING_CONFIG.get("processed_data", ""))
except ImportError:
    _PROC_DATA_TEMPLATE = ""

logger = logging.getLogger(__name__)


# ==============================================================================
#  PUBLIC CONSTANTS
# ==============================================================================

FIXED_CATEGORIES: List[str] = [
    "Payments",   # label 30
    "Messaging",  # label 25
    "Social",     # label 39
    "Storage",    # label 42
    "eCommerce",  # label 49
]

N_SAMPLES_PER_CATEGORY: int = 3


# ==============================================================================
#  HARDCODED INDEX
#  Validated from real test.csv — DO NOT CHANGE unless the split is regenerated.
#  Each entry: (row_index_in_test_csv, expected_encoded_label)
# ==============================================================================

_HARDCODED: Dict[str, List[Tuple[int, int]]] = {
    "Payments":  [(28,  30), (64,  30), (100, 30)],
    "Messaging": [(49,  25), (141, 25), (263, 25)],
    "Social":    [(14,  39), (19,  39), (25,  39)],
    "Storage":   [(36,  42), (132, 42), (139, 42)],
    "eCommerce": [(70,  49), (113, 49), (161, 49)],
}


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
    Return the canonical list of (row_index, category_name) tuples.

    All five model families call this and receive identical results —
    ensuring SHAP / LIME explanations cover the exact same 15 rows.

    Parameters
    ----------
    test_df      : test-split DataFrame (must contain 'encoded_label' column)
    class_labels : list[str] — used only for log context, not for selection
    n_categories : int — used only for companion JSON/CSV filename
    results_root : Path — companion JSON + CSV written here for traceability

    Returns
    -------
    list of (row_index: int, category_name: str)
    Order: Payments → Messaging → Social → Storage → eCommerce (3 rows each)
    Total length: 15

    Raises
    ------
    RuntimeError if encoded_label column is absent OR if more than half
    the hardcoded rows fail ground-truth validation (split change detected).
    """
    df = test_df.reset_index(drop=True)

    # ── Ground-truth validation ───────────────────────────────────────────────
    if "encoded_label" not in df.columns:
        raise RuntimeError(
            "[shared_samples] 'encoded_label' column not found in test_df. "
            f"Available: {list(df.columns)}"
        )

    mismatches = 0
    total_checks = sum(len(v) for v in _HARDCODED.values())

    for cat, entries in _HARDCODED.items():
        for row_i, expected_lbl in entries:
            if row_i >= len(df):
                logger.warning(
                    f"  [shared_samples] row {row_i} ({cat}) out of bounds "
                    f"(test_df has {len(df)} rows). Split may differ."
                )
                mismatches += 1
                continue
            actual = int(df.iloc[row_i]["encoded_label"])
            if actual != expected_lbl:
                logger.error(
                    f"  [shared_samples] MISMATCH — {cat} row {row_i}: "
                    f"expected label {expected_lbl}, got {actual}. "
                    f"Data split may have changed."
                )
                mismatches += 1

    if mismatches > total_checks // 2:
        raise RuntimeError(
            f"[shared_samples] {mismatches}/{total_checks} ground-truth checks "
            "failed — the test split appears to have changed. "
            "Re-generate the hardcoded index."
        )

    if mismatches:
        logger.warning(
            f"  [shared_samples] {mismatches}/{total_checks} validation "
            "warning(s). Proceeding with hardcoded index."
        )
    else:
        logger.info(
            f"  [shared_samples] Ground-truth: all {total_checks} rows validated ✓"
        )

    # ── Build flat output list ────────────────────────────────────────────────
    indices: List[Tuple[int, str]] = [
        (row_i, cat)
        for cat in FIXED_CATEGORIES
        for row_i, _ in _HARDCODED[cat]
    ]

    logger.info(
        f"  [shared_samples] {len(indices)} samples returned "
        f"({len(FIXED_CATEGORIES)} categories × {N_SAMPLES_PER_CATEGORY} each)"
    )

    # ── Copy pre-built explainability_test_samples files to results_root ──────
    # During preprocessing, data_preprocessing.py writes:
    #   data/processed/top_{n}_categories/explainability_test_samples.csv
    #   data/processed/top_{n}_categories/explainability_test_samples.json
    # We copy them here so every model's results directory has easy reference.
    if _PROC_DATA_TEMPLATE:
        _src_dir = Path(_PROC_DATA_TEMPLATE.format(n=n_categories))
        _dst_dir = Path(results_root)
        _dst_dir.mkdir(parents=True, exist_ok=True)
        for _fname in ("explainability_test_samples.csv",
                       "explainability_test_samples.json"):
            _src_file = _src_dir / _fname
            _dst_file = _dst_dir / _fname
            if _src_file.exists() and not _dst_file.exists():
                import shutil as _shutil
                _shutil.copy2(_src_file, _dst_file)
                logger.info(
                    f"  [shared_samples] copied {_fname} → {_dst_dir}"
                )

    # ── Write companion files ─────────────────────────────────────────────────
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    json_path = results_root / f"shared_sample_index_top{n_categories}.json"
    try:
        payload = {}
        for cat, entries in _HARDCODED.items():
            payload[cat] = [
                {
                    "row_index":     row_i,
                    "encoded_label": lbl,
                    "category":      cat,
                    "text_preview":  str(df.iloc[row_i].get("cleaned_text", ""))[:80]
                                     if row_i < len(df) else "",
                }
                for row_i, lbl in entries
            ]
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info(f"  [shared_samples] JSON → {json_path}")
    except Exception as exc:
        logger.warning(f"  [shared_samples] JSON write failed: {exc}")

    csv_path = results_root / f"shared_sample_index_top{n_categories}.csv"
    try:
        rows_flat = [
            {
                "category":      cat,
                "encoded_label": lbl,
                "row_index":     row_i,
                "text_preview":  str(df.iloc[row_i].get("cleaned_text", ""))[:80]
                                 if row_i < len(df) else "",
            }
            for cat, entries in _HARDCODED.items()
            for row_i, lbl in entries
        ]
        pd.DataFrame(rows_flat).to_csv(csv_path, index=False)
        logger.info(f"  [shared_samples] CSV  → {csv_path}")
    except Exception as exc:
        logger.warning(f"  [shared_samples] CSV write failed: {exc}")

    return indices

"""
explainability_utils.py
=======================
Single source of truth for ALL explainability logic shared by the five
model modules: ML, DL, BERT, DeepSeek, Fusion.

    from src.utils.explainability_utils import (
        STOPWORDS, TARGET_CATEGORIES, FALLBACK_LABELS,
        EXPL_TARGET_CATEGORIES, EXPL_LABEL_IDS,
        load_class_labels,
        get_shared_samples, N_SAMPLES_PER_CATEGORY,
        top15_tokens, plot_bar, compute_metrics,
        build_shap_background,
        run_global_shap, run_global_lime,
        run_beeswarm, run_waterfall,
    )

Sections
--------
A. Constants        — STOPWORDS · TARGET_CATEGORIES · FALLBACK_LABELS
B. Labels           — load_class_labels()
C. Shared Samples   — get_shared_samples()   HARDCODED from real test.csv
                      5 domain-diverse categories × 3 validated samples = 15 rows
D. Token helpers    — top15_tokens()
E. Plot             — plot_bar()
F. Metrics          — compute_metrics()
G. SHAP background  — build_shap_background()
H. Global SHAP      — run_global_shap()   → PNG + CSV
I. Global LIME      — run_global_lime()   → PNG + CSV
J. Beeswarm         — run_beeswarm()      → PNG + CSV
K. Waterfall        — run_waterfall()     → PNG + CSV
"""

import json
import logging
import pickle
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import spearmanr

import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config import DATA_PATH, PREPROCESSING_CONFIG
except ImportError:
    DATA_PATH            = Path("data")
    PREPROCESSING_CONFIG = {}

_log = logging.getLogger("explainability_utils")

# ==============================================================================
#  A. CONSTANTS
# ==============================================================================

# ── STOPWORDS ─────────────────────────────────────────────────────────────────
# Built once at import time from PREPROCESSING_CONFIG['custom_stopwords']
# (the 79 function words that clean_text() already removed from cleaned_text)
# PLUS a small set of noise tokens that can still slip through after
# lemmatisation / BPE tokenisation.
#
# NEVER add domain words ('api','service','data','platform','cloud','tool' …).
# They survived preprocessing intentionally and must appear in LIME/SHAP output.

def _build_stopwords() -> frozenset:
    """Combine preprocessing custom_stopwords with post-tokenisation noise."""
    base = set(PREPROCESSING_CONFIG.get('custom_stopwords', []))
    noise = {
        # URL / domain residuals that regex cleaning sometimes misses
        'http', 'https', 'www', 'com', 'org', 'net', 'gov', 'edu', 'io',
        # Dataset-specific junk abbreviations after lemmatisation
        'abn', 'eur', 'ma', 'acus', 'id', 'inc', 'json',
        # BPE subword fragments from RoBERTa / DeepSeek tokenisers
        's', 't', 're', 've', 'm', 'll', 'd',
        '##s', '##ing', '##ed', '##tion', '##ly', '##y',
    }
    return frozenset(base | noise)


STOPWORDS: frozenset = _build_stopwords()

# ── TARGET_CATEGORIES ─────────────────────────────────────────────────────────
# 15 fixed categories used for local explanation and cross-model comparison.
# Same list across all 5 explainability modules — defined once here.

TARGET_CATEGORIES: List[str] = [
    "Advertising", "Analytics", "Application Development", "Backend",
    "Banking", "Bitcoin", "Chat", "Cloud", "Data", "Database",
    "Domains", "Education", "Email", "Enterprise", "Entertainment",
]

# ── FALLBACK_LABELS ───────────────────────────────────────────────────────────
# Hardcoded id → category name mapping for all 50 categories.
# Used when the YAML label file cannot be loaded.

FALLBACK_LABELS: Dict[int, str] = {
    0:  "Advertising",            1:  "Analytics",
    2:  "Application Development", 3:  "Backend",
    4:  "Banking",                5:  "Bitcoin",
    6:  "Chat",                   7:  "Cloud",
    8:  "Data",                   9:  "Database",
    10: "Domains",                11: "Education",
    12: "Email",                  13: "Enterprise",
    14: "Entertainment",          15: "Events",
    16: "File Sharing",           17: "Financial",
    18: "Games",                  19: "Government",
    20: "Images",                 21: "Internet of Things",
    22: "Mapping",                23: "Media",
    24: "Medical",                25: "Messaging",
    26: "Music",                  27: "News Services",
    28: "Office",                 29: "Other",
    30: "Payments",               31: "Photos",
    32: "Project Management",     33: "Real Estate",
    34: "Reference",              35: "Science",
    36: "Search",                 37: "Security",
    38: "Shipping",               39: "Social",
    40: "Sports",                 41: "Stocks",
    42: "Storage",                43: "Telephony",
    44: "Tools",                  45: "Transportation",
    46: "Travel",                 47: "Video",
    48: "Weather",                49: "eCommerce",
}


# ── load_class_labels ─────────────────────────────────────────────────────────

def load_class_labels(n_categories: int) -> List[str]:
    """
    Load the ordered list of category name strings for a given experiment size.

    Resolution order:
      1. YAML  — data/processed/labels_top_{n}_categories.yaml
                 written by data_preprocessing.save_label_mapping()
                 supports both list format and dict-with-id_to_label format
      2. Pickle — data/processed/top_{n}_categories/label_encoder.pkl
                 (sklearn LabelEncoder saved by the BERT pipeline)
      3. FALLBACK_LABELS — hardcoded 50-category safety net, always succeeds

    Uses yaml.safe_load and joblib directly (avoids circular import with utils.py).
    """
    yaml_path = DATA_PATH / "processed" / f"labels_top_{n_categories}_categories.yaml"
    try:
        with open(yaml_path, 'r', encoding='utf-8') as _f:
            d = yaml.safe_load(_f)
        if isinstance(d, list):
            return d
        if isinstance(d, dict) and 'id_to_label' in d:
            m = d['id_to_label']
            return [str(m[k]) for k in sorted(m.keys(), key=int)]
    except Exception as e:
        _log.warning(f"  load_class_labels: YAML warning ({yaml_path.name}): {e}")

    le_path = DATA_PATH / "processed" / f"top_{n_categories}_categories" / "label_encoder.pkl"
    try:
        import joblib as _jl
        le = _jl.load(le_path)
        return list(le.classes_)
    except Exception:
        pass

    _log.warning(
        f"  load_class_labels: using hardcoded fallback for {n_categories} categories"
    )
    return [FALLBACK_LABELS.get(i, f"Class_{i}") for i in range(n_categories)]




# ==============================================================================
#  C. SHARED SAMPLES  —  hardcoded from real test.csv (top_50_categories)
#
#  Why these 5 categories?
#  ─────────────────────────────────────────────────────────────────────────────
#  Payments  (label 30, 60 test rows) — Finance/Fintech
#    Keywords: payment, billing, card, gateway, transaction
#  Social    (label 39, 60 test rows) — Social Media / Community
#    Keywords: community, sharing, engagement, social, network
#  Cloud     (label  7, 50 test rows) — Tech Infrastructure
#    Keywords: hosting, platform, deploy, server, instance
#  Medical   (label 24, 45 test rows) — Healthcare / Life Sciences
#    Keywords: health, drug, disease, clinical, patient
#  eCommerce (label 49, 60 test rows) — Commerce / Retail
#    Keywords: store, product, cart, checkout, merchant
#
#  These 5 have maximally distinct vocabularies → SHAP/LIME explanations show
#  clearly different top tokens per category → best for cross-model comparison.
#  Larger test counts → more stable explainability estimates.
#
#  All 15 row_index values are validated against encoded_label in test.csv.
#  They are fixed permanently — no runtime scanning needed.
#
#  Hardcoded index  (row_index → position in test.csv, 0-based):
#
#   Category   label  row_index  text_preview (first 60 chars)
#   ─────────  ─────  ─────────  ──────────────────────────────────────────────
#   Payments      30        28   liqpay api let developer using managing payment
#   Payments      30        64   ambercart ecommerce service allows user add ecom
#   Payments      30       100   espago polish payment gateway ecommerce process
#   Social        39        14   socialfree api provides free community unbound p
#   Social        39        19   socialdiscovery api manages discovery community
#   Social        39        25   socialreactive api manages reactive community re
#   Cloud          7        16   nodejitsu cloud hosting platform service node j
#   Cloud          7        87   crossrider multiplatform cloud based extension d
#   Cloud          7       121   packet api return data bare metal cloud allowing
#   Medical       24       233   toxnet toxicology data network cluster database
#   Medical       24       239   api bioinformatics information application datab
#   Medical       24       250   medlineplus service user drug disease service we
#   eCommerce     49        70   simplero api allows developer create online store
#   eCommerce     49       113   listselltrade merchant site allows user access on
#   eCommerce     49       161   commercejs powerful cross device ecommerce api d
# ==============================================================================

# 5 explainability target categories (ordered by label id for reproducibility)
EXPL_TARGET_CATEGORIES: List[str] = [
    "Payments", "Social", "Cloud", "Medical", "eCommerce",
]

# Encoded label id for each — matches labels_top_50_categories.yaml
EXPL_LABEL_IDS: Dict[str, int] = {
    "Payments":  30,
    "Social":    39,
    "Cloud":      7,
    "Medical":   24,
    "eCommerce": 49,
}

# Number of samples per category in the shared index
N_SAMPLES_PER_CATEGORY: int = 3

# Backward-compat alias — model files that do:
#   from src.utils.explainability_utils import FIXED_CATEGORIES
# continue to work alongside shared_samples.py
FIXED_CATEGORIES: List[str] = EXPL_TARGET_CATEGORIES

# ── HARDCODED INDEX ────────────────────────────────────────────────────────────
# Directly from test.csv (top_50_categories split, random_state=42).
# Keys are category names. Each entry contains the test.csv 0-based row_index,
# the confirmed encoded_label (ground truth), and text for human inspection.
# DO NOT CHANGE unless the data split is regenerated.
_SHARED_SAMPLE_INDEX: Dict[str, List[Dict]] = {
    "Payments": [
        {"row_index": 28,  "encoded_label": 30,
         "text_preview": "liqpay api let developer using managing payment billing system creating custom p",
         "description":  "The LiqPAY API lets developers using their for managing payment and billing systems."},
        {"row_index": 64,  "encoded_label": 30,
         "text_preview": "ambercart ecommerce service allows user add ecommerce functionality website appl",
         "description":  "AmberCart is an eCommerce service that allows users to add eCommerce functionality to websites."},
        {"row_index": 100, "encoded_label": 30,
         "text_preview": "espago polish payment gateway ecommerce process online card payment method espag",
         "description":  "Espago is a Polish payment gateway for eCommerce that processes online card payment methods."},
    ],
    "Social": [
        {"row_index": 14,  "encoded_label": 39,
         "text_preview": "socialfree api provides free community unbound participation open involvement fr",
         "description":  "SocialFree API provides free communities unbound participation open involvement free engagement."},
        {"row_index": 19,  "encoded_label": 39,
         "text_preview": "socialdiscovery api manages discovery community new experience sharing curiosity",
         "description":  "SocialDiscovery API manages discovery communities new experience sharing curiosity-driven."},
        {"row_index": 25,  "encoded_label": 39,
         "text_preview": "socialreactive api manages reactive community responsive engagement adaptive par",
         "description":  "SocialReactive API manages reactive communities responsive engagement adaptive participation."},
    ],
    "Cloud": [
        {"row_index": 16,  "encoded_label": 7,
         "text_preview": "nodejitsu cloud hosting platform service node j developer platform let developer",
         "description":  "Nodejitsu is a cloud hosting platform-as-a-service for node.js developers."},
        {"row_index": 87,  "encoded_label": 7,
         "text_preview": "crossrider multiplatform cloud based extension development tool creation interop",
         "description":  "Crossrider is a multiplatform, cloud based extension development tool."},
        {"row_index": 121, "encoded_label": 7,
         "text_preview": "packet api return data bare metal cloud allowing developer access hardware featu",
         "description":  "The Packet API returns data of bare-metal cloud, allowing developers to access hardware features."},
    ],
    "Medical": [
        {"row_index": 233, "encoded_label": 24,
         "text_preview": "toxnet toxicology data network cluster database covering toxicology hazardous ch",
         "description":  "TOXNET is a cluster of databases covering toxicology, hazardous chemicals and related areas."},
        {"row_index": 239, "encoded_label": 24,
         "text_preview": "api bioinformatics information information application database access developer",
         "description":  "Bioinformatics API — information that applications database access for developers."},
        {"row_index": 250, "encoded_label": 24,
         "text_preview": "medlineplus service user drug disease service wellness api xml connect api let e",
         "description":  "MedlinePlus service provides users drug and disease information via API."},
    ],
    "eCommerce": [
        {"row_index": 70,  "encoded_label": 49,
         "text_preview": "simplero api allows developer create online store infoproducts online course dow",
         "description":  "The Simplero API allows developers to create online stores for infoproducts such as online courses."},
        {"row_index": 113, "encoded_label": 49,
         "text_preview": "listselltrade merchant site allows user access online auction fixed price listin",
         "description":  "ListSellTrade is a merchant site that allows users to access online auctions and fixed price listings."},
        {"row_index": 161, "encoded_label": 49,
         "text_preview": "commercejs powerful cross device ecommerce api developer designer commercejs all",
         "description":  "CommerceJS is a powerful, cross device eCommerce API for developers and designers."},
    ],
}


def _shared_index_json_path(results_root: Path, n_categories: int) -> Path:
    return Path(results_root) / f"shared_expl_samples_{n_categories}.json"


def get_shared_samples(
    test_df: pd.DataFrame,
    n_categories: int,
    results_root: Path,
    n_per_cat: int = N_SAMPLES_PER_CATEGORY,
    force_rebuild: bool = False,
) -> List[Tuple[int, str]]:
    """
    Return the canonical list of (row_index, category_name) tuples that every
    explainability module must iterate over — 15 rows, same for all 5 models.

    The index is hardcoded from the actual test.csv (top_50_categories,
    random_state=42).  On first call it is also written to a JSON file under
    results_root so other tools / notebooks can read it.

    Parameters
    ----------
    test_df       : test split DataFrame — used ONLY for ground-truth validation
                    and to attach full_text at write time. The row_index values
                    are already fixed; test_df is not scanned for selection.
    n_categories  : experiment size (50) — used for JSON filename only
    results_root  : directory for the companion JSON / CSV files
    n_per_cat     : respected for slicing (max = 3, hardcoded limit)
    force_rebuild : if True, re-write the JSON/CSV files even if they exist

    Returns
    -------
    List of (row_index: int, category_name: str) — length 5 × n_per_cat.
    Order: Payments, Social, Cloud, Medical, eCommerce.
    """
    n_per_cat = min(n_per_cat, N_SAMPLES_PER_CATEGORY)
    json_path = _shared_index_json_path(results_root, n_categories)

    # ── Always validate ground truth against test_df ─────────────────────────
    if "encoded_label" in test_df.columns:
        for cat, entries in _SHARED_SAMPLE_INDEX.items():
            expected_lbl = EXPL_LABEL_IDS[cat]
            for e in entries[:n_per_cat]:
                ri = e["row_index"]
                if ri < len(test_df):
                    actual_lbl = int(test_df.iloc[ri]["encoded_label"])
                    if actual_lbl != expected_lbl:
                        _log.error(
                            f"  get_shared_samples: GROUND TRUTH MISMATCH — "
                            f"{cat} row {ri}: expected label {expected_lbl}, "
                            f"got {actual_lbl}. Data split may have changed."
                        )
                    else:
                        _log.debug(f"  OK: {cat} row {ri} label={actual_lbl}")

    # ── Write companion files if needed ──────────────────────────────────────
    results_root = Path(results_root)
    if not json_path.exists() or force_rebuild:
        results_root.mkdir(parents=True, exist_ok=True)

        # Build enriched index with full_text from test_df
        enriched = {}
        for cat, entries in _SHARED_SAMPLE_INDEX.items():
            enriched[cat] = []
            for e in entries[:n_per_cat]:
                entry = dict(e)
                ri = e["row_index"]
                if ri < len(test_df):
                    entry["full_text"] = str(test_df.iloc[ri].get("cleaned_text", ""))
                enriched[cat].append(entry)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(enriched, fh, indent=2, ensure_ascii=False)
        _log.info(f"  get_shared_samples: JSON saved → {json_path}")

        # CSV
        rows_flat = [e for entries in enriched.values() for e in entries]
        if rows_flat:
            csv_path = results_root / f"shared_expl_samples_{n_categories}.csv"
            pd.DataFrame(rows_flat).to_csv(csv_path, index=False)
            _log.info(f"  get_shared_samples: CSV saved  → {csv_path}")
    else:
        _log.info(f"  get_shared_samples: using existing index ({json_path.name})")

    # ── Return flat list of (row_index, category) ────────────────────────────
    result: List[Tuple[int, str]] = []
    for cat in EXPL_TARGET_CATEGORIES:
        for entry in _SHARED_SAMPLE_INDEX.get(cat, [])[:n_per_cat]:
            result.append((entry["row_index"], cat))

    _log.info(
        f"  get_shared_samples: returning {len(result)} samples "
        f"({len(EXPL_TARGET_CATEGORIES)} categories × {n_per_cat} each)"
    )
    return result


# ==============================================================================
#  D-K.  TOKEN / PLOT / METRICS / SHAP / LIME / BEESWARM / WATERFALL
# ==============================================================================

# ── top15_tokens ──────────────────────────────────────────────────────────────

def top15_tokens(
    features,
    weights,
    stopwords: frozenset = STOPWORDS,
    clean_glyph: bool = False,
) -> List[Tuple[str, float]]:
    """
    Return up to 15 (token, weight) pairs sorted by |weight| descending,
    with stopword filtering and deduplication.

    Parameters
    ----------
    features    : iterable of token strings
    weights     : iterable of float weights (same length as features)
    stopwords   : set to filter against (default: module-level STOPWORDS)
    clean_glyph : if True, strip RoBERTa 'Ġ' prefix before filtering —
                  set True in bert / deepseek / fusion callers
    """
    paired = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)
    seen: set = set()
    out: List[Tuple[str, float]] = []

    for f, w in paired:
        fs = str(f).lower().strip()
        if clean_glyph:
            fs = fs.replace('ġ', '').strip()
        if fs in stopwords or len(fs) < 2 or (clean_glyph and fs.isnumeric()):
            continue
        if fs not in seen:
            out.append((fs, float(w)))
            seen.add(fs)
        if len(out) >= 15:
            break

    # Relax stopword constraint if fewer than 15 found
    if len(out) < 15:
        for f, w in paired:
            fs = str(f).lower().strip()
            if clean_glyph:
                fs = fs.replace('ġ', '').strip()
            if fs not in seen:
                out.append((fs, float(w)))
                seen.add(fs)
            if len(out) >= 15:
                break

    return out[:15]


# ── plot_bar ──────────────────────────────────────────────────────────────────

def plot_bar(
    items: List[Tuple],
    title: str,
    output_path: Path,
    plot_dpi: int = 300,
) -> None:
    """
    Uniform horizontal bar chart for cross-model LIME / SHAP comparison.

    Design rules — identical across all 5 models so charts are side-by-side
    comparable without rescaling:
      • Fixed x-axis [-1.0, 1.0] — same scale for every model.
      • Only real tokens shown — empty / stopword / dim_ rows are excluded.
      • No value labels on bars — bar length is the visual comparison signal.
      • Positive → blue (#1f77b4),  negative → orange (#ff7f0e).
      • Subtle x-grid, no top/right spines.
      • Companion CSV written alongside the PNG for cross-model comparison.

    CSV columns: Token, Score, Direction
    """
    # ── Strip invalid tokens before plotting ─────────────────────────────────
    valid_pairs = [
        (str(tok), float(w))
        for tok, w in items
        if tok                              # not empty
        and not str(tok).startswith("dim_") # not an SBERT dimension label
        and not str(tok).isnumeric()        # not a bare number
        and len(str(tok)) >= 2              # not a single character
    ]

    N = min(15, len(valid_pairs))
    if N == 0:
        _log.warning(f"  plot_bar: no valid tokens to plot for '{title}' — skipping.")
        return

    pairs   = valid_pairs[:N]
    names   = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#1f77b4' if w >= 0 else '#ff7f0e' for w in weights]
    bars = ax.barh(range(N), weights, color=colors, height=0.65, edgecolor='none')

    ax.set_yticks(range(N))
    ax.set_yticklabels(names, fontsize=12, fontweight='normal')
    ax.invert_yaxis()

    # Dynamic x-axis: fixed [-1, 1] when all values fit, else pad by 10 %
    max_abs = max(abs(w) for w in weights) if weights else 1.0
    xlim = 1.0 if max_abs <= 1.0 else max_abs * 1.10
    ax.set_xlim(-xlim, xlim)
    ax.axvline(x=0, color='#333333', linewidth=0.9, linestyle='-')
    ax.set_xlabel("LIME / SHAP Impact Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14, wrap=True)

    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()
    _log.info(f"  plot_bar: PNG -> {output_path}")

    # ── Companion CSV ─────────────────────────────────────────────────────────
    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    pd.DataFrame({
        "Token":     names,
        "Score":     weights,
        "Direction": ["positive" if w >= 0 else "negative" for w in weights],
    }).to_csv(csv_path, index=False)
    _log.info(f"  plot_bar: CSV -> {csv_path}")


# ── compute_metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    lime_score: Optional[float],
    shap_top15: List[Tuple],
    lime_top15: List[Tuple],
    category_shap_vectors: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute three honest explainability metrics. No clamping, no random noise.

    Fidelity
        sqrt(|R²|) from LIME's local linear fit score.
        Measures how well LIME's surrogate approximates the black-box locally.

    Jaccard
        |SHAP_tokens ∩ LIME_tokens| / |SHAP_tokens ∪ LIME_tokens|
        Measures agreement on which tokens matter (set overlap of top-15).

    Stability
        Mean pairwise Spearman-r of SHAP vectors across same-category samples.
        Measures how consistently the model explains the same category.
        Falls back to Jaccard when fewer than 2 SHAP vectors are available.
    """
    raw_r2   = abs(lime_score) if lime_score is not None else 0.0
    fidelity = round(float(np.sqrt(min(raw_r2, 1.0))), 4)

    s_set   = {str(x[0]).lower().strip() for x in shap_top15 if x[0]}
    l_set   = {str(x[0]).lower().strip() for x in lime_top15 if x[0]}
    union   = s_set | l_set
    jaccard = round(len(s_set & l_set) / len(union), 4) if union else 0.0

    if category_shap_vectors and len(category_shap_vectors) >= 2:
        corrs: List[float] = []
        ref = category_shap_vectors[0]
        for vec in category_shap_vectors[1:]:
            if len(vec) == len(ref) and np.std(vec) > 1e-9 and np.std(ref) > 1e-9:
                r, _ = spearmanr(ref, vec)
                corrs.append(float(r))
        stability = round(float(np.mean(corrs)), 4) if corrs else jaccard
    else:
        stability = jaccard

    return {'Fidelity': fidelity, 'Jaccard': jaccard, 'Stability': stability}


# ── build_shap_background ─────────────────────────────────────────────────────

def build_shap_background(X_train: np.ndarray, n: int = 50) -> np.ndarray:
    """
    Compress the training set into n kmeans cluster centres for use as the
    KernelExplainer background distribution.

    n=50 balances accuracy and speed for 384-dim SBERT embeddings.
    """
    n = min(n, len(X_train))
    _log.info(f"  Building KernelExplainer background ({n} kmeans clusters)…")
    return shap.kmeans(X_train, n).data


# ── run_global_shap ───────────────────────────────────────────────────────────

def run_global_shap(
    kernel_explainer,
    X_sample: np.ndarray,
    class_labels: List[str],
    model_name: str,
    output_path: Path,
    plot_dpi: int = 300,
) -> None:
    """
    Run global SHAP over X_sample, compute per-target-category mean |SHAP|,
    and save a category importance bar chart.

    Used by ML and DL models (KernelExplainer on SBERT embeddings).
    BERT / DeepSeek / Fusion use shap.Explainer (text masker) instead and
    build their global aggregation inline.
    """
    _log.info(f"  Global SHAP for {model_name} ({len(X_sample)} samples)…")
    try:
        shap_vals = kernel_explainer.shap_values(
            X_sample, silent=True, nsamples='auto'
        )
    except Exception as _e:
        _log.warning(
            f"  run_global_shap: batch shap_values failed ({_e}). "
            f"Falling back to row-by-row (slower but safe)."
        )
        # Row-by-row fallback: avoids maskMatrix shape crash in SHAP KernelExplainer
        _rows = []
        for _i in range(len(X_sample)):
            try:
                _sv = kernel_explainer.shap_values(
                    X_sample[_i:_i+1], silent=True, nsamples='auto'
                )
                _rows.append(_sv)
            except Exception:
                continue
        if not _rows:
            _log.warning(f"  run_global_shap: all rows failed for {model_name}.")
            return
        # Reconstruct shap_vals in the same format as the batch call
        if isinstance(_rows[0], list):
            shap_vals = [np.vstack([r[i] for r in _rows]) for i in range(len(_rows[0]))]
        else:
            shap_vals = np.vstack(_rows)

    category_impact: List[Tuple[str, float]] = []
    if isinstance(shap_vals, list):
        for idx, sv in enumerate(shap_vals):
            if idx < len(class_labels) and class_labels[idx] in TARGET_CATEGORIES:
                category_impact.append(
                    (class_labels[idx], float(np.mean(np.abs(sv))))
                )
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        for idx in range(shap_vals.shape[2]):
            if idx < len(class_labels) and class_labels[idx] in TARGET_CATEGORIES:
                category_impact.append(
                    (class_labels[idx], float(np.mean(np.abs(shap_vals[:, :, idx]))))
                )

    if not category_impact:
        _log.warning(f"  No category impact extracted for {model_name}.")
        return

    # Normalise if raw margins are very large (e.g. XGBoost)
    vals = [v for _, v in category_impact]
    if max(vals, default=0) > 100:
        total = sum(vals) + 1e-9
        category_impact = [(c, v / total) for c, v in category_impact]

    # Pad any missing target categories with 0
    existing = {c for c, _ in category_impact}
    for cat in TARGET_CATEGORIES:
        if cat not in existing:
            category_impact.append((cat, 0.0))
    category_impact.sort(key=lambda x: x[1], reverse=True)

    plot_bar(
        category_impact,
        f"Global Category Importance (SBERT) — {model_name}",
        output_path,
        plot_dpi=plot_dpi,
    )


# ── run_global_lime ───────────────────────────────────────────────────────────

def run_global_lime(
    lime_explainer,
    predict_fn,
    test_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    sample_limit: int = 15,
    clean_glyph: bool = False,
    plot_dpi: int = 300,
) -> None:
    """
    Aggregate LIME word weights across up to sample_limit samples and save a
    global importance bar chart.

    Parameters
    ----------
    lime_explainer : LimeTextExplainer instance
    predict_fn     : callable  texts → probability array
    test_df        : DataFrame with 'cleaned_text' column
    model_name     : used in chart title and log messages
    output_path    : full path for the saved PNG
    sample_limit   : max distinct prediction classes to sample (default 15)
    clean_glyph    : strip RoBERTa 'Ġ' byte prefix — set True for
                     BERT / DeepSeek / Fusion
    plot_dpi       : output resolution
    """
    _log.info(f"  Global LIME for {model_name} ({sample_limit} samples)…")
    global_w: Dict[str, float] = defaultdict(float)
    seen: set = set()

    for i in range(len(test_df)):
        if len(seen) >= sample_limit:
            break
        try:
            text    = str(test_df.iloc[i]['cleaned_text'])
            probs   = predict_fn([text])[0]
            top_cls = int(np.argmax(probs))
            exp = lime_explainer.explain_instance(
                text, predict_fn,
                labels=[top_cls], num_features=15, num_samples=300,
            )
            for word, w in exp.as_list(label=top_cls):
                fs = word.lower().strip()
                if clean_glyph:
                    fs = fs.replace('ġ', '').strip()
                if fs not in STOPWORDS and len(fs) >= 2 and not fs.isnumeric():
                    global_w[fs] += abs(w)
            seen.add(top_cls)
        except Exception:
            continue

    if global_w:
        items = sorted(global_w.items(), key=lambda x: x[1], reverse=True)
        top   = top15_tokens(
            [k for k, _ in items],
            [v for _, v in items],
            clean_glyph=clean_glyph,
        )
        plot_bar(
            top,
            f"Global LIME Aggregated — {model_name}",
            output_path,
            plot_dpi=plot_dpi,
        )



# ── run_beeswarm ──────────────────────────────────────────────────────────────

def run_beeswarm(
    beeswarm_rows: List[Dict],
    model_name: str,
    output_path: Path,
    plot_dpi: int = 300,
    n_top: int = 15,
) -> None:
    """
    Build a beeswarm (strip plot) from (Token, SHAP Value) rows accumulated
    during the local explanation loop. Save PNG + CSV.

    PNG  -> output_path
    CSV  -> output_path.parent / f"{stem}_data.csv"
            Columns: Token, SHAP_Value

    beeswarm_rows is built by the caller: one dict per (word, shap_projection)
    pair for every shared sample processed.  With 5 categories × 3 samples ×
    ~10 words each → ~150 rows → meaningful per-token distribution.

    Parameters
    ----------
    beeswarm_rows : list of {'Token': str, 'SHAP Value': float}
    model_name    : used in chart title
    output_path   : full path for saved PNG
    plot_dpi      : output resolution
    n_top         : top tokens to display (default 15)
    """
    if not beeswarm_rows:
        _log.warning(f"  run_beeswarm: no data for {model_name} — skipping.")
        return

    df = pd.DataFrame(beeswarm_rows)

    top_tokens: List[str] = (
        df.groupby("Token")["SHAP Value"]
        .apply(lambda x: x.abs().mean())
        .nlargest(n_top)
        .index.tolist()
    )
    df_plot = df[df["Token"].isin(top_tokens)].copy()
    df_plot["Token"] = pd.Categorical(
        df_plot["Token"], categories=top_tokens, ordered=True
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.stripplot(
        data=df_plot, x="SHAP Value", y="Token",
        jitter=0.25, alpha=0.65, palette="viridis", size=6, ax=ax,
    )
    ax.axvline(x=0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xlabel("SHAP Projection Score", fontsize=11)
    ax.set_title(
        f"SHAP Beeswarm (Top {n_top} Tokens) — {model_name}",
        fontsize=13, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    _log.info(f"  run_beeswarm: PNG -> {output_path}")

    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    df_plot[["Token", "SHAP Value"]].rename(
        columns={"SHAP Value": "SHAP_Value"}
    ).to_csv(csv_path, index=False)
    _log.info(f"  run_beeswarm: CSV -> {csv_path}")


# ── run_waterfall ─────────────────────────────────────────────────────────────

def run_waterfall(
    shap_top15: List[Tuple[str, float]],
    base_val: float,
    model_name: str,
    cat_name: str,
    output_path: Path,
    plot_dpi: int = 300,
) -> None:
    """
    Render a SHAP waterfall plot from (token, shap_value) pairs.
    Save PNG + CSV.

    PNG  -> output_path
    CSV  -> output_path.parent / f"{stem}_data.csv"
            Columns: Token, SHAP_Value, Base_Value, Model, Category

    Called once per model for the first valid shared sample
    (waterfall_done flag in caller prevents multiple renders per model).

    Parameters
    ----------
    shap_top15  : list of (token, shap_projection) tuples — top 15
    base_val    : SHAP expected value E[f(X)] for the predicted class
    model_name  : used in chart title and CSV
    cat_name    : category name — used in title and CSV
    output_path : full path for saved PNG
    plot_dpi    : output resolution
    """
    if not shap_top15:
        _log.warning(
            f"  run_waterfall: empty shap_top15 for {model_name}/{cat_name} — skipping."
        )
        return

    w_names = np.array([x[0] for x in shap_top15])
    w_vals  = np.array([x[1] for x in shap_top15])

    exp_obj = shap.Explanation(
        values=w_vals,
        base_values=base_val,
        data=w_names,
        feature_names=list(w_names),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(exp_obj, max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — {model_name} — {cat_name}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    _log.info(f"  run_waterfall: PNG -> {output_path}")

    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    pd.DataFrame({
        "Token":      [x[0] for x in shap_top15],
        "SHAP_Value": [x[1] for x in shap_top15],
        "Base_Value": base_val,
        "Model":      model_name,
        "Category":   cat_name,
    }).to_csv(csv_path, index=False)
    _log.info(f"  run_waterfall: CSV -> {csv_path}")

# ==============================================================================
#  L. run_global_category_bar  — category importance bar (ML + DL SHAP output)
# ==============================================================================

def run_global_category_bar(
    shap_values,
    class_labels: List[str],
    model_name: str,
    target_categories: List[str],
    output_path: Path,
    plot_dpi: int = 300,
) -> None:
    """
    Compute mean |SHAP| per target category from a KernelExplainer / TreeExplainer
    shap_values array and save a ranked bar chart (PNG + CSV).

    Handles three common SHAP output shapes:
      • list of arrays (RandomForest / multi-output KernelExplainer)
      • 3-D ndarray shape (n_samples, n_features, n_classes)   (XGBoost)
      • 2-D ndarray shape (n_samples, n_features)              (binary)

    Normalises raw margins when max > 100 (avoids unreadable x-axis for XGB).

    PNG  -> output_path
    CSV  -> output_path.parent / f"{stem}_data.csv"
            Columns: Category, Mean_Abs_SHAP
    """
    category_impact: List[Tuple[str, float]] = []

    if isinstance(shap_values, list):
        for idx, sv in enumerate(shap_values):
            if idx < len(class_labels) and class_labels[idx] in target_categories:
                category_impact.append(
                    (class_labels[idx], float(np.mean(np.abs(sv))))
                )
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            for idx in range(shap_values.shape[2]):
                if idx < len(class_labels) and class_labels[idx] in target_categories:
                    category_impact.append(
                        (class_labels[idx], float(np.mean(np.abs(shap_values[:, :, idx]))))
                    )
        elif shap_values.ndim == 2:
            # Binary or single-output: attribute entire mean |SHAP| to first target cat
            pass   # callers can handle this edge case themselves

    if not category_impact:
        _log.warning(f"  run_global_category_bar: no category impact for {model_name}.")
        return

    # Normalise when raw margins are very large (e.g. XGBoost)
    vals = [v for _, v in category_impact]
    if max(vals, default=0) > 100:
        total = sum(vals) + 1e-9
        category_impact = [(c, v / total) for c, v in category_impact]

    # Pad missing target categories with 0
    existing = {c for c, _ in category_impact}
    for cat in target_categories:
        if cat not in existing:
            category_impact.append((cat, 0.0))
    category_impact.sort(key=lambda x: x[1], reverse=True)

    output_path = Path(output_path)
    plot_bar(
        category_impact,
        f"Global Category Importance (SHAP) — {model_name}",
        output_path,
        plot_dpi=plot_dpi,
    )
    # plot_bar already writes the companion CSV alongside the PNG


# ==============================================================================
#  M. extract_global_tokens  — pull top-15 tokens per category from SHAP values
# ==============================================================================

def extract_global_tokens(
    shap_values,
    class_labels: List[str],
    feature_names,
    target_categories: List[str],
    clean_glyph: bool = False,
) -> Dict[str, List[str]]:
    """
    Extract the top-15 clean token strings for every target category from a
    SHAP values array (same shapes as run_global_category_bar).

    Returns a dict  {category_name: [token, token, …]}  (up to 15 tokens each).
    dim_* features are always excluded.

    Parameters
    ----------
    shap_values      : SHAP output (list or ndarray, see run_global_category_bar)
    class_labels     : ordered list of category name strings
    feature_names    : ordered list of feature name strings (TF-IDF vocabulary, etc.)
    target_categories: only these categories are populated in the result
    clean_glyph      : strip RoBERTa Ġ prefix before filtering (BERT/DS/Fusion)
    """
    result: Dict[str, List[str]] = {}

    def _get_class_vals(idx: int):
        if isinstance(shap_values, list) and idx < len(shap_values):
            return np.mean(np.abs(shap_values[idx]), axis=0)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            return np.mean(np.abs(shap_values[:, :, idx]), axis=0)
        return None

    for idx, cat in enumerate(class_labels):
        if cat not in target_categories:
            continue
        vals = _get_class_vals(idx)
        if vals is None:
            continue
        top = top15_tokens(feature_names, vals, clean_glyph=clean_glyph)
        result[cat] = [t for t, _ in top if not str(t).startswith("dim_")]

    return result


# ==============================================================================
#  N. save_metrics_report  — write CSV + grouped bar chart from metrics list
# ==============================================================================

def save_metrics_report(
    metrics_storage: List[Dict],
    model_col: str,
    output_csv: Path,
    output_png: Path,
    title: str = "XAI Metrics Comparison",
    plot_dpi: int = 300,
) -> None:
    """
    Persist the per-sample metrics list collected during an explainability run.

    CSV  -> output_csv      (one row per sample, columns = model_col + metric names)
    PNG  -> output_png      (grouped bar chart of mean metrics per model, with
                             bar labels — this is the METRICS chart, not a token
                             bar, so bar labels on numeric scores are appropriate)

    Parameters
    ----------
    metrics_storage : list of dicts, each with at least {model_col, Fidelity,
                      Jaccard, Stability} — as built by compute_metrics() callers
    model_col       : column name that identifies the model (e.g. 'model')
    output_csv      : full path for the CSV file
    output_png      : full path for the PNG file
    title           : chart title
    plot_dpi        : output resolution
    """
    if not metrics_storage:
        _log.warning("  save_metrics_report: empty metrics_storage — skipping.")
        return

    df = pd.DataFrame(metrics_storage)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    _log.info(f"  save_metrics_report: CSV -> {output_csv}")

    metric_cols = [c for c in ["Fidelity", "Jaccard", "Stability"] if c in df.columns]
    if not metric_cols or model_col not in df.columns:
        return

    summary = df.groupby(model_col)[metric_cols].mean().reset_index()
    melted  = summary.melt(id_vars=model_col, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=melted, x="Metric", y="Score", hue=model_col,
                palette="viridis", ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", padding=4, fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=model_col.capitalize())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=plot_dpi, bbox_inches="tight")
    plt.close()
    _log.info(f"  save_metrics_report: PNG -> {output_png}")


# ==============================================================================
#  EXPORTS
# ==============================================================================

__all__ = [
    # A. Constants
    "STOPWORDS", "TARGET_CATEGORIES", "FALLBACK_LABELS",
    # B. Labels
    "load_class_labels",
    # C. Shared samples
    "EXPL_TARGET_CATEGORIES", "EXPL_LABEL_IDS", "N_SAMPLES_PER_CATEGORY",
    "FIXED_CATEGORIES",
    "get_shared_samples",
    # D. Token helpers
    "top15_tokens",
    # E. Plotting
    "plot_bar",
    # F. Metrics
    "compute_metrics",
    # G. SHAP background
    "build_shap_background",
    # H. Global SHAP
    "run_global_shap",
    # I. Global LIME
    "run_global_lime",
    # J. Beeswarm
    "run_beeswarm",
    # K. Waterfall
    "run_waterfall",
    # L. Global category bar (ML + DL)
    "run_global_category_bar",
    # M. Token extractor from SHAP values
    "extract_global_tokens",
    # N. Metrics CSV + chart saver
    "save_metrics_report",
]

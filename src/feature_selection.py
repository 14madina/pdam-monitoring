"""
src/feature_selection.py — 5 metode feature selection untuk skenario S1.

Rules:
- Semua metode di-fit pada TRAIN saja.
- Target: target_next_usage.
- Output: raw score, normalized score, rank, vote_count, consensus_score.
- Threshold default: vote_count >= 3 untuk masuk S1_SELECTED_FEATURES.
- Feature selection hanya dilakukan SATU KALI sebelum modeling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================
# INPUT SANITIZATION HELPER
# =============================================================================

def _clean_input_xy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: Optional[int] = 50000,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sanitasi input X dan y untuk feature selection:
    1. Hanya gunakan kolom numerik.
    2. Hapus kolom yang 100% bernilai NaN.
    3. Imputasi NaN pada X dengan median, lalu fallback ke 0.0 jika masih ada NaN.
    4. Imputasi NaN pada y dengan median, lalu fallback ke 0.0.
    5. Ganti nilai inf dan -inf dengan 0.0.
    6. Lakukan representational subsampling (default 50,000 baris) jika dataset sangat besar
       agar mutual info, Extra Trees, dan Elastic Net berjalan cepat dan stabil tanpa memory freeze.
    """
    X = X_train.select_dtypes(include=[np.number]).copy()

    # Drop kolom yang 100% NaN
    valid_cols = [c for c in X.columns if X[c].notna().sum() > 0]
    X = X[valid_cols].copy()

    # Imputasi X: median kolom -> fallback 0.0
    X_med = X.median()
    X_filled = X.fillna(X_med).fillna(0.0)
    X_filled = X_filled.replace([np.inf, -np.inf], 0.0)

    # Imputasi y: median -> fallback 0.0
    y_med = y_train.median() if y_train.notna().any() else 0.0
    y_filled = y_train.fillna(y_med).fillna(0.0)
    y_filled = y_filled.replace([np.inf, -np.inf], 0.0)

    # Subsampling jika dataset besar (> sample_size)
    if sample_size and len(X_filled) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_filled), size=sample_size, replace=False)
        X_filled = X_filled.iloc[idx].reset_index(drop=True)
        y_filled = y_filled.iloc[idx].reset_index(drop=True)

    return X_filled, y_filled


# =============================================================================
# INDIVIDUAL METHODS (fit pada Train, target = target_next_usage)
# =============================================================================

def run_spearman_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.Series:
    """
    Hitung Spearman correlation |ρ| antara setiap fitur dan target.
    Lebih robust terhadap outlier dan distribusi non-normal.

    Returns
    -------
    pd.Series: |spearman_rho| per fitur, index = feature names.
    """
    X_clean, y_clean = _clean_input_xy(X_train, y_train, sample_size=50000)
    scores = {}
    for col in X_clean.columns:
        valid = X_clean[col].notna() & y_clean.notna()
        if valid.sum() < 10 or X_clean[col].nunique() <= 1:
            scores[col] = 0.0
            continue
        rho, _ = spearmanr(X_clean.loc[valid, col], y_clean[valid])
        scores[col] = abs(rho) if not np.isnan(rho) else 0.0

    result = pd.Series(scores, name="spearman_abs_rho")
    # Pastikan seluruh kolom numerik awal ada
    for col in X_train.select_dtypes(include=[np.number]).columns:
        if col not in result:
            result[col] = 0.0
    logger.info(f"Spearman correlation computed for {len(result)} features.")
    return result


def run_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> pd.Series:
    """
    Mutual Information Regression — menangkap hubungan non-linear.

    Returns
    -------
    pd.Series: MI score per fitur.
    """
    X_clean, y_clean = _clean_input_xy(X_train, y_train, sample_size=30000, random_state=random_state)
    all_num_cols = X_train.select_dtypes(include=[np.number]).columns

    if X_clean.empty or len(y_clean) < 10:
        return pd.Series(0.0, index=all_num_cols, name="mutual_info")

    mi_scores = mutual_info_regression(
        X_clean, y_clean,
        random_state=random_state,
        n_jobs=-1,
    )
    result = pd.Series(mi_scores, index=X_clean.columns, name="mutual_info")
    for col in all_num_cols:
        if col not in result:
            result[col] = 0.0
    logger.info(f"Mutual Information computed for {len(result)} features.")
    return result


def run_f_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.Series:
    """
    F-statistic regression (linear association strength).

    Returns
    -------
    pd.Series: F-score per fitur.
    """
    X_clean, y_clean = _clean_input_xy(X_train, y_train, sample_size=100000)
    all_num_cols = X_train.select_dtypes(include=[np.number]).columns

    if X_clean.empty or len(y_clean) < 10:
        return pd.Series(0.0, index=all_num_cols, name="f_regression")

    f_scores, _ = f_regression(X_clean, y_clean)
    result = pd.Series(f_scores, index=X_clean.columns, name="f_regression").fillna(0.0)
    for col in all_num_cols:
        if col not in result:
            result[col] = 0.0
    logger.info(f"F-Regression computed for {len(result)} features.")
    return result


def run_extra_trees_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = config.RANDOM_STATE,
) -> pd.Series:
    """
    Extra Trees Feature Importance — captures non-linear interactions.

    Returns
    -------
    pd.Series: feature importance per fitur.
    """
    X_clean, y_clean = _clean_input_xy(X_train, y_train, sample_size=30000, random_state=random_state)
    all_num_cols = X_train.select_dtypes(include=[np.number]).columns

    if X_clean.empty or len(y_clean) < 10:
        return pd.Series(0.0, index=all_num_cols, name="extra_trees_importance")

    et = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    et.fit(X_clean, y_clean)

    result = pd.Series(et.feature_importances_, index=X_clean.columns, name="extra_trees_importance")
    for col in all_num_cols:
        if col not in result:
            result[col] = 0.0
    logger.info(f"Extra Trees importance computed for {len(result)} features.")
    return result


def run_elastic_net(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> pd.Series:
    """
    Elastic Net — regularized regression untuk fitur selection via koefisien non-zero.
    Fit dengan StandardScaler.

    Returns
    -------
    pd.Series: |coefficient| per fitur.
    """
    X_clean, y_clean = _clean_input_xy(X_train, y_train, sample_size=30000, random_state=random_state)
    all_num_cols = X_train.select_dtypes(include=[np.number]).columns

    if X_clean.empty or len(y_clean) < 10:
        return pd.Series(0.0, index=all_num_cols, name="elastic_net_abs_coef")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    en = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
        cv=3,
        random_state=random_state,
        max_iter=2000,
        n_jobs=-1,
    )
    en.fit(X_scaled, y_clean)

    result = pd.Series(np.abs(en.coef_), index=X_clean.columns, name="elastic_net_abs_coef")
    for col in all_num_cols:
        if col not in result:
            result[col] = 0.0
    logger.info(
        f"Elastic Net computed. l1_ratio={en.l1_ratio_:.3f}, alpha={en.alpha_:.6f}, "
        f"non-zero features: {(np.abs(en.coef_) > 0).sum()}"
    )
    return result


# =============================================================================
# AGGREGATE SCORES
# =============================================================================

def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-Max normalize ke [0, 1]."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def aggregate_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_votes: int = config.FEATURE_SELECTION_MIN_VOTES,
) -> pd.DataFrame:
    """
    Jalankan 5 metode feature selection dan agregasikan hasilnya.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix dari training set.
    y_train : pd.Series
        Target dari training set.
    min_votes : int
        Threshold minimum vote_count untuk masuk S1_SELECTED_FEATURES.

    Returns
    -------
    pd.DataFrame dengan kolom:
        feature, spearman, mutual_info, f_regression, extra_trees, elastic_net,
        spearman_norm, mutual_info_norm, f_regression_norm,
        extra_trees_norm, elastic_net_norm,
        spearman_rank, ..., elastic_net_rank,
        spearman_selected, ..., elastic_net_selected,
        vote_count, consensus_score, selected_s1
    """
    logger.info("Running 5-method feature selection ...")

    scores = {
        "spearman"    : run_spearman_correlation(X_train, y_train),
        "mutual_info" : run_mutual_information(X_train, y_train),
        "f_regression": run_f_regression(X_train, y_train),
        "extra_trees" : run_extra_trees_importance(X_train, y_train),
        "elastic_net" : run_elastic_net(X_train, y_train),
    }

    df_scores = pd.DataFrame(scores)
    df_scores.index.name = "feature"
    df_scores = df_scores.reset_index()

    methods = list(scores.keys())

    # Normalized scores
    for m in methods:
        df_scores[f"{m}_norm"] = _normalize_series(df_scores[m])

    # Ranks (1 = terbaik)
    for m in methods:
        df_scores[f"{m}_rank"] = df_scores[m].rank(ascending=False, method="min").astype(int)

    # Selected per method (top 50% atau non-zero)
    n_features = len(df_scores)
    top_k = max(1, n_features // 2)

    df_scores["spearman_selected"]    = (df_scores["spearman_rank"]     <= top_k).astype(int)
    df_scores["mutual_info_selected"] = (df_scores["mutual_info_rank"]  <= top_k).astype(int)
    df_scores["f_regression_selected"]= (df_scores["f_regression_rank"] <= top_k).astype(int)
    df_scores["extra_trees_selected"] = (df_scores["extra_trees_rank"]  <= top_k).astype(int)
    df_scores["elastic_net_selected"] = (df_scores["elastic_net"] > 0).astype(int)

    # Vote count (0–5)
    selected_cols = [f"{m}_selected" for m in methods]
    df_scores["vote_count"] = df_scores[selected_cols].sum(axis=1)

    # Consensus score (rata-rata normalized)
    norm_cols = [f"{m}_norm" for m in methods]
    df_scores["consensus_score"] = df_scores[norm_cols].mean(axis=1)

    # S1 selection
    df_scores["selected_s1"] = (df_scores["vote_count"] >= min_votes).astype(int)

    n_selected = df_scores["selected_s1"].sum()
    logger.info(
        f"✅ Feature selection complete. "
        f"Total features: {n_features}, "
        f"S1 selected (vote >= {min_votes}): {n_selected}"
    )

    return df_scores.sort_values("consensus_score", ascending=False).reset_index(drop=True)


# =============================================================================
# GET FEATURE LISTS
# =============================================================================

def get_selected_features(
    df_scores: pd.DataFrame,
    min_votes: int = config.FEATURE_SELECTION_MIN_VOTES,
) -> Dict[str, List[str]]:
    """
    Ambil daftar fitur untuk S0 (all) dan S1 (selected).

    Returns
    -------
    dict: {'all_features': [...], 'selected_features': [...]}
    """
    all_features = df_scores["feature"].tolist()
    selected = df_scores.loc[df_scores["vote_count"] >= min_votes, "feature"].tolist()

    logger.info(f"S0 (all features): {len(all_features)} features")
    logger.info(f"S1 (selected features): {len(selected)} features")

    return {
        "all_features"      : all_features,
        "selected_features" : selected,
    }


# =============================================================================
# SAVE / LOAD FEATURE LISTS
# =============================================================================

def save_feature_lists(
    feature_dict: Dict[str, List[str]],
    all_path: Path = config.DATA_PROCESSED / "all_features.json",
    sel_path: Path = config.DATA_PROCESSED / "selected_features.json",
) -> None:
    """Simpan daftar fitur ke JSON."""
    with open(all_path, "w") as f:
        json.dump(feature_dict["all_features"], f, indent=2)
    with open(sel_path, "w") as f:
        json.dump(feature_dict["selected_features"], f, indent=2)

    logger.info(f"all_features.json saved: {len(feature_dict['all_features'])} features")
    logger.info(f"selected_features.json saved: {len(feature_dict['selected_features'])} features")


def load_feature_lists(
    all_path: Path = config.DATA_PROCESSED / "all_features.json",
    sel_path: Path = config.DATA_PROCESSED / "selected_features.json",
) -> Dict[str, List[str]]:
    """Load daftar fitur dari JSON."""
    with open(all_path) as f:
        all_features = json.load(f)
    with open(sel_path) as f:
        selected_features = json.load(f)

    return {
        "all_features"      : all_features,
        "selected_features" : selected_features,
    }


# =============================================================================
# SUMMARY TABLE BUILDER
# =============================================================================

def build_feature_selection_summary_table(df_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Buat tabel ringkasan 5 metode feature selection yang mencakup:
    metode, input features, metrik/strategi, fitur terpilih, dan evaluasi ringkas.

    Returns
    -------
    pd.DataFrame
    """
    n_features = len(df_scores)
    
    summary_data = [
        {
            "Method": "Spearman Correlation",
            "Input Features": f"All {n_features} Candidate Features",
            "Metric / Strategy": "|Spearman Rho| (Non-linear Monotonic Correlation)",
            "Selected Count": df_scores["spearman_selected"].sum(),
            "Top Selected Features": ", ".join(df_scores.sort_values("spearman_rank")["feature"].head(5).tolist()),
            "Evaluation / Impact": "Menangkap hubungan monotonis non-linear; terbukti stabil terhadap outlier."
        },
        {
            "Method": "Mutual Information Regression",
            "Input Features": f"All {n_features} Candidate Features",
            "Metric / Strategy": "Mutual Information (K-NN non-parametric entropy)",
            "Selected Count": df_scores["mutual_info_selected"].sum(),
            "Top Selected Features": ", ".join(df_scores.sort_values("mutual_info_rank")["feature"].head(5).tolist()),
            "Evaluation / Impact": "Menangkap ketergantungan non-linear kompleks antar variabel time-series."
        },
        {
            "Method": "F-Regression",
            "Input Features": f"All {n_features} Candidate Features",
            "Metric / Strategy": "ANOVA F-Statistic (Linear Association Test)",
            "Selected Count": df_scores["f_regression_selected"].sum(),
            "Top Selected Features": ", ".join(df_scores.sort_values("f_regression_rank")["feature"].head(5).tolist()),
            "Evaluation / Impact": "Menilai kekuatan linier antar fitur dengan target."
        },
        {
            "Method": "Extra Trees Importance",
            "Input Features": f"All {n_features} Candidate Features",
            "Metric / Strategy": "Gini / Variance Reduction Importance (200 Trees)",
            "Selected Count": df_scores["extra_trees_selected"].sum(),
            "Top Selected Features": ", ".join(df_scores.sort_values("extra_trees_rank")["feature"].head(5).tolist()),
            "Evaluation / Impact": "Sangat efektif dalam menangkap interaksi kompleks antar fitur histori."
        },
        {
            "Method": "Elastic Net",
            "Input Features": f"All {n_features} Candidate Features",
            "Metric / Strategy": "Regularized L1+L2 Regression (|coef| > 0)",
            "Selected Count": df_scores["elastic_net_selected"].sum(),
            "Top Selected Features": ", ".join(df_scores.sort_values("elastic_net", ascending=False)["feature"].head(5).tolist()),
            "Evaluation / Impact": "Mengeliminasi fitur redundan melalui koefisien zero (L1 sparsity)."
        },
    ]

    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("feature_selection.py loaded OK")


"""
src/evaluation.py — Evaluasi komprehensif K-Means dan ANN.

Mencakup:
- Metrik regresi: RMSE, MAE, MARE, NSC/NSE, R²
- Baseline Lag-1
- Perbandingan S0 vs S1 untuk K-Means dan ANN
- Evaluasi pada subset observed-only test
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "model",
    epsilon: float = 1e-6,
) -> Dict:
    """
    Hitung metrik evaluasi regresi lengkap.

    Metrik:
    - RMSE  : Root Mean Squared Error
    - MAE   : Mean Absolute Error
    - MARE  : Mean Absolute Relative Error (hati-hati saat aktual mendekati 0)
    - NSC   : Nash-Sutcliffe Coefficient (NSE)
    - R²    : Coefficient of Determination
    - n_obs : Jumlah observasi valid

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    label  : str
    epsilon: float, untuk MARE agar tidak divide by zero

    Returns
    -------
    dict
    """
    # Handle NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t  = y_true[mask]
    y_p  = y_pred[mask]
    n    = len(y_t)

    if n == 0:
        logger.warning(f"[{label}] Tidak ada data valid untuk evaluasi.")
        return {
            "label": label, "n_obs": 0,
            "rmse": None, "mae": None, "mare": None,
            "nsc": None, "r2": None,
        }

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae  = float(mean_absolute_error(y_t, y_p))

    # MARE — gunakan max(|y_true|, epsilon) untuk hindari div-zero
    denom = np.maximum(np.abs(y_t), epsilon)
    mare  = float(np.mean(np.abs(y_t - y_p) / denom))

    # NSC/NSE = 1 - SS_res / SS_tot
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    nsc    = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    r2 = float(r2_score(y_t, y_p))

    metrics = {
        "label" : label,
        "n_obs" : n,
        "rmse"  : round(rmse, 6),
        "mae"   : round(mae,  6),
        "mare"  : round(mare, 6),
        "nsc"   : round(nsc,  6) if nsc is not None else None,
        "r2"    : round(r2,   6),
    }

    logger.info(
        f"[{label}] n={n:,} | "
        f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
        f"MARE={metrics['mare']:.4f} | NSC={metrics['nsc']} | R²={metrics['r2']:.4f}"
    )

    return metrics


# =============================================================================
# BASELINE LAG-1
# =============================================================================

def compute_baseline_lag1(df: pd.DataFrame) -> Dict:
    """
    Baseline Lag-1: prediksi = usage periode saat ini (sebagai prediksi periode berikutnya).

    prediction(t+1) = usage(t) = lag_1 dari perspektif t+1
    Jika model ANN tidak bisa mengalahkan baseline ini, model tidak berguna.

    Parameters
    ----------
    df : pd.DataFrame
        Harus memiliki 'target_next_usage' dan 'usage_current' (atau 'Ini' dan 'Lalu').

    Returns
    -------
    dict metrik
    """
    if "usage_current" in df.columns:
        curr = df["usage_current"]
    elif "Ini" in df.columns and "Lalu" in df.columns:
        curr = pd.to_numeric(df["Ini"], errors="coerce") - pd.to_numeric(df["Lalu"], errors="coerce")
    elif "usage_original" in df.columns:
        curr = df["usage_original"]
    else:
        curr = pd.Series(np.nan, index=df.index)

    temp = pd.DataFrame({
        "target_next_usage": df["target_next_usage"],
        "curr_usage": curr
    }).dropna()

    y_true = temp["target_next_usage"].values
    y_pred = temp["curr_usage"].values   # Lag-1 baseline

    metrics = compute_regression_metrics(y_true, y_pred, label="Baseline_Lag1")
    return metrics


# =============================================================================
# COMPARE FEATURE SCENARIOS
# =============================================================================

def compare_feature_scenarios(
    results_list: List[Dict],
) -> pd.DataFrame:
    """
    Buat tabel perbandingan S0 vs S1 dari list hasil metrik.

    Parameters
    ----------
    results_list : List[Dict]
        Setiap dict harus memiliki key: label, n_obs, rmse, mae, mare, nsc, r2.
        Tambahkan key 'n_features' dan 'scenario' untuk informasi lengkap.

    Returns
    -------
    pd.DataFrame tabel perbandingan.
    """
    df = pd.DataFrame(results_list)
    return df


def compute_delta_comparison(
    df_comparison: pd.DataFrame,
    baseline_label: str = "Baseline_Lag1",
) -> pd.DataFrame:
    """
    Hitung delta perubahan metrik relatif terhadap baseline.

    Returns
    -------
    pd.DataFrame dengan kolom delta_* tambahan.
    """
    df = df_comparison.copy()

    if baseline_label in df["label"].values:
        baseline = df[df["label"] == baseline_label].iloc[0]

        for metric in ["rmse", "mae", "nsc", "r2"]:
            if metric in df.columns and baseline.get(metric) is not None:
                df[f"delta_{metric}"] = df[metric] - baseline[metric]
                df[f"delta_{metric}_pct"] = (
                    (df[metric] - baseline[metric]) / abs(baseline[metric]) * 100
                ).round(2)

    return df


# =============================================================================
# OBSERVED-ONLY EVALUATION
# =============================================================================

def evaluate_observed_only(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    observed_flag_col: str = "zero_original_flag",
) -> Dict:
    """
    Evaluasi hanya pada subset test yang observed (bukan hasil imputasi).

    Parameter observed_flag_col menentukan baris yang tidak diimputasi.
    Dalam konteks ini: baris dengan imputation_required_flag=0.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test set dengan kolom 'target_next_usage' dan flag kualitas.
    y_pred : np.ndarray
        Prediksi (aligned dengan df_test).
    observed_flag_col : str
        Kolom flag yang menandai baris bermasalah.

    Returns
    -------
    dict metrik pada observed-only subset.
    """
    # Filter: hanya baris yang tidak butuh imputasi
    if "imputation_required_flag" in df_test.columns:
        mask = df_test["imputation_required_flag"] == 0
    else:
        mask = pd.Series(True, index=df_test.index)

    df_obs = df_test[mask].copy()
    y_obs  = df_obs["target_next_usage"].values
    y_pred_obs = y_pred[mask.values]

    metrics = compute_regression_metrics(
        y_obs, y_pred_obs, label="observed_only"
    )
    logger.info(f"Observed-only test: {mask.sum():,} dari {len(df_test):,} baris")
    return metrics


# =============================================================================
# KMEANS EVALUATION
# =============================================================================

def evaluate_kmeans_scenarios(
    results_all: pd.DataFrame,      # k-search results S0
    results_selected: pd.DataFrame, # k-search results S1
    k_final_all: int,
    k_final_selected: int,
) -> pd.DataFrame:
    """
    Buat tabel perbandingan K-Means S0 vs S1 untuk K final masing-masing.

    Returns
    -------
    pd.DataFrame
    """
    row_all = results_all[results_all["K"] == k_final_all].iloc[0].to_dict()
    row_sel = results_selected[results_selected["K"] == k_final_selected].iloc[0].to_dict()

    row_all["scenario"] = config.SCENARIO_ALL
    row_sel["scenario"] = config.SCENARIO_SELECTED

    df_cmp = pd.DataFrame([row_all, row_sel])
    return df_cmp


# =============================================================================
# FINAL PIPELINE COMPARISON
# =============================================================================

def build_final_pipeline_comparison(
    kmeans_comparison: pd.DataFrame,
    ann_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gabungkan tabel perbandingan K-Means dan ANN menjadi satu laporan akhir.
    """
    kmeans_comparison = kmeans_comparison.copy()
    kmeans_comparison["model_type"] = "K-Means"

    ann_comparison = ann_comparison.copy()
    ann_comparison["model_type"] = "ANN"

    # Pilih kolom yang overlap
    common_cols = ["model_type", "scenario", "label"]
    kmeans_cols = [c for c in kmeans_comparison.columns if c in
                   ["scenario", "K", "silhouette", "dbi", "chi", "inertia"]]
    ann_cols    = [c for c in ann_comparison.columns if c in
                   ["scenario", "label", "n_features", "rmse", "mae", "mare", "nsc", "r2"]]

    df_k = kmeans_comparison[["model_type"] + kmeans_cols]
    df_a = ann_comparison[["model_type"] + ann_cols]

    return pd.concat([df_k, df_a], ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("evaluation.py loaded OK")

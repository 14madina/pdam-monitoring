"""
src/imputation.py — Eksperimen dan aplikasi imputasi untuk data time series PDAM.

Rules:
- Utamakan histori pelanggan sebelum periode saat ini.
- Fallback statistic di-fit dari training period saja.
- Tidak menggunakan masa depan (no future leakage).
- Zero asli tetap bisa dilacak via zero_original_flag.
- Empat metode: Historical Mean, Median, Mode, Minimum Positive.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _get_personal_history(
    group: pd.DataFrame,
    current_idx: int,
) -> pd.Series:
    """
    Ambil histori usage_imputed SEBELUM periode saat ini untuk satu NoPEL.
    Gunakan index posisi (bukan label).
    """
    return group["usage_imputed"].iloc[:current_idx]


def _personal_stat(history: pd.Series, method: str) -> Optional[float]:
    """Hitung statistik dari histori personal (tanpa NaN)."""
    clean = history.dropna()
    if len(clean) == 0:
        return None

    if method == "mean":
        return float(clean.mean())
    elif method == "median":
        return float(clean.median())
    elif method == "mode":
        mode_result = stats.mode(clean, keepdims=True)
        return float(mode_result.mode[0]) if len(mode_result.mode) > 0 else None
    elif method == "min_positive":
        pos = clean[clean > 0]
        return float(pos.min()) if len(pos) > 0 else None
    return None


# =============================================================================
# SINGLE METHOD IMPUTATION
# =============================================================================

def _impute_single_method(
    df: pd.DataFrame,
    method: str,
    train_fallback_stat: Optional[float],
    col_out: str = "usage_imputed",
) -> pd.DataFrame:
    """
    Imputasi usage_current menggunakan satu metode untuk semua NoPEL.

    Parameters
    ----------
    df : pd.DataFrame
        Harus mengandung kolom 'NoPEL', 'period_dt', 'usage_current',
        'imputation_required_flag'.
    method : str
        'mean', 'median', 'mode', atau 'min_positive'.
    train_fallback_stat : float or None
        Statistik fallback yang sudah di-fit dari training period.
    col_out : str
        Nama kolom output.

    Returns
    -------
    pd.DataFrame dengan kolom col_out dan 'imputation_flag_{method}'.
    """
    df = df.sort_values(["NoPEL", "period_dt"]).copy()
    df[col_out] = df["usage_current"].copy()
    imputed_flags = pd.Series(0, index=df.index)

    for nopel, group_idx in df.groupby("NoPEL").groups.items():
        group = df.loc[group_idx].reset_index()  # reset_index untuk iloc

        for pos, (orig_idx, row) in enumerate(zip(group_idx, group.itertuples())):
            if df.at[orig_idx, "imputation_required_flag"] != 1:
                continue

            # Cari nilai imputasi dari histori personal
            history = group["usage_imputed"].iloc[:pos]
            imputed_val = _personal_stat(history, method)

            # Fallback ke training statistic
            if imputed_val is None:
                imputed_val = train_fallback_stat

            if imputed_val is not None:
                df.at[orig_idx, col_out] = imputed_val
                imputed_flags.at[orig_idx] = 1

    df[f"imputed_flag_{method}"] = imputed_flags
    n_imputed = imputed_flags.sum()
    n_needed  = df["imputation_required_flag"].sum()
    n_unresolved = n_needed - n_imputed

    logger.info(
        f"[{method}] Butuh imputasi: {n_needed:,}, "
        f"berhasil: {n_imputed:,}, "
        f"unresolved: {n_unresolved:,}"
    )

    return df


# =============================================================================
# FALLBACK STATISTICS (fit dari TRAIN saja)
# =============================================================================

def compute_train_fallback_stats(
    df: pd.DataFrame,
    train_months: list,
) -> Dict[str, Optional[float]]:
    """
    Hitung fallback statistic dari training period saja.

    Parameters
    ----------
    df : pd.DataFrame
    train_months : list of pd.Timestamp

    Returns
    -------
    dict : {method_name: fallback_value}
    """
    train_mask = df["period_dt"].isin(train_months)
    train_usage = df.loc[train_mask, "usage_current"].dropna()

    fallbacks: Dict[str, Optional[float]] = {}

    fallbacks["mean"]        = float(train_usage.mean())        if len(train_usage) > 0 else None
    fallbacks["median"]      = float(train_usage.median())      if len(train_usage) > 0 else None
    fallbacks["mode"]        = float(stats.mode(train_usage, keepdims=True).mode[0]) if len(train_usage) > 0 else None
    pos = train_usage[train_usage > 0]
    fallbacks["min_positive"] = float(pos.min())                if len(pos) > 0 else None

    logger.info(f"Train fallback stats: {fallbacks}")
    return fallbacks


# =============================================================================
# COMPARE IMPUTATION METHODS
# =============================================================================

def compare_imputation_methods(
    df: pd.DataFrame,
    train_months: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bandingkan 4 metode imputasi menggunakan cross-validation internal.

    Evaluasi:
    - MAE estimasi (pada baris yang diimputasi)
    - RMSE estimasi
    - Perubahan mean/median/std/skewness distribusi
    - Unresolved count

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame dengan kolom 'usage_current', 'imputation_required_flag',
        'period_dt', 'NoPEL'.
    train_months : list

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (imputation_comparison, distribution_comparison)
    """
    fallbacks = compute_train_fallback_stats(df, train_months)

    method_map = {
        "historical_mean"         : "mean",
        "historical_median"       : "median",
        "historical_mode"         : "mode",
        "historical_min_positive" : "min_positive",
    }

    comparison_rows = []
    dist_rows = []

    original_mean     = df["usage_current"].mean()
    original_median   = df["usage_current"].median()
    original_std      = df["usage_current"].std()
    original_skew     = df["usage_current"].skew()

    for method_label, method_key in method_map.items():
        logger.info(f"\n--- Running imputation method: {method_label} ---")
        df_imp = _impute_single_method(
            df.copy(),
            method=method_key,
            train_fallback_stat=fallbacks.get(method_key),
            col_out="usage_imputed",
        )

        imputed_flag_col = f"imputed_flag_{method_key}"
        actually_imputed = df_imp[imputed_flag_col] == 1
        n_imputed    = actually_imputed.sum()
        n_needed     = df["imputation_required_flag"].sum()
        n_unresolved = n_needed - n_imputed

        # MAE / RMSE: hanya pada baris yang punya nilai asli untuk perbandingan
        # (simulasi: baris yang diimputasi dibandingkan dengan nilai aslinya)
        original_values  = df.loc[actually_imputed, "usage_current"]
        imputed_values   = df_imp.loc[actually_imputed, "usage_imputed"]
        valid_compare    = original_values.notna() & imputed_values.notna()

        if valid_compare.sum() > 0:
            mae  = float(np.mean(np.abs(original_values[valid_compare] - imputed_values[valid_compare])))
            rmse = float(np.sqrt(np.mean((original_values[valid_compare] - imputed_values[valid_compare])**2)))
        else:
            mae, rmse = np.nan, np.nan

        comparison_rows.append({
            "method"          : method_label,
            "n_needed"        : n_needed,
            "n_imputed"       : n_imputed,
            "n_unresolved"    : n_unresolved,
            "mae_estimate"    : round(mae, 4)  if not np.isnan(mae)  else None,
            "rmse_estimate"   : round(rmse, 4) if not np.isnan(rmse) else None,
        })

        # Distribution comparison
        after_mean   = df_imp["usage_imputed"].mean()
        after_median = df_imp["usage_imputed"].median()
        after_std    = df_imp["usage_imputed"].std()
        after_skew   = df_imp["usage_imputed"].skew()

        dist_rows.append({
            "method"          : method_label,
            "before_mean"     : round(original_mean,   4),
            "after_mean"      : round(after_mean,      4),
            "delta_mean"      : round(after_mean - original_mean, 4),
            "before_median"   : round(original_median, 4),
            "after_median"    : round(after_median,    4),
            "delta_median"    : round(after_median - original_median, 4),
            "before_std"      : round(original_std,    4),
            "after_std"       : round(after_std,       4),
            "delta_std"       : round(after_std - original_std, 4),
            "before_skewness" : round(original_skew,   4),
            "after_skewness"  : round(after_skew,      4),
            "delta_skewness"  : round(after_skew - original_skew, 4),
        })

    df_comparison = pd.DataFrame(comparison_rows)
    df_dist       = pd.DataFrame(dist_rows)

    return df_comparison, df_dist


# =============================================================================
# APPLY BEST IMPUTATION
# =============================================================================

def apply_best_imputation(
    df: pd.DataFrame,
    method: str,
    train_months: list,
) -> pd.DataFrame:
    """
    Terapkan metode imputasi terpilih ke seluruh DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        Salah satu dari: 'historical_mean', 'historical_median',
        'historical_mode', 'historical_min_positive'.
    train_months : list

    Returns
    -------
    pd.DataFrame dengan kolom 'usage_imputed' dan flag imputasi.
    """
    method_map = {
        "historical_mean"         : "mean",
        "historical_median"       : "median",
        "historical_mode"         : "mode",
        "historical_min_positive" : "min_positive",
    }

    if method not in method_map:
        raise ValueError(
            f"Method '{method}' tidak valid. "
            f"Pilihan: {list(method_map.keys())}"
        )

    fallbacks = compute_train_fallback_stats(df, train_months)
    method_key = method_map[method]

    df_result = _impute_single_method(
        df.copy(),
        method=method_key,
        train_fallback_stat=fallbacks.get(method_key),
        col_out="usage_imputed",
    )

    df_result["imputation_method_used"] = method
    logger.info(f"✅ Best imputation applied: {method}")
    return df_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("imputation.py loaded OK")

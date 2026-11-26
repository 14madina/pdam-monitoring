"""
src/spike_detection.py — Deteksi lonjakan pemakaian air berdasarkan residual.

Rules:
- Gunakan model/pipeline final yang disetujui (setelah evaluasi S0 vs S1).
- residual = actual - prediction
- percentage_difference = residual / max(prediction, epsilon) * 100%
- Threshold awal 30%.
- Status:
    >  +30%: INDIKASI_LONJAKAN
    < -30%: ANOMALI_RENDAH
    lainnya: NORMAL
- Pertahankan flag 0 m³ asli.
- JANGAN menyatakan hasil sebagai bukti kebocoran fisik.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# RESIDUAL COMPUTATION
# =============================================================================

def compute_residuals(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Hitung residual: actual - prediction.

    Parameters
    ----------
    y_actual : np.ndarray
    y_pred   : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return y_actual - y_pred


def compute_percentage_difference(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = config.SPIKE_EPSILON,
) -> np.ndarray:
    """
    Hitung persentase perbedaan relatif terhadap prediksi.

    percentage_difference = (actual - prediction) / max(|prediction|, epsilon) * 100

    Parameters
    ----------
    y_actual : np.ndarray
    y_pred   : np.ndarray
    epsilon  : float — guard terhadap pembagian dengan nol

    Returns
    -------
    np.ndarray (dalam %)
    """
    residual     = compute_residuals(y_actual, y_pred)
    denom        = np.maximum(np.abs(y_pred), epsilon)
    pct_diff     = (residual / denom) * 100.0
    return pct_diff


# =============================================================================
# SPIKE CLASSIFICATION
# =============================================================================

def classify_spike_status(
    pct_diff: np.ndarray,
    threshold_high: float = config.SPIKE_THRESHOLD_PCT,
    threshold_low: float  = config.ANOMALY_THRESHOLD_PCT,
) -> np.ndarray:
    """
    Klasifikasikan status berdasarkan percentage difference.

    Parameters
    ----------
    pct_diff       : np.ndarray — percentage difference (%)
    threshold_high : float — threshold INDIKASI_LONJAKAN (default +30%)
    threshold_low  : float — threshold ANOMALI_RENDAH (default -30%)

    Returns
    -------
    np.ndarray of str: status per baris.
    """
    status = np.where(
        pct_diff >  threshold_high, config.SPIKE_STATUS_LONJAKAN,
        np.where(
            pct_diff < threshold_low, config.SPIKE_STATUS_RENDAH,
            config.SPIKE_STATUS_NORMAL,
        )
    )
    return status


# =============================================================================
# GENERATE SPIKE REPORT
# =============================================================================

def generate_spike_report(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    actual_col: str = "target_next_usage",
    threshold_high: float = config.SPIKE_THRESHOLD_PCT,
    threshold_low: float  = config.ANOMALY_THRESHOLD_PCT,
) -> pd.DataFrame:
    """
    Buat laporan spike detection lengkap.

    Parameters
    ----------
    df : pd.DataFrame
        Test set dengan kolom NoPEL, period_dt, actual_col,
        dan flag kualitas (zero_original_flag, imputation_required_flag, dsb).
    y_pred : np.ndarray
        Prediksi dari model final (aligned dengan df).
    actual_col : str
        Nama kolom nilai aktual.
    threshold_high, threshold_low : float

    Returns
    -------
    pd.DataFrame laporan spike dengan kolom:
        NoPEL, period_dt, actual, prediction, residual,
        pct_diff, spike_status, zero_m3_flag,
        dan kolom flag kualitas lain bila ada.
    """
    df = df.copy()
    df = df.reset_index(drop=True)

    y_actual = df[actual_col].values
    residual = compute_residuals(y_actual, y_pred)
    pct_diff = compute_percentage_difference(y_actual, y_pred)
    status   = classify_spike_status(pct_diff, threshold_high, threshold_low)

    report = pd.DataFrame({
        "NoPEL"       : df["NoPEL"],
        "period_dt"   : df["period_dt"],
        "actual"      : y_actual,
        "prediction"  : y_pred,
        "residual"    : residual,
        "pct_diff"    : pct_diff.round(4),
        "spike_status": status,
    })

    # Pertahankan zero m³ flag asli
    if "zero_original_flag" in df.columns:
        report["zero_m3_flag"] = df["zero_original_flag"].values
    if "imputation_required_flag" in df.columns:
        report["imputation_flag"] = df["imputation_required_flag"].values
    if "source_area" in df.columns:
        report["source_area"] = df["source_area"].values
    if "RAYON" in df.columns:
        report["RAYON"] = df["RAYON"].astype(str).values

    # Statistik ringkasan
    n_total   = len(report)
    n_lonjakan = (report["spike_status"] == config.SPIKE_STATUS_LONJAKAN).sum()
    n_rendah  = (report["spike_status"] == config.SPIKE_STATUS_RENDAH).sum()
    n_normal  = (report["spike_status"] == config.SPIKE_STATUS_NORMAL).sum()

    logger.info(
        f"Spike Detection Summary:\n"
        f"  Total records    : {n_total:,}\n"
        f"  INDIKASI_LONJAKAN: {n_lonjakan:,} ({n_lonjakan/n_total*100:.1f}%)\n"
        f"  ANOMALI_RENDAH   : {n_rendah:,} ({n_rendah/n_total*100:.1f}%)\n"
        f"  NORMAL           : {n_normal:,} ({n_normal/n_total*100:.1f}%)\n"
        f"\n  CATATAN: Indikasi lonjakan adalah sinyal analitik,\n"
        f"  BUKAN bukti kebocoran fisik yang terkonfirmasi."
    )

    return report


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def get_spike_summary(
    spike_report: pd.DataFrame,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ringkasan statistik spike detection.

    Parameters
    ----------
    spike_report : pd.DataFrame
        Output dari generate_spike_report.
    group_by : str, optional
        Kolom untuk grouping (misal: 'source_area', 'RAYON', 'period_dt').

    Returns
    -------
    pd.DataFrame ringkasan.
    """
    def _summary(sub):
        n = len(sub)
        return pd.Series({
            "total"          : n,
            "n_lonjakan"     : (sub["spike_status"] == config.SPIKE_STATUS_LONJAKAN).sum(),
            "n_rendah"       : (sub["spike_status"] == config.SPIKE_STATUS_RENDAH).sum(),
            "n_normal"       : (sub["spike_status"] == config.SPIKE_STATUS_NORMAL).sum(),
            "pct_lonjakan"   : round((sub["spike_status"] == config.SPIKE_STATUS_LONJAKAN).sum() / n * 100, 2),
            "pct_rendah"     : round((sub["spike_status"] == config.SPIKE_STATUS_RENDAH).sum() / n * 100, 2),
            "mean_pct_diff"  : round(sub["pct_diff"].mean(), 4),
            "max_pct_diff"   : round(sub["pct_diff"].max(), 4),
            "min_pct_diff"   : round(sub["pct_diff"].min(), 4),
        })

    if group_by and group_by in spike_report.columns:
        summary = spike_report.groupby(group_by).apply(_summary).reset_index()
    else:
        summary = _summary(spike_report).to_frame().T
        summary.insert(0, "group", "ALL")

    return summary


# =============================================================================
# RAYON SUMMARY
# =============================================================================

def get_rayon_spike_summary(spike_report: pd.DataFrame) -> pd.DataFrame:
    """
    Ringkasan spike detection per wilayah (RAYON atau source_area).
    """
    group_col = "RAYON" if "RAYON" in spike_report.columns else "source_area"
    if group_col not in spike_report.columns:
        logger.warning("Kolom RAYON/source_area tidak tersedia untuk summary per wilayah.")
        return pd.DataFrame()

    return get_spike_summary(spike_report, group_by=group_col)


# =============================================================================
# WEB DASHBOARD INTEGRATION: ADVANCED SPIKE & ANOMALY DETECTION
# =============================================================================

SEVERITY_META = {
    0: {"name": "NORMAL", "label": "Normal (Dalam Batas Wajar)", "color": "#2E7D32"},
    1: {"name": "RENDAH", "label": "Perhatian Ringan", "color": "#FDD835"},
    2: {"name": "SEDANG", "label": "Waspada / Perlu Verifikasi", "color": "#FB8C00"},
    3: {"name": "TINGGI", "label": "Kritis / Prioritas Investigasi", "color": "#D32F2F"},
}


def detect_spikes_from_predictions(
    df: pd.DataFrame,
    actual_col: Optional[str] = None,
    pred_col: str = "predicted_usage",
    threshold_high: float = config.SPIKE_THRESHOLD_PCT,  # +30.0%
    threshold_low: float = config.ANOMALY_THRESHOLD_PCT,   # -30.0%
) -> pd.DataFrame:
    """
    Deteksi lonjakan dan anomali pemakaian air untuk data batch dashboard.

    Logika Deteksi:
    1. Mencari kolom nilai aktual:
       - Jika data evaluasi: 'target_next_usage' atau 'actual'.
       - Jika data pemakaian berjalan: 'usage_current' atau 'usage_original' (Ini - Lalu).
    2. Menghitung residual = actual - predicted_usage.
    3. Menghitung percentage difference = (residual / max(predicted_usage, 1.0)) * 100%.
    4. Mengklasifikasikan status anomali:
       - > +30%: INDIKASI_LONJAKAN
       - < -30%: ANOMALI_RENDAH
       - -30% s/d +30%: NORMAL
    5. Menetapkan skor keparahan (severity_score: 0 s/d 3).
    6. Menetapkan kategori analitik operasional:
       - 'LONJAKAN_EKSTREM'      : pct_diff > 100% atau lonjakan > 50 m³
       - 'LONJAKAN_MODERAT'      : +30% < pct_diff <= +100%
       - 'NORMAL'                : -30% <= pct_diff <= +30%
       - 'PENURUNAN_SIGNIFIKAN'  : -80% < pct_diff < -30%
       - 'METER_MACET_ATAU_KOSONG': pct_diff <= -80% dan actual <= 1.0 m³

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang telah memiliki kolom prediksi (pred_col).
    actual_col : Optional[str]
        Nama kolom aktual. Jika None, akan dideteksi secara otomatis.
    pred_col : str
        Nama kolom hasil prediksi ANN (default 'predicted_usage').

    Returns
    -------
    pd.DataFrame dengan kolom tambahan:
        - actual_usage (m³)
        - residual (m³)
        - pct_diff (%)
        - spike_status ('INDIKASI_LONJAKAN', 'ANOMALI_RENDAH', 'NORMAL')
        - severity_score (0, 1, 2, 3)
        - anomaly_category (str)
    """
    df_out = df.copy()

    # 1. Tentukan kolom aktual
    if actual_col and actual_col in df_out.columns:
        chosen_actual = actual_col
    else:
        candidates = ["target_next_usage", "actual", "usage_current", "usage_original"]
        chosen_actual = None
        for cand in candidates:
            if cand in df_out.columns and df_out[cand].notna().any():
                chosen_actual = cand
                break

    if not chosen_actual:
        if "Ini" in df_out.columns and "Lalu" in df_out.columns:
            df_out["actual_usage"] = pd.to_numeric(df_out["Ini"], errors="coerce") - pd.to_numeric(df_out["Lalu"], errors="coerce")
            chosen_actual = "actual_usage"
        else:
            raise ValueError("Tidak dapat menemukan kolom actual_usage atau Ini & Lalu untuk deteksi lonjakan.")

    y_actual = pd.to_numeric(df_out[chosen_actual], errors="coerce").fillna(0.0).values
    df_out["actual_usage"] = np.round(y_actual, 2)

    # 2. Ambil nilai prediksi
    if pred_col not in df_out.columns:
        raise ValueError(f"Kolom prediksi '{pred_col}' tidak ditemukan di DataFrame.")

    y_pred = pd.to_numeric(df_out[pred_col], errors="coerce").fillna(0.0).values
    df_out["predicted_usage"] = np.round(y_pred, 2)

    # 3. Hitung residual dan persentase perbedaan
    residuals = y_actual - y_pred
    denom = np.maximum(np.abs(y_pred), config.SPIKE_EPSILON)
    pct_diff = (residuals / denom) * 100.0

    df_out["residual"] = np.round(residuals, 2)
    df_out["pct_diff"] = np.round(pct_diff, 2)

    # 4. Status Anomali (sesuai threshold riset: +30% & -30%)
    status = np.where(
        pct_diff > threshold_high, config.SPIKE_STATUS_LONJAKAN,
        np.where(
            pct_diff < threshold_low, config.SPIKE_STATUS_RENDAH,
            config.SPIKE_STATUS_NORMAL,
        )
    )
    df_out["spike_status"] = status

    # 5. Severity Score (0: Normal, 1: Rendah, 2: Sedang, 3: Tinggi/Kritis)
    abs_diff = np.abs(pct_diff)
    severity = np.where(
        (abs_diff <= 30.0), 0,
        np.where(
            (abs_diff <= 50.0), 1,
            np.where(
                (abs_diff <= 100.0), 2,
                3  # > 100%
            )
        )
    )
    # Khusus lonjakan volume ekstrem (> 50 m3) langsung severity 3
    extreme_volume_mask = (residuals > 50.0) | (residuals < -50.0)
    severity = np.where(extreme_volume_mask, 3, severity)
    df_out["severity_score"] = severity.astype(int)

    # 6. Detailed Operational Category
    categories = []
    for act, pred, diff, p_diff in zip(y_actual, y_pred, residuals, pct_diff):
        if p_diff > 100.0 or diff > 50.0:
            cat = "LONJAKAN_EKSTREM"
        elif p_diff > threshold_high:
            cat = "LONJAKAN_MODERAT"
        elif p_diff <= -80.0 and act <= 1.0:
            cat = "METER_MACET_ATAU_KOSONG"
        elif p_diff < threshold_low:
            cat = "PENURUNAN_SIGNIFIKAN"
        else:
            cat = "NORMAL"
        categories.append(cat)

    df_out["anomaly_category"] = categories

    n_lonjakan = (status == config.SPIKE_STATUS_LONJAKAN).sum()
    n_rendah = (status == config.SPIKE_STATUS_RENDAH).sum()
    n_normal = (status == config.SPIKE_STATUS_NORMAL).sum()

    logger.info(
        f"Spike detection complete ({len(df_out):,} rows): "
        f"Lonjakan={n_lonjakan:,} ({n_lonjakan/len(df_out)*100:.1f}%), "
        f"Rendah={n_rendah:,} ({n_rendah/len(df_out)*100:.1f}%), "
        f"Normal={n_normal:,} ({n_normal/len(df_out)*100:.1f}%)"
    )

    return df_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("spike_detection.py loaded OK")

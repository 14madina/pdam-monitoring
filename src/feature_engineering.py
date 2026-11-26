"""
src/feature_engineering.py — One-Time Feature Extraction dari master_clean.

Rules:
- Feature extraction dilakukan SATU KALI dari data bersih.
- Semua rolling/lag features menggunakan shift(1) agar past-only (no leakage).
- target_next_usage hanya dibuat jika next record benar-benar bulan berikutnya.
- Urutan: NoPEL → period_dt dipertahankan.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN FEATURE EXTRACTION — HANYA DARI DATA GABUNGAN + TARIF TERHITUNG
# =============================================================================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ekstraksi fitur murni HANYA dari kolom data gabungan + kolom tarif saja.
    TIDAK ada kolom tambahan lain seperti golongan_tarif, estimated_bill_rp, lag, rolling, delta, dsb.
    Kolom data gabungan yang dipertahankan:
    tglBaca, Waktu, NoPEL, Nama, Alamat, Lalu, Ini, Cater, RAYON, Status,
    periode, keterangan, lati, longi, alti, time, updatedDate, updatedBy
    ditambah kolom tambahan:
    tarif
    dan kolom target pemodelan: target_next_usage.

    Input
    -----
    df : pd.DataFrame
        Master bersih dari master_clean.parquet

    Output
    ------
    pd.DataFrame berisi kolom data gabungan + tarif dan target_next_usage.
    """
    logger.info("Extracting features strictly from merged dataset + tarif column only ...")
    df = df.copy()

    # Pastikan period_dt ada untuk sorting kronologis
    if "period_dt" not in df.columns and "periode" in df.columns:
        df["period_dt"] = pd.to_datetime(df["periode"].astype(str), format="%Y%m", errors="coerce")

    df = df.sort_values(["NoPEL", "period_dt"]).copy()

    # Pastikan koordinat ada di 'lati' dan 'longi' (nama asli di data)
    if "lati" not in df.columns and "latitude" in df.columns:
        df["lati"] = df["latitude"]
    if "longi" not in df.columns and "longitude" in df.columns:
        df["longi"] = df["longitude"]

    # Pastikan kolom numerik bertipe numerik
    numeric_cols = ["Lalu", "Ini", "tarif", "lati", "longi", "alti", "time"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tambahkan target_next_usage
    df = _add_target(df)

    # Filter HANYA kolom data gabungan + tarif + period_dt + target_next_usage
    allowed_cols = [
        "NoPEL", "Nama", "Alamat", "tglBaca", "Waktu", "Lalu", "Ini",
        "tarif",
        "Cater", "RAYON", "Status", "periode", "keterangan",
        "lati", "longi", "alti", "time", "updatedDate", "updatedBy",
        "period_dt", config.TARGET_COL
    ]
    cols_to_keep = [c for c in allowed_cols if c in df.columns]
    df = df[cols_to_keep].copy()

    n_features = len([c for c in df.columns if c in config.ALL_CANDIDATE_FEATURES])
    logger.info(
        f"✅ Feature extraction complete. Total records: {len(df):,}, "
        f"candidate features dari data gabungan: {n_features}"
    )

    return df



# =============================================================================
# TARGET — target_next_usage
# =============================================================================

def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat target_next_usage: pemakaian air periode berikutnya (next_Ini - next_Lalu).

    Target hanya valid jika:
    - Record berikutnya benar-benar milik NoPEL yang sama, DAN
    - period_dt berikutnya = period_dt saat ini + 1 bulan (toleransi ±15 hari).

    Jika tidak valid, target_next_usage = NaN.
    """
    df = df.sort_values(["NoPEL", "period_dt"]).copy()

    # Hitung pemakaian bulan ini: Ini - Lalu
    if "Ini" in df.columns and "Lalu" in df.columns:
        curr_usage = pd.to_numeric(df["Ini"], errors="coerce") - pd.to_numeric(df["Lalu"], errors="coerce")
    elif "usage_current" in df.columns:
        curr_usage = df["usage_current"]
    elif "usage_original" in df.columns:
        curr_usage = df["usage_original"]
    else:
        curr_usage = pd.Series(np.nan, index=df.index)

    next_usage  = curr_usage.groupby(df["NoPEL"]).shift(-1)
    next_period = df.groupby("NoPEL")["period_dt"].shift(-1)
    next_nopel  = df.groupby("NoPEL")["NoPEL"].shift(-1)

    # Validasi: NoPEL sama dan periode = +1 bulan
    period_diff = (next_period - df["period_dt"]).dt.days

    valid_target = (
        (next_nopel == df["NoPEL"]) &
        (period_diff >= 16) &  # minimal setengah bulan
        (period_diff <= 46)    # maksimal ~1.5 bulan
    )

    df[config.TARGET_COL] = np.where(valid_target, next_usage, np.nan)

    n_valid_target = df[config.TARGET_COL].notna().sum()
    logger.info(
        f"target_next_usage: {n_valid_target:,} valid dari {len(df):,} records "
        f"({n_valid_target/len(df)*100:.1f}%)"
    )

    return df


# =============================================================================
# FEATURE LIST HELPERS
# =============================================================================

def get_available_features(df: pd.DataFrame) -> List[str]:
    """
    Kembalikan daftar candidate features yang benar-benar ada di DataFrame
    dan hanya bersumber dari data gabungan + tarif terhitung (config.ALL_CANDIDATE_FEATURES).
    """
    available = [
        c for c in config.ALL_CANDIDATE_FEATURES
        if c in df.columns
    ]
    return available



def check_leakage(feature_list: List[str], target: str = config.TARGET_COL) -> List[str]:
    """
    Periksa daftar fitur dari leakage sederhana.
    Kembalikan daftar fitur yang mencurigakan.
    """
    suspicious = []
    leak_keywords = ["next", "future", "t+1", "target"]

    for f in feature_list:
        for kw in leak_keywords:
            if kw in f.lower() and f != target:
                suspicious.append(f)
                break

    if suspicious:
        logger.warning(f"Fitur yang dicurigai mengandung leakage: {suspicious}")

    return suspicious


# =============================================================================
# WEB DASHBOARD INTEGRATION: INFERENCE FEATURE PREPARATION & TIME-SERIES CONTEXT
# =============================================================================

def prepare_model_input(
    df: pd.DataFrame,
    scenario: str = "S1",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ekstrak dan susun matriks fitur input untuk inference model K-Means & ANN.
    
    Menjamin urutan fitur SAMA PERSIS dengan konfigurasi training:
    - Skenario S1 (Selected Features): ['Ini', 'Lalu', 'tarif']
    - Skenario S0 (All Features)     : ['Ini', 'Lalu', 'tarif', 'lati', 'longi', 'alti', 'time']

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang telah divalidasi dan distandardisasi.
    scenario : str, optional ('S1' atau 'S0', default 'S1')
        Skenario model penelitian yang digunakan.

    Returns
    -------
    (X_df, feature_names)
        X_df : pd.DataFrame berisi kolom fitur dengan tipe float64 terurut.
        feature_names : List[str] daftar nama fitur yang digunakan.
    """
    if scenario.upper() == "S1":
        feature_cols = config.SELECTED_FEATURES  # ['Ini', 'Lalu', 'tarif']
    elif scenario.upper() == "S0":
        feature_cols = config.ALL_FEATURES       # 7 fitur
    else:
        raise ValueError(f"Skenario tidak dikenal: {scenario}. Pilih 'S1' atau 'S0'.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame tidak memiliki fitur yang dibutuhkan untuk skenario {scenario}: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    # Ekstrak kolom sesuai urutan eksak dan pastikan numerik
    X_df = df[feature_cols].copy()
    for col in feature_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").astype(float)
        # Handle NaN jika ada
        if X_df[col].isna().any():
            if col == "tarif":
                X_df[col] = X_df[col].fillna(1.0)
            elif col in ["Ini", "Lalu"]:
                # Imputasi fallback dengan median jika data upload memiliki cell kosong
                med = X_df[col].median()
                X_df[col] = X_df[col].fillna(med if pd.notna(med) else 0.0)
            else:
                X_df[col] = X_df[col].fillna(0.0)

    return X_df, feature_cols


def get_customer_historical_context(
    nopel: str,
    db_path: Optional[Path] = None,
    limit: int = 24,
) -> pd.DataFrame:
    """
    Ambil data historis pemakaian pelanggan untuk membentuk konteks time series.
    Prioritas:
    1. Database SQLite (data/database/pdam_database.db tabel readings)
    2. Fallback: master_features.parquet / master_clean.parquet jika DB belum terisi

    Parameters
    ----------
    nopel : str
        ID Pelanggan (NoPEL), otomatis di-zfill 8 digit.
    db_path : Optional[Path]
        Lokasi file database SQLite.
    limit : int
        Jumlah periode historis maksimum yang diambil (default 24 bulan).

    Returns
    -------
    pd.DataFrame dengan urutan kronologis period_dt, Ini, Lalu, pemakaian_air, tarif.
    """
    clean_nopel = str(nopel).strip().replace(".0", "").zfill(8)

    # 1. Coba dari database SQLite
    try:
        from . import database
        df_hist = database.get_customer_historical_readings(clean_nopel, db_path=db_path)
        if not df_hist.empty:
            df_hist = df_hist.sort_values("period_dt").tail(limit).reset_index(drop=True)
            return df_hist
    except Exception as e:
        logger.debug(f"Pencarian historis di database gagal: {e}")

    # 2. Fallback ke file Parquet jika DB kosong
    parquet_path = config.DATA_PROCESSED / "master_features.parquet"
    if not parquet_path.exists():
        parquet_path = config.DATA_PROCESSED / "master_clean.parquet"

    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq
            # Filter hanya NoPEL yang bersangkutan menggunakan Parquet pushdown filter
            table = pq.read_table(
                parquet_path,
                filters=[("NoPEL", "in", [clean_nopel, int(clean_nopel) if clean_nopel.isdigit() else clean_nopel])],
            )
            df_pq = table.to_pandas()
            if not df_pq.empty:
                df_pq["NoPEL"] = df_pq["NoPEL"].astype(str).str.zfill(8)
                if "period_dt" in df_pq.columns:
                    df_pq["period_dt"] = pd.to_datetime(df_pq["period_dt"])
                elif "periode" in df_pq.columns:
                    df_pq["period_dt"] = pd.to_datetime(df_pq["periode"].astype(str), format="%Y%m", errors="coerce")
                
                if "pemakaian_air" not in df_pq.columns and "Ini" in df_pq.columns and "Lalu" in df_pq.columns:
                    df_pq["pemakaian_air"] = pd.to_numeric(df_pq["Ini"]) - pd.to_numeric(df_pq["Lalu"])

                df_pq = df_pq.sort_values("period_dt").tail(limit).reset_index(drop=True)
                return df_pq
        except Exception as e:
            logger.warning(f"Fallback pembacaan parquet untuk NoPEL {clean_nopel} gagal: {e}")

    return pd.DataFrame()


def calculate_customer_time_series_stats(df_history: pd.DataFrame) -> Dict[str, object]:
    """
    Hitung ringkasan statistik time series dari data historis pelanggan:
    - Rata-rata pemakaian (mean)
    - Standar deviasi (std)
    - Pemakaian minimum & maksimum
    - Tren pemakaian (NAIK, TURUN, STABIL)
    - Koefisien variasi (CV)
    """
    if df_history.empty:
        return {
            "has_history" : False,
            "record_count": 0,
            "mean_usage"  : 0.0,
            "std_usage"   : 0.0,
            "min_usage"   : 0.0,
            "max_usage"   : 0.0,
            "trend"       : "TIDAK ADA DATA HISTORIS",
        }

    # Cari kolom pemakaian
    usage_col = None
    for c in ["pemakaian_air", "usage_original", "usage_current", "usage"]:
        if c in df_history.columns:
            usage_col = c
            break

    if not usage_col and "Ini" in df_history.columns and "Lalu" in df_history.columns:
        df_history["calc_usage"] = pd.to_numeric(df_history["Ini"], errors="coerce") - pd.to_numeric(df_history["Lalu"], errors="coerce")
        usage_col = "calc_usage"

    if not usage_col:
        return {
            "has_history" : False,
            "record_count": len(df_history),
            "trend"       : "KOLOM PEMAKAIAN TIDAK TERSEDIA",
        }

    series = pd.to_numeric(df_history[usage_col], errors="coerce").dropna()
    if series.empty:
        return {
            "has_history" : False,
            "record_count": 0,
            "trend"       : "SEMUA NILAI PEMAKAIAN KOSONG",
        }

    mean_val = float(series.mean())
    std_val  = float(series.std()) if len(series) > 1 else 0.0
    min_val  = float(series.min())
    max_val  = float(series.max())

    # Estimasi tren via linear slope jika minimal 3 data point
    trend_str = "STABIL"
    if len(series) >= 3:
        x = np.arange(len(series))
        y = series.values
        slope, _ = np.polyfit(x, y, 1)
        if slope > 0.5:
            trend_str = "MENINGKAT (TREN NAIK)"
        elif slope < -0.5:
            trend_str = "MENURUN (TREN TURUN)"
        else:
            trend_str = "STABIL / FLUKTUATIF WAJAR"

    return {
        "has_history" : True,
        "record_count": len(series),
        "mean_usage"  : round(mean_val, 2),
        "std_usage"   : round(std_val, 2),
        "min_usage"   : round(min_val, 2),
        "max_usage"   : round(max_val, 2),
        "last_usage"  : round(float(series.iloc[-1]), 2),
        "trend"       : trend_str,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("feature_engineering.py loaded OK")

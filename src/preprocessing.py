"""
src/preprocessing.py — Standardisasi schema, quality flags, dan time-split.

Rules:
- NoPEL sebagai string.
- period_dt sebagai awal bulan (datetime).
- usage_original = Ini - Lalu (numerik).
- 7 quality flags dibuat tanpa mengubah nilai asli.
- Split Train/Val/Test secara kronologis (unique months).
- Identitas (Nama, Alamat, Cater) tidak dijadikan fitur.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA STANDARDIZATION
# =============================================================================

# Mapping nama kolom kasar → nama standar (case-insensitive key)
_COL_ALIASES: Dict[str, str] = {
    "nopel"       : "NoPEL",
    "no_pel"      : "NoPEL",
    "no pel"      : "NoPEL",
    "periode"     : "periode",
    "tglbaca"     : "tglBaca",
    "tgl_baca"    : "tglBaca",
    "waktu"       : "Waktu",
    "alamat"      : "Alamat",
    "lalu"        : "Lalu",
    "ini"         : "Ini",
    "cater"             : "Cater",
    "rayon"             : "RAYON",
    "status"            : "Status",
    "keterangan"        : "keterangan",
    "ket"               : "keterangan",
    "lati"              : "latitude",
    "latitude"          : "latitude",
    "longi"             : "longitude",
    "longitude"         : "longitude",
    "alti"              : "alti",
    "nama"              : "Nama",
    "alamat"            : "Alamat",
    "time"              : "time",
    "updateddate"       : "updatedDate",
    "updated_date"      : "updatedDate",
    "updatedby"         : "updatedBy",
    "updated_by"        : "updatedBy",
    "tarif"             : "tarif",
    "tarif_raw"         : "tarif",
    "golongan_tarif"    : "golongan_tarif",
    "estimated_bill_rp" : "estimated_bill_rp",
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename kolom berdasarkan alias map (case-insensitive)."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in _COL_ALIASES:
            rename_map[col] = _COL_ALIASES[key]
    return df.rename(columns=rename_map)


def clean_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan kolom-kolom mentah yang duplikat atau tidak relevan:
    - Hapus duplikasi kolom case-insensitive (misal: 'alamat' jika sudah ada 'Alamat').
    - Hapus HANYA kolom ID internal database mentah: '_id', '_id.1', 'kode', 'kode_unit'.
    - PERTAHANKAN semua kolom data: NoPEL, Nama, Alamat, Lalu, Ini, Cater, RAYON, Status,
      periode, keterangan, lati, longi, alti, time, updatedDate, updatedBy.
    """
    df = df.copy()
    cols_seen = {}
    cols_to_keep = []
    for c in df.columns:
        cl = c.strip().lower()
        if cl not in cols_seen:
            cols_seen[cl] = c
            cols_to_keep.append(c)
    df = df[cols_to_keep].copy()

    drop_unwanted = ["_id", "_id.1", "kode", "kode_unit"]
    drops = [c for c in df.columns if c.lower().strip() in drop_unwanted]
    if drops:
        df = df.drop(columns=drops)
    return df


def assign_file_tariff_and_golongan(
    df: pd.DataFrame,
    df_tarif_ref: pd.DataFrame,
    area_name: str = "",
    known_tarif_map: Optional[Dict[str, int]] = None,
    default_tarif_code: int = 10,
) -> pd.DataFrame:
    """
    Sebelum data digabung, lakukan penentuan tarif, penambahan kolom golongan tarif,
    dan perhitungan tagihan tarif (estimated_bill_rp) untuk setiap file Excel:
    1. Jika kolom 'tarif' sudah ada (misal PERKOTAAN), gunakan nilai tarif tersebut.
    2. Jika kolom 'tarif' belum ada (misal BUKATEJA, KEMANGKON 1/2/3):
       - Lookup NoPEL ke known_tarif_map.
       - Jika belum ada, klasifikasi cerdas dari Nama / Alamat pelanggan.
       - Default pelanggan kecamatan: default_tarif_code (10: R3 Rumah Tangga C).
    3. Tambahkan kolom 'golongan_tarif' di samping kolom 'tarif'.
    4. Hitung tagihan tarif 'estimated_bill_rp' dengan rumus tarif progresif Perkotaan.
    """
    df = clean_raw_columns(df)
    df = _rename_columns(df)

    # Pastikan NoPEL berupa string
    if "NoPEL" in df.columns:
        df["NoPEL"] = df["NoPEL"].astype(str).str.strip()

    # Pastikan Lalu & Ini numerik dan hitung usage_original
    if "Ini" in df.columns and "Lalu" in df.columns:
        ini = pd.to_numeric(df["Ini"], errors="coerce")
        lalu = pd.to_numeric(df["Lalu"], errors="coerce")
        df["Lalu"] = lalu
        df["Ini"] = ini
        df["usage_original"] = ini - lalu
    elif "usage_original" not in df.columns:
        df["usage_original"] = np.nan

    # 1. Tentukan tarif_code
    if "tarif" in df.columns and df["tarif"].notna().sum() > 0:
        tarif_codes = pd.to_numeric(df["tarif"], errors="coerce")
    else:
        tarif_codes = pd.Series(np.nan, index=df.index, dtype=float)

    # Lookup ke known_tarif_map jika ada
    if known_tarif_map and tarif_codes.isna().any():
        tarif_codes = tarif_codes.fillna(df["NoPEL"].map(known_tarif_map))

    # Klasifikasi cerdas dari Nama / Alamat jika masih NaN
    still_nan = tarif_codes.isna()
    if still_nan.any():
        inferred = []
        names = df["Nama"].astype(str) if "Nama" in df.columns else pd.Series("", index=df.index)
        addrs = df["Alamat"].astype(str) if "Alamat" in df.columns else pd.Series("", index=df.index)
        for name, addr in zip(names, addrs):
            text = (name + " " + addr).upper()
            if any(k in text for k in ["KORAMIL", "POLSEK", "TNI", "KODIM"]):
                code = 1  # A1: ABRI/TNI
            elif any(k in text for k in ["DINAS", "DESA", "KECAMATAN", "KANTOR", "PUSKESMAS", "SD ", "SDN", "SMP", "SMA", "BALAI"]):
                code = 5  # IP: Instansi Pemerintah
            elif any(k in text for k in ["MASJID", "MUSHOLA", "GEREJA", "YAYASAN"]):
                code = 11  # S1: Sosial Umum
            elif any(k in text for k in ["TOKO", "WARUNG", "BENGKEL", "RUKO", "KAFE", "HOTEL", "PT ", "CV "]):
                code = 6  # N1: Niaga Kecil
            else:
                code = default_tarif_code  # 10: R3 Rumah Tangga C (standar default Purbalingga)
            inferred.append(code)
        tarif_codes = tarif_codes.fillna(pd.Series(inferred, index=df.index))

    df["tarif"] = tarif_codes

    # 2. Tambahkan kolom golongan_tarif di samping kolom tarif
    if df_tarif_ref is not None and not df_tarif_ref.empty:
        ref_idx = df_tarif_ref.drop_duplicates(subset=["tarif_code"]).copy()
        ref_idx["tarif_code"] = pd.to_numeric(ref_idx["tarif_code"], errors="coerce")
        ref_idx = ref_idx.set_index("tarif_code")

        golongan_map = ref_idx["tarif_symbol"] + " - " + ref_idx["tarif_name"]
        df["golongan_tarif"] = tarif_codes.map(golongan_map)

        # 3. Hitung tagihan tarif (estimated_bill_rp) menggunakan metode progresif Perkotaan
        df = enrich_with_tariff_info(df, df_tarif_ref, usage_col="usage_original")

    logger.info(f"File {area_name}: tarif & golongan assigned ({len(df):,} baris)")
    return df


def export_consolidated_15_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat consolidated dataset yang hanya berisi 15 kolom persis yang diminta:
    nopel, tglbaca, waktu, alamat, lalu, ini, tarif, cater, rayon, status, periode, keterangan, lati, longi, alti
    """
    col_mapping = {
        "NoPEL": "nopel",
        "tglBaca": "tglbaca",
        "Waktu": "waktu",
        "Alamat": "alamat",
        "Lalu": "lalu",
        "Ini": "ini",
        "tarif": "tarif",
        "Cater": "cater",
        "RAYON": "rayon",
        "Status": "status",
        "periode": "periode",
        "keterangan": "keterangan",
        "latitude": "lati",
        "longitude": "longi",
        "alti": "alti",
    }
    
    df_out = pd.DataFrame()
    for std_col, req_col in col_mapping.items():
        if std_col in df.columns:
            df_out[req_col] = df[std_col]
        elif req_col in df.columns:
            df_out[req_col] = df[req_col]
        else:
            df_out[req_col] = np.nan

    return df_out[config.CONSOLIDATED_COLUMNS]


def export_consolidated_with_tariff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat consolidated dataset lengkap dengan tarif terhitung dan golongan tarif:
    nopel, tglbaca, waktu, alamat, lalu, ini, tarif, golongan_tarif, estimated_bill_rp,
    cater, rayon, status, periode, keterangan, lati, longi, alti, source_area
    """
    col_mapping = {
        "NoPEL": "nopel",
        "tglBaca": "tglbaca",
        "Waktu": "waktu",
        "Alamat": "alamat",
        "Lalu": "lalu",
        "Ini": "ini",
        "tarif": "tarif",
        "golongan_tarif": "golongan_tarif",
        "estimated_bill_rp": "estimated_bill_rp",
        "Cater": "cater",
        "RAYON": "rayon",
        "Status": "status",
        "periode": "periode",
        "keterangan": "keterangan",
        "latitude": "lati",
        "longitude": "longi",
        "alti": "alti",
        "source_area": "source_area",
    }
    df_out = pd.DataFrame()
    for std_col, req_col in col_mapping.items():
        if std_col in df.columns:
            df_out[req_col] = df[std_col]
        elif req_col in df.columns:
            df_out[req_col] = df[req_col]
        else:
            df_out[req_col] = np.nan

    return df_out[config.CONSOLIDATED_WITH_TARIFF_COLUMNS]


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardisasi schema DataFrame ke format minimal.
    """
    df = clean_raw_columns(df)
    df = _rename_columns(df)
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    # --- NoPEL ---
    if "NoPEL" in df.columns:
        df["NoPEL"] = df["NoPEL"].astype(str).str.strip()
    else:
        logger.warning("Kolom NoPEL tidak ditemukan!")

    # --- Lalu, Ini → float ---
    for col in ["Lalu", "Ini"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning(f"Kolom {col} tidak ditemukan, diisi NaN.")
            df[col] = np.nan

    # --- usage_original ---
    df["usage_original"] = df["Ini"] - df["Lalu"]

    # --- periode ---
    if "periode" in df.columns:
        df["periode"] = df["periode"].astype(str).str.strip()
    else:
        logger.warning("Kolom 'periode' tidak ditemukan!")
        df["periode"] = pd.NA

    # --- period_dt: parse dari 'periode' → awal bulan ---
    df["period_dt"] = _parse_period(df["periode"])

    # --- tglBaca ---
    if "tglBaca" in df.columns:
        df["tglBaca"] = pd.to_datetime(df["tglBaca"], errors="coerce")

    # --- latitude / longitude → float ---
    for geo_col in ["latitude", "longitude"]:
        if geo_col in df.columns:
            df[geo_col] = pd.to_numeric(df[geo_col], errors="coerce")
        else:
            df[geo_col] = np.nan

    # --- tarif, golongan_tarif, estimated_bill_rp ---
    if "tarif" in df.columns:
        df["tarif"] = pd.to_numeric(df["tarif"], errors="coerce")
        df["tarif_raw"] = df["tarif"]
    elif "tarif_raw" in df.columns:
        df["tarif"] = pd.to_numeric(df["tarif_raw"], errors="coerce")
    else:
        df["tarif"] = np.nan
        df["tarif_raw"] = np.nan

    if "golongan_tarif" not in df.columns:
        df["golongan_tarif"] = np.nan

    if "estimated_bill_rp" in df.columns:
        df["estimated_bill_rp"] = pd.to_numeric(df["estimated_bill_rp"], errors="coerce")
    else:
        df["estimated_bill_rp"] = np.nan

    # Pastikan alias koordinat lati & longi terisi
    if "latitude" in df.columns and ("lati" not in df.columns or df["lati"].isna().all()):
        df["lati"] = df["latitude"]
    if "longitude" in df.columns and ("longi" not in df.columns or df["longi"].isna().all()):
        df["longi"] = df["longitude"]

    # --- Kolom schema yang harus ada ---
    for col in config.SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Drop HANYA artefak ID internal database mentah (_id, _id.1, kode, kode_unit)
    # PERTAHANKAN: NoPEL, Nama, Alamat, Lalu, Ini, Cater, RAYON, Status, periode, keterangan, lati, longi, alti, time, updatedDate, updatedBy
    drop_cols = [c for c in df.columns if c.lower() in ["_id", "_id.1", "kode", "kode_unit"]]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    logger.info(f"Schema standardized: {len(df):,} rows, {df.shape[1]} kolom")
    return df


# =============================================================================
# GAZETTEER & COORDINATE CORRECTION
# =============================================================================

PURBALINGGA_GAZETTEER = {
    # Perkotaan / Purbalingga Kota
    "purbalingga lor": (-7.3824, 109.3621),
    "purbalingga kidul": (-7.3941, 109.3642),
    "purbalingga kulon": (-7.3885, 109.3567),
    "purbalingga wetan": (-7.3862, 109.3712),
    "kandang gampang": (-7.3821, 109.3541),
    "kandangbgampang": (-7.3821, 109.3541),
    "bancar": (-7.3789, 109.3685),
    "penambongan": (-7.3956, 109.3524),
    "wirasana": (-7.3742, 109.3756),
    "kedungmenjangan": (-7.3982, 109.3785),
    "kedung menjangan": (-7.3982, 109.3785),
    "jatisaba": (-7.3695, 109.3812),
    "kembaran kulon": (-7.3802, 109.3491),
    "kembaran": (-7.3802, 109.3491),
    # Bojongsari & Bobotsari
    "bojongsari": (-7.3485, 109.3562),
    "gembong": (-7.3521, 109.3641),
    "kajongan": (-7.3412, 109.3512),
    "banjaran": (-7.3591, 109.3481),
    "bobotsari": (-7.3156, 109.3871),
    # Kalimanah & Padamara & Kutasari
    "kalimanah": (-7.3972, 109.3361),
    "kalimanah kulon": (-7.4012, 109.3289),
    "kalimanah wetan": (-7.3985, 109.3412),
    "grecol": (-7.4052, 109.3421),
    "selabaya": (-7.3991, 109.3465),
    "mewek": (-7.4082, 109.3385),
    "manduraga": (-7.4112, 109.3295),
    "karangpetir": (-7.4021, 109.3312),
    "babakan": (-7.3965, 109.3391),
    "padamara": (-7.3789, 109.3142),
    "kutasari": (-7.3421, 109.3305),
    # Bukateja
    "bukateja": (-7.4277, 109.4312),
    "kebutuh": (-7.4321, 109.4215),
    "majasari": (-7.4395, 109.4421),
    "penaruban": (-7.4189, 109.4285),
    "cipawon": (-7.4412, 109.4189),
    "karanggedang": (-7.4215, 109.4485),
    "kedungjati": (-7.4152, 109.4521),
    "kembangan": (-7.4485, 109.4356),
    "kutawis": (-7.4352, 109.4589),
    "tidu": (-7.4512, 109.4215),
    "wirasaba": (-7.4589, 109.4125),
    # Kemangkon
    "kemangkon": (-7.4725, 109.3789),
    "panican": (-7.4612, 109.3895),
    "bakulan": (-7.4812, 109.3685),
    "bokol": (-7.4785, 109.3952),
    "gambarsari": (-7.4685, 109.3621),
    "jetis": (-7.4895, 109.3812),
    "kalialang": (-7.4652, 109.3512),
    "karangkemiri": (-7.4821, 109.3589),
    "karangtengah": (-7.4752, 109.3712),
    "kedunglegok": (-7.4695, 109.3912),
    "majatengah": (-7.4852, 109.4012),
    "pegandekan": (-7.4912, 109.3715),
    "senon": (-7.4952, 109.3891),
    "sumilir": (-7.4795, 109.4125),
    "toyareka": (-7.4521, 109.3752),
    # Area Lainnya
    "kaligondang": (-7.3921, 109.4125),
    "kejobong": (-7.4251, 109.4892),
    "pengadegan": (-7.4123, 109.4621),
    "karangmoncol": (-7.2891, 109.4821),
    "karanganyar": (-7.3012, 109.4123),
    "rembang": (-7.3025, 109.5218),
    "mrebet": (-7.3245, 109.3412),
    "serang": (-7.2351, 109.2985),
    "karangreja": (-7.2412, 109.3125),
    "kutabawa": (-7.2485, 109.3052),
}

AREA_CENTROIDS = {
    "PERKOTAAN": (-7.3880, 109.3638),
    "BUKATEJA": (-7.4277, 109.4312),
    "KEMANGKON-1": (-7.4650, 109.3800),
    "KEMANGKON-2": (-7.4750, 109.3750),
    "KEMANGKON-3": (-7.4850, 109.3850),
}


def correct_zero_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perbaiki nilai koordinat (latitude & longitude) yang bernilai 0, NaN,
    atau di luar batas geografis Purbalingga (-8.2 s/d -6.8, 108.8 s/d 110.2).

    Strategi:
    1. Propagasi dari record pelanggan yang sama (NoPEL) yang memiliki koordinat valid.
    2. Lookup nama Desa/Kelurahan/Kecamatan dari teks 'Alamat' ke PURBALINGGA_GAZETTEER.
    3. Fallback ke Centroid Area/Rayon dengan jitter mikro deterministik (hash NoPEL)
       agar titik pada visualisasi peta menyebar alami dan tidak menumpuk 1 pixel.

    Returns
    -------
    pd.DataFrame dengan latitude & longitude yang 100% valid dan kolom 'coord_corrected_flag'.
    """
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    # Pastikan numerik
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    # Mask valid
    valid_mask = (
        lat.notna() & lon.notna()
        & (lat != 0) & (lon != 0)
        & (lat >= -8.2) & (lat <= -6.8)
        & (lon >= 108.8) & (lon <= 110.2)
    )

    df["coord_corrected_flag"] = (~valid_mask).astype(int)
    n_invalid = int((~valid_mask).sum())
    logger.info(f"Koordinat invalid/nol ditemukan: {n_invalid:,} dari {len(df):,} baris ({n_invalid/max(len(df),1)*100:.1f}%)")

    if n_invalid == 0:
        df["latitude"] = lat
        df["longitude"] = lon
        return df

    # 1. Propagasi per NoPEL (vectorized via map)
    valid_df = df[valid_mask]
    if not valid_df.empty:
        nopel_lat_map = valid_df.groupby("NoPEL")["latitude"].first().to_dict()
        nopel_lon_map = valid_df.groupby("NoPEL")["longitude"].first().to_dict()

        lat = lat.where(valid_mask, df["NoPEL"].map(nopel_lat_map))
        lon = lon.where(valid_mask, df["NoPEL"].map(nopel_lon_map))

    # Cek ulang yang masih invalid
    still_invalid = (
        lat.isna() | lon.isna()
        | (lat == 0) | (lon == 0)
        | (lat < -8.2) | (lat > -6.8)
        | (lon < 108.8) | (lon > 110.2)
    )

    if still_invalid.any():
        # 2. Geocode deduplicated unique NoPEL/Alamat/Area (hanya subset pelanggan unik!)
        sub_cols = [c for c in ["NoPEL", "Alamat", "source_area"] if c in df.columns]
        sub = df.loc[still_invalid, sub_cols].drop_duplicates(subset=["NoPEL"])

        geo_lat = {}
        geo_lon = {}
        for _, row in sub.iterrows():
            nopel = str(row["NoPEL"])
            addr_raw = row["Alamat"] if "Alamat" in row else ""
            if isinstance(addr_raw, (pd.Series, np.ndarray, list)):
                addr_raw = addr_raw[0] if len(addr_raw) > 0 else ""
            addr = str(addr_raw).lower() if pd.notna(addr_raw) else ""
            area = str(row.get("source_area", "PERKOTAAN"))

            found_coord = None
            for key, coord in PURBALINGGA_GAZETTEER.items():
                if key in addr:
                    found_coord = coord
                    break

            if found_coord is None:
                found_coord = AREA_CENTROIDS.get(area, (-7.3880, 109.3638))

            h = abs(hash(nopel)) % 10000
            lat_jitter = ((h % 100) - 50) * 0.00004
            lon_jitter = (((h // 100) % 100) - 50) * 0.00004

            geo_lat[nopel] = round(found_coord[0] + lat_jitter, 6)
            geo_lon[nopel] = round(found_coord[1] + lon_jitter, 6)

        lat = lat.where(~still_invalid, df["NoPEL"].map(geo_lat))
        lon = lon.where(~still_invalid, df["NoPEL"].map(geo_lon))

    # Fallback terakhir jika masih ada NaN
    lat = lat.fillna(-7.3880)
    lon = lon.fillna(109.3638)

    df["latitude"] = lat
    df["longitude"] = lon

    logger.info(f"✅ Seluruh {n_invalid:,} koordinat invalid telah diperbaiki via address geocoding & centroid mapping.")
    return df



def _parse_period(series: pd.Series) -> pd.Series:
    """
    Parse kolom 'periode' ke datetime awal bulan.

    Format yang didukung:
    - 'YYYY-MM' atau 'YYYY/MM'
    - 'YYYYMM' (6 digit)
    - datetime langsung
    """
    def _parse_single(val):
        if pd.isna(val) or str(val).strip() in ["", "nan", "None"]:
            return pd.NaT

        val = str(val).strip()

        # Format YYYYMM (6 digit tanpa separator)
        if len(val) == 6 and val.isdigit():
            try:
                return pd.Timestamp(f"{val[:4]}-{val[4:]}-01")
            except Exception:
                return pd.NaT

        # Format YYYY-MM atau YYYY/MM
        for fmt in ["%Y-%m", "%Y/%m", "%m-%Y", "%m/%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"]:
            try:
                dt = pd.to_datetime(val, format=fmt)
                return dt.replace(day=1)
            except Exception:
                continue

        # Fallback: pandas inference
        try:
            dt = pd.to_datetime(val, infer_datetime_format=True)
            return dt.replace(day=1)
        except Exception:
            return pd.NaT

    result = series.apply(_parse_single)
    n_failed = result.isna().sum()
    if n_failed > 0:
        logger.warning(f"period_dt parse gagal untuk {n_failed:,} baris.")
    return result


# =============================================================================
# QUALITY FLAGS
# =============================================================================

def create_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat 7 quality flags per record. Nilai asli tidak diubah.

    Flags:
    - zero_original_flag         : usage_original == 0
    - negative_usage_flag        : usage_original < 0
    - missing_usage_flag         : Lalu atau Ini NaN
    - invalid_meter_flag         : Ini < Lalu (meter rollback/rusak)
    - duplicate_period_flag      : NoPEL × periode duplikat
    - gap_month_flag             : ada bulan yang terlewat sebelum baris ini
    - imputation_required_flag   : gabungan dari missing/negative/zero (sesuai kebutuhan)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan schema standar, sudah ada usage_original dan period_dt.

    Returns
    -------
    pd.DataFrame dengan kolom flag tambahan.
    """
    df = df.copy()

    # 1. Zero usage
    df["zero_original_flag"] = (df["usage_original"] == 0).astype(int)

    # 2. Negative usage
    df["negative_usage_flag"] = (df["usage_original"] < 0).astype(int)

    # 3. Missing usage (Lalu atau Ini NaN)
    df["missing_usage_flag"] = (df["Lalu"].isna() | df["Ini"].isna()).astype(int)

    # 4. Invalid meter (Ini < Lalu, tapi bukan karena missing)
    df["invalid_meter_flag"] = (
        (~df["missing_usage_flag"].astype(bool)) &
        (df["Ini"] < df["Lalu"])
    ).astype(int)

    # 5. Duplicate period per NoPEL
    dup_mask = df.duplicated(subset=["NoPEL", "period_dt"], keep=False)
    df["duplicate_period_flag"] = dup_mask.astype(int)

    # 6. Gap month — sort dulu per NoPEL × period_dt
    df = df.sort_values(["NoPEL", "period_dt"]).copy()
    df["gap_month_flag"] = _compute_gap_flag(df)

    # 7. Imputation required = ada masalah nilai usage
    df["imputation_required_flag"] = (
        (df["missing_usage_flag"] == 1) |
        (df["negative_usage_flag"] == 1) |
        (df["invalid_meter_flag"] == 1)
    ).astype(int)

    n_flags = {
        "zero"       : df["zero_original_flag"].sum(),
        "negative"   : df["negative_usage_flag"].sum(),
        "missing"    : df["missing_usage_flag"].sum(),
        "invalid"    : df["invalid_meter_flag"].sum(),
        "duplicate"  : df["duplicate_period_flag"].sum(),
        "gap"        : df["gap_month_flag"].sum(),
        "need_impute": df["imputation_required_flag"].sum(),
    }
    logger.info(f"Quality flags created: {n_flags}")

    return df


def _compute_gap_flag(df: pd.DataFrame) -> pd.Series:
    """
    Hitung gap_month_flag: 1 jika ada bulan yang terlewat antara
    baris sebelumnya dan baris ini untuk NoPEL yang sama.
    """
    gap_flag = pd.Series(0, index=df.index)

    for nopel, grp in df.groupby("NoPEL", sort=False):
        grp = grp.sort_values("period_dt")
        period_vals = grp["period_dt"]
        prev_vals   = period_vals.shift(1)

        expected_next = prev_vals + pd.DateOffset(months=1)
        has_gap = (
            (~prev_vals.isna()) &
            (~period_vals.isna()) &
            (period_vals > expected_next.dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=15))
        )
        gap_flag.loc[has_gap[has_gap].index] = 1

    return gap_flag


# =============================================================================
# MONTHLY QUALITY REPORT
# =============================================================================

def get_monthly_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buat laporan kualitas data per periode (bulan).

    Returns
    -------
    pd.DataFrame dengan kolom:
        periode, rows, customers, missing, zero, negative,
        invalid_meter, duplicate, gap, imputation_required
    """
    required_flags = [
        "missing_usage_flag", "zero_original_flag", "negative_usage_flag",
        "invalid_meter_flag", "duplicate_period_flag", "gap_month_flag",
        "imputation_required_flag",
    ]
    for f in required_flags:
        if f not in df.columns:
            raise ValueError(f"Flag '{f}' belum ada. Jalankan create_quality_flags() dulu.")

    report = (
        df.groupby("periode")
        .agg(
            rows      = ("NoPEL", "count"),
            customers = ("NoPEL", "nunique"),
            missing   = ("missing_usage_flag", "sum"),
            zero      = ("zero_original_flag", "sum"),
            negative  = ("negative_usage_flag", "sum"),
            invalid   = ("invalid_meter_flag", "sum"),
            duplicate = ("duplicate_period_flag", "sum"),
            gap       = ("gap_month_flag", "sum"),
            need_impute = ("imputation_required_flag", "sum"),
        )
        .reset_index()
        .sort_values("periode")
    )

    logger.info(f"Monthly quality report: {len(report)} periode")
    return report


# =============================================================================
# TIME-SPLIT BOUNDARY
# =============================================================================

def define_time_split(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    test_ratio:  float = config.TEST_RATIO,
) -> Dict[str, object]:
    """
    Tetapkan boundary Train/Validation/Test secara kronologis
    berdasarkan unique months.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame dengan kolom 'period_dt'.
    train_ratio, val_ratio, test_ratio : float
        Rasio split. Harus berjumlah 1.0.

    Returns
    -------
    dict dengan keys:
        - 'unique_months'      : sorted list of period_dt
        - 'n_total_months'     : jumlah bulan unik
        - 'train_end'          : batas akhir training (inclusive)
        - 'val_end'            : batas akhir validation (inclusive)
        - 'test_end'           : batas akhir test (inclusive)
        - 'train_months'       : list bulan train
        - 'val_months'         : list bulan validation
        - 'test_months'        : list bulan test
        - 'n_train', 'n_val', 'n_test' : jumlah bulan per split
        - 'split_summary'      : DataFrame ringkasan
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Rasio split harus berjumlah 1.0"

    unique_months = sorted(df["period_dt"].dropna().unique())
    n = len(unique_months)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val  # sisa untuk test

    train_months = unique_months[:n_train]
    val_months   = unique_months[n_train:n_train + n_val]
    test_months  = unique_months[n_train + n_val:]

    train_end = train_months[-1] if train_months else None
    val_end   = val_months[-1]   if val_months   else None
    test_end  = test_months[-1]  if test_months  else None

    split_summary = pd.DataFrame({
        "split"    : ["TRAIN", "VALIDATION", "TEST"],
        "n_months" : [len(train_months), len(val_months), len(test_months)],
        "start"    : [
            str(train_months[0])[:10] if train_months else None,
            str(val_months[0])[:10]   if val_months   else None,
            str(test_months[0])[:10]  if test_months  else None,
        ],
        "end"      : [
            str(train_months[-1])[:10] if train_months else None,
            str(val_months[-1])[:10]   if val_months   else None,
            str(test_months[-1])[:10]  if test_months  else None,
        ],
        "ratio_pct": [
            round(len(train_months)/n*100, 1),
            round(len(val_months)/n*100, 1),
            round(len(test_months)/n*100, 1),
        ],
    })

    logger.info(f"\nTime Split Summary:\n{split_summary.to_string(index=False)}")

    return {
        "unique_months"   : unique_months,
        "n_total_months"  : n,
        "train_end"       : train_end,
        "val_end"         : val_end,
        "test_end"        : test_end,
        "train_months"    : train_months,
        "val_months"      : val_months,
        "test_months"     : test_months,
        "n_train"         : len(train_months),
        "n_val"           : len(val_months),
        "n_test"          : len(test_months),
        "split_summary"   : split_summary,
    }


def add_split_label(df: pd.DataFrame, split_info: Dict) -> pd.DataFrame:
    """
    Tambahkan kolom 'split' ('TRAIN'/'VALIDATION'/'TEST') ke DataFrame
    berdasarkan period_dt.
    """
    df = df.copy()

    conditions = [
        df["period_dt"].isin(split_info["train_months"]),
        df["period_dt"].isin(split_info["val_months"]),
        df["period_dt"].isin(split_info["test_months"]),
    ]
    choices = ["TRAIN", "VALIDATION", "TEST"]
    df["split"] = np.select(conditions, choices, default="UNKNOWN")

    dist = df["split"].value_counts().to_dict()
    logger.info(f"Split distribution: {dist}")
    return df


# =============================================================================
# DUPLICATE HANDLING
# =============================================================================

def remove_duplicates(
    df: pd.DataFrame,
    strategy: str = "keep_first",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle duplikasi NoPEL × period_dt.

    Parameters
    ----------
    strategy : str
        'keep_first'  — pertahankan baris pertama
        'keep_last'   — pertahankan baris terakhir
        'flag_only'   — tidak hapus, hanya set duplicate_period_flag

    Returns
    -------
    (df_clean, df_duplicates) — DataFrame bersih dan DataFrame duplikat.
    """
    dup_mask = df.duplicated(subset=["NoPEL", "period_dt"], keep=False)
    df_dup   = df[dup_mask].copy()
    n_dup    = dup_mask.sum()

    if n_dup == 0:
        logger.info("Tidak ada duplikasi NoPEL × period_dt.")
        return df, df_dup

    logger.warning(f"Ditemukan {n_dup:,} baris duplikat (NoPEL × period_dt).")

    if strategy == "keep_first":
        df_clean = df.drop_duplicates(subset=["NoPEL", "period_dt"], keep="first")
    elif strategy == "keep_last":
        df_clean = df.drop_duplicates(subset=["NoPEL", "period_dt"], keep="last")
    elif strategy == "flag_only":
        df_clean = df.copy()
    else:
        raise ValueError(f"strategy tidak dikenal: {strategy}. Pilih: keep_first, keep_last, flag_only")

    logger.info(f"Setelah remove_duplicates ({strategy}): {len(df_clean):,} rows")
    return df_clean, df_dup


# =============================================================================
# TARIFF & RATE CALCULATION HELPERS
# =============================================================================

def calculate_water_bill(
    usage: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    abondemen: float,
) -> float:
    """
    Hitung tagihan air PDAM (Rp) menggunakan rumus blok tarif progresif:
      - Blok 1 (0-10 m³): usage × b1
      - Blok 2 (11-20 m³): 10×b1 + (usage-10)×b2
      - Blok 3 (21-30 m³): 10×b1 + 10×b2 + (usage-20)×b3
      - Blok 4 (>30 m³): 10×b1 + 10×b2 + 10×b3 + (usage-30)×b4
      + Abondemen (jasameter + danameter)
    """
    if pd.isna(usage) or usage < 0:
        return np.nan

    if usage <= 10:
        water_cost = usage * b1
    elif usage <= 20:
        water_cost = (10 * b1) + ((usage - 10) * b2)
    elif usage <= 30:
        water_cost = (10 * b1) + (10 * b2) + ((usage - 20) * b3)
    else:
        water_cost = (10 * b1) + (10 * b2) + (10 * b3) + ((usage - 30) * b4)

    return water_cost + abondemen


def enrich_with_tariff_info(
    df: pd.DataFrame,
    df_tarif_ref: pd.DataFrame,
    usage_col: str = "usage_original",
) -> pd.DataFrame:
    """
    Pengayaan DataFrame pelanggan dengan metadata tarif dan perhitungan tagihan (Rp).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pelanggan dengan kolom 'tarif_raw' dan usage_col.
    df_tarif_ref : pd.DataFrame
        Dataset referensi tarif dari load_tarif_reference().
    usage_col : str
        Kolom pemakaian air yang digunakan untuk menghitung tagihan.

    Returns
    -------
    pd.DataFrame dengan kolom tambahan:
        tarif_code_clean, tarif_symbol, tarif_name, estimated_bill_rp
    """
    df = df.copy()

    if df_tarif_ref is None or df_tarif_ref.empty or ("tarif_raw" not in df.columns and "tarif" not in df.columns):
        df["estimated_bill_rp"] = np.nan
        df["tarif_symbol"] = np.nan
        df["tarif_name"] = np.nan
        return df

    # Konversi tarif atau tarif_raw ke numeric
    if "tarif" in df.columns:
        tarif_codes = pd.to_numeric(df["tarif"], errors="coerce")
    else:
        tarif_codes = pd.to_numeric(df["tarif_raw"], errors="coerce")
    df["tarif_code_clean"] = tarif_codes
    df["tarif"] = tarif_codes

    # Set index pada df_tarif_ref untuk lookup berbasis index/mapping
    ref = df_tarif_ref.drop_duplicates(subset=["tarif_code"]).copy()
    ref["tarif_code"] = pd.to_numeric(ref["tarif_code"], errors="coerce")
    ref = ref.set_index("tarif_code")

    # Map metadata simbol dan nama
    df["tarif_symbol"] = tarif_codes.map(ref["tarif_symbol"])
    df["tarif_name"]   = tarif_codes.map(ref["tarif_name"])

    # Ambil rate blocks dan abondemen untuk baris df via map
    b1 = tarif_codes.map(ref["rate_block_0_10"])
    b2 = tarif_codes.map(ref["rate_block_10_20"])
    b3 = tarif_codes.map(ref["rate_block_20_30"])
    b4 = tarif_codes.map(ref["rate_block_above_30"])
    abon = tarif_codes.map(ref["abondemen"])

    # Vectorized calculation
    usages = pd.to_numeric(df[usage_col], errors="coerce")
    valid = tarif_codes.notna() & usages.notna() & b1.notna()

    bills = pd.Series(np.nan, index=df.index, dtype=float)

    if valid.any():
        u_val = usages[valid].values
        b1_val = b1[valid].values
        b2_val = b2[valid].values
        b3_val = b3[valid].values
        b4_val = b4[valid].values
        abon_val = abon[valid].values

        u_clamped = np.maximum(u_val, 0.0)
        c1 = np.minimum(u_clamped, 10.0)
        c2 = np.minimum(np.maximum(u_clamped - 10.0, 0.0), 10.0)
        c3 = np.minimum(np.maximum(u_clamped - 20.0, 0.0), 10.0)
        c4 = np.maximum(u_clamped - 30.0, 0.0)

        calc_bills = (c1 * b1_val) + (c2 * b2_val) + (c3 * b3_val) + (c4 * b4_val) + abon_val
        bills.loc[valid] = calc_bills

    df["estimated_bill_rp"] = bills
    n_calculated = valid.sum()
    logger.info(f"Tariff info enriched: {n_calculated:,} rows tagihan terhitung")
    return df


# =============================================================================
# WEB DASHBOARD INTEGRATION: EXCEL VALIDATION & BATCH PREPROCESSING
# =============================================================================

def validate_excel_input(
    file_or_df: object,
    sheet_name: object = 0,
) -> Tuple[bool, pd.DataFrame, Dict[str, object]]:
    """
    Validasi ketat file Excel / DataFrame yang diunggah dari Dashboard PDAM.

    Aturan Validasi:
    1. Kolom wajib minimum: NoPEL, Lalu, Ini (case-insensitive & alias-supported).
    2. Kolom periode & tarif sangat dianjurkan (dapat di-imputasi jika kosong).
    3. NoPEL distandardisasi menjadi string 8 digit (zfill 8).
    4. Indeks meter 'Ini' dan 'Lalu' harus berupa angka non-negatif.
    5. Indikasi rollover/pergantian meter (Ini < Lalu) ditandai pada kolom flag.

    Parameters
    ----------
    file_or_df : str, Path, BytesIO, or pd.DataFrame
        Input data dari upload web dashboard atau path lokal.
    sheet_name : str or int
        Nama sheet Excel jika input berupa file.

    Returns
    -------
    (is_valid, df_cleaned, validation_report)
        is_valid : bool (True jika data memenuhi kriteria minimum untuk diproses)
        df_cleaned : pd.DataFrame (Data terstandarisasi yang siap diproses)
        validation_report : dict (Statistik baris, peringatan, dan daftar error)
    """
    errors = []
    warnings = []

    # 1. Baca data jika berupa file/buffer
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        try:
            if hasattr(file_or_df, "read"):
                # File-like object (e.g. Streamlit UploadedFile)
                df = pd.read_excel(file_or_df, sheet_name=sheet_name)
            else:
                p = Path(file_or_df)
                if not p.exists():
                    return False, pd.DataFrame(), {
                        "is_valid": False,
                        "errors": [f"File tidak ditemukan: {p}"],
                        "warnings": [],
                        "total_rows": 0,
                    }
                if p.suffix.lower() in [".xlsx", ".xls"]:
                    df = pd.read_excel(p, sheet_name=sheet_name)
                elif p.suffix.lower() == ".csv":
                    df = pd.read_csv(p)
                elif p.suffix.lower() == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    return False, pd.DataFrame(), {
                        "is_valid": False,
                        "errors": [f"Format file tidak didukung: {p.suffix}"],
                        "warnings": [],
                        "total_rows": 0,
                    }
        except Exception as e:
            return False, pd.DataFrame(), {
                "is_valid": False,
                "errors": [f"Gagal membaca file: {str(e)}"],
                "warnings": [],
                "total_rows": 0,
            }

    if df.empty:
        return False, df, {
            "is_valid": False,
            "errors": ["Dataset kosong (0 baris)."],
            "warnings": [],
            "total_rows": 0,
        }

    total_rows_raw = len(df)

    # 2. Standardisasi nama kolom via alias map
    df = clean_raw_columns(df)
    df = _rename_columns(df)

    # 3. Verifikasi kolom wajib
    required_cols = ["NoPEL", "Ini", "Lalu"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        errors.append(f"Kolom wajib tidak ditemukan: {missing_required}. Kolom tersedia: {list(df.columns)}")
        return False, df, {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "total_rows": total_rows_raw,
            "valid_rows": 0,
        }

    # 4. Standardisasi NoPEL
    df["NoPEL"] = df["NoPEL"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    invalid_nopel_mask = (df["NoPEL"] == "") | (df["NoPEL"] == "nan") | (df["NoPEL"] == "None")
    n_invalid_nopel = invalid_nopel_mask.sum()
    if n_invalid_nopel > 0:
        warnings.append(f"Ditemukan {n_invalid_nopel:,} baris dengan NoPEL kosong/tidak valid (akan diabaikan).")
        df = df[~invalid_nopel_mask].copy()

    df["NoPEL"] = df["NoPEL"].str.zfill(8)

    # 5. Konversi dan validasi numerik Lalu dan Ini
    df["Lalu"] = pd.to_numeric(df["Lalu"], errors="coerce")
    df["Ini"]  = pd.to_numeric(df["Ini"], errors="coerce")

    nan_meter_mask = df["Lalu"].isna() | df["Ini"].isna()
    n_nan_meter = nan_meter_mask.sum()
    if n_nan_meter > 0:
        warnings.append(f"Ditemukan {n_nan_meter:,} baris dengan angka meter Ini/Lalu NaN/non-numerik.")

    # Indeks negatif
    neg_meter_mask = (df["Lalu"] < 0) | (df["Ini"] < 0)
    n_neg = neg_meter_mask.sum()
    if n_neg > 0:
        warnings.append(f"Ditemukan {n_neg:,} baris dengan angka meteran negatif.")

    # Hitung pemakaian air (usage_original = Ini - Lalu)
    df["usage_original"] = df["Ini"] - df["Lalu"]

    # Deteksi rollover / meter dibalik (Ini < Lalu)
    rollover_mask = df["usage_original"] < 0
    n_rollover = rollover_mask.sum()
    if n_rollover > 0:
        warnings.append(f"Ditemukan {n_rollover:,} baris pemakaian negatif (Ini < Lalu, kemungkinan rollover/ganti meter).")

    # 6. Validasi dan standardisasi Periode
    if "periode" in df.columns:
        # Standardisasi ke string YYYYMM
        periode_str = df["periode"].astype(str).str.strip().str.replace(r"[-\s]", "", regex=True)
        # Handle decimal like 202401.0
        periode_str = periode_str.str.replace(r"\.0$", "", regex=True)
        # Ambil 6 karakter pertama jika format YYYYMMDD
        periode_clean = periode_str.str.slice(0, 6)
        df["periode"] = periode_clean
        df["period_dt"] = pd.to_datetime(periode_clean, format="%Y%m", errors="coerce")
    else:
        now_period = pd.Timestamp.now().strftime("%Y%m")
        warnings.append(f"Kolom 'periode' tidak ditemukan. Menggunakan periode default berjalan: {now_period}")
        df["periode"] = now_period
        df["period_dt"] = pd.to_datetime(now_period, format="%Y%m")

    # 7. Validasi tarif
    if "tarif" in df.columns:
        df["tarif"] = pd.to_numeric(df["tarif"], errors="coerce").fillna(1)
    else:
        df["tarif"] = 1
        warnings.append("Kolom 'tarif' tidak ditemukan. Di-imputasi dengan nilai default tarif = 1.")

    # Validasi koordinat jika ada
    if "lati" in df.columns and "longi" in df.columns:
        df["lati"] = pd.to_numeric(df["lati"], errors="coerce")
        df["longi"] = pd.to_numeric(df["longi"], errors="coerce")
    elif "latitude" in df.columns and "longitude" in df.columns:
        df["lati"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longi"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Kolom teks pelengkap
    for opt_col in ["Nama", "Alamat", "RAYON", "Cater", "Status", "keterangan"]:
        if opt_col not in df.columns:
            # Check lowercase
            match = [c for c in df.columns if c.lower() == opt_col.lower()]
            if match:
                df[opt_col] = df[match[0]]
            else:
                df[opt_col] = "-"

    is_valid = len(errors) == 0 and len(df) > 0

    report = {
        "is_valid"      : is_valid,
        "total_rows_raw": total_rows_raw,
        "valid_rows"    : len(df),
        "errors"        : errors,
        "warnings"      : warnings,
        "summary": {
            "n_nan_meter": int(n_nan_meter),
            "n_negative" : int(n_neg),
            "n_rollover" : int(n_rollover),
            "unique_nopel": int(df["NoPEL"].nunique()),
            "unique_periods": [str(p) for p in df["periode"].unique().tolist() if pd.notna(p)],
        }
    }

    return is_valid, df, report


def preprocess_new_batch(
    df_raw: pd.DataFrame,
    check_db_duplicates: bool = True,
    db_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Pipeline preprocessing lengkap untuk batch data baru:
    1. Validasi input Excel/DataFrame.
    2. Pengecekan duplikasi terhadap database lokal (menghindari duplikasi NoPEL + Periode).
    3. Pengayaan tarif dan estimasi tagihan jika referensi tersedia.
    4. Penyiapan kolom siap clustering & prediksi.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Data mentah hasil upload.
    check_db_duplicates : bool
        Jika True, periksa duplikasi ke SQLite database.
    db_path : Optional[Path]
        Lokasi database SQLite.

    Returns
    -------
    (df_ready, summary_dict)
    """
    is_valid, df, val_report = validate_excel_input(df_raw)
    if not is_valid:
        raise ValueError(f"Validasi data gagal: {val_report['errors']}")

    # Cek duplikasi terhadap DB jika diminta
    n_existing_in_db = 0
    if check_db_duplicates:
        try:
            from . import database
            existing_keys = database.check_existing_keys(df, db_path=db_path)
            if existing_keys:
                n_existing_in_db = len(existing_keys)
                df["is_already_in_db"] = df.apply(
                    lambda r: (str(r["NoPEL"]).zfill(8), str(r["periode"])) in existing_keys,
                    axis=1,
                )
            else:
                df["is_already_in_db"] = False
        except Exception as e:
            logger.warning(f"Gagal memeriksa duplikasi DB: {e}")
            df["is_already_in_db"] = False

    # Pengayaan tarif jika file TARIF.xlsx ada
    try:
        from . import data_loader
        df_tarif = data_loader.load_tarif_reference()
        if df_tarif is not None and not df_tarif.empty:
            df = enrich_with_tariff_info(df, df_tarif, usage_col="usage_original")
    except Exception as e:
        logger.info(f"Lewati pengayaan tarif otomatis: {e}")

    summary = {
        "validation"       : val_report,
        "n_processed"      : len(df),
        "n_already_in_db"  : n_existing_in_db,
    }

    return df, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("preprocessing.py loaded OK")


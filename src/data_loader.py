"""
src/data_loader.py — Load data Excel PDAM dari berbagai sumber.

Rules:
- Tidak ada sampling.
- TARIF hanya reference, tidak dicampur ke data pelanggan.
- Sheet AUDIT pada PERKOTAAN diabaikan.
- NoPEL selalu string.
- Validasi row count sebelum dan sesudah concat.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _read_excel_sheet(
    filepath: Path,
    sheet_name: str,
    source_area: str,
    source_file: str,
) -> Optional[pd.DataFrame]:
    """
    Baca satu sheet Excel dan tambahkan kolom provenance.
    Return None jika file/sheet tidak ditemukan.
    """
    if not filepath.exists():
        logger.warning(f"File tidak ditemukan: {filepath}")
        return None

    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, dtype=str)
        df["source_area"]  = source_area
        df["source_file"]  = str(filepath.name)
        df["source_sheet"] = sheet_name
        logger.info(f"  Loaded {source_area}/{sheet_name}: {len(df):,} rows")
        return df
    except Exception as e:
        logger.error(f"Gagal membaca {filepath} sheet '{sheet_name}': {e}")
        return None


# =============================================================================
# PUBLIC LOADERS
# =============================================================================

def load_perkotaan() -> pd.DataFrame:
    """
    Load data PERKOTAAN dari dua sheet:
      - PERKOTAAN_2020_2024
      - LANJUTAN_2024_2025
    Sheet AUDIT diabaikan.

    Returns
    -------
    pd.DataFrame gabungan dua sheet PERKOTAAN.
    """
    filepath = config.RAW_FILES["PERKOTAAN"]
    frames = []

    for sheet in config.PERKOTAAN_SHEETS:
        df = _read_excel_sheet(filepath, sheet, source_area="PERKOTAAN", source_file=filepath.name)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"Tidak ada sheet PERKOTAAN yang berhasil dimuat dari {filepath}. "
            "Periksa path dan nama sheet di config.py."
        )

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"PERKOTAAN total: {len(result):,} rows dari {len(frames)} sheet(s)")
    return result


def load_wilayah(area: str) -> pd.DataFrame:
    """
    Load data wilayah non-PERKOTAAN (BUKATEJA, KEMANGKON-1/2/3)
    dari sheet 'Table1'.

    Parameters
    ----------
    area : str
        Nama wilayah. Harus ada di config.RAW_FILES.

    Returns
    -------
    pd.DataFrame
    """
    if area not in config.RAW_FILES:
        raise ValueError(f"Area '{area}' tidak ada di config.RAW_FILES. "
                         f"Pilihan: {list(config.RAW_FILES.keys())}")

    filepath = config.RAW_FILES[area]
    df = _read_excel_sheet(filepath, config.WILAYAH_SHEET, source_area=area, source_file=filepath.name)

    if df is None:
        raise FileNotFoundError(
            f"Gagal memuat {area} dari {filepath}. "
            f"Pastikan sheet '{config.WILAYAH_SHEET}' ada."
        )

    return df


def load_tarif() -> Optional[pd.DataFrame]:
    """
    Load file TARIF sebagai data referensi saja.
    TIDAK dicampur ke data pelanggan.

    Returns
    -------
    pd.DataFrame atau None jika file tidak ditemukan.
    """
    filepath = config.TARIF_FILE
    if not filepath.exists():
        logger.warning(f"TARIF file tidak ditemukan: {filepath}. Lanjutkan tanpa tarif.")
        return None

    try:
        df = pd.read_excel(filepath, dtype=str)
        logger.info(f"TARIF loaded: {len(df):,} rows, kolom: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Gagal membaca TARIF: {e}")
        return None


def load_tarif_reference() -> pd.DataFrame:
    """
    Parse dan konsolidasikan TARIF.xlsx menjadi satu dataset referensi tarif terpadu.

    Dataset ini berisi struktur tarif progresif per golongan:
      - tarif_code: kode angka (1..17)
      - tarif_symbol: kode singkatan (A1, R1, N1, dll.)
      - tarif_name: nama golongan pelanggan
      - rate_block_0_10: tarif per m³ (0-10 m³)
      - rate_block_10_20: tarif per m³ (10-20 m³)
      - rate_block_20_30: tarif per m³ (20-30 m³)
      - rate_block_above_30: tarif per m³ (>30 m³)
      - jasameter: pemeliharaan meter (Rp)
      - danameter: dana meter (Rp)
      - abondemen: biaya tetap total (jasameter + danameter)

    Returns
    -------
    pd.DataFrame
    """
    import openpyxl

    filepath = config.TARIF_FILE
    ref_csv  = config.DATA_REFERENCE / "tarif_reference.csv"

    # Jika file CSV referensi sudah ada dan TARIF.xlsx tidak tersedia, load CSV
    if ref_csv.exists() and not filepath.exists():
        df_ref = pd.read_csv(ref_csv)
        logger.info(f"Loaded tarif_reference.csv: {len(df_ref)} rows")
        return df_ref

    if not filepath.exists():
        logger.warning(f"TARIF.xlsx tidak ditemukan di {filepath}")
        return pd.DataFrame()

    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        ws = wb["Sheet1"]
        rows = list(ws.iter_rows(values_only=True))

        records = []
        for r in rows[4:]:
            if r[0] is not None and isinstance(r[0], int):
                records.append({
                    "tarif_code"         : int(r[0]),
                    "tarif_symbol"       : str(r[1]).strip() if r[1] else "",
                    "tarif_name"         : str(r[2]).strip() if r[2] else "",
                    "rate_block_0_10"    : float(r[3]) if r[3] else 0.0,
                    "rate_block_10_20"   : float(r[4]) if r[4] else 0.0,
                    "rate_block_20_30"   : float(r[5]) if r[5] else 0.0,
                    "rate_block_above_30": float(r[6]) if r[6] else 0.0,
                    "jasameter"          : float(r[7]) if r[7] else 0.0,
                    "danameter"          : float(r[8]) if r[8] else 0.0,
                    "abondemen"          : float(r[9]) if r[9] else 0.0,
                })

        df_ref = pd.DataFrame(records)

        # Simpan ke data/reference/tarif_reference.csv
        config.DATA_REFERENCE.mkdir(parents=True, exist_ok=True)
        df_ref.to_csv(ref_csv, index=False)
        logger.info(f"✅ Consolidated tarif reference saved to {ref_csv}: {len(df_ref)} rows")
        return df_ref
    except Exception as e:
        logger.error(f"Gagal memproses TARIF reference: {e}")
        return pd.DataFrame()



def load_all_areas(
    areas: Optional[List[str]] = None,
    validate_row_count: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load dan concat semua wilayah menjadi satu DataFrame master.

    Parameters
    ----------
    areas : list of str, optional
        Daftar area yang dimuat. Default = config.SOURCE_AREAS.
    validate_row_count : bool
        Jika True, validasi total row setelah concat == sum per area.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        (master_df, row_counts_per_area)
    """
    if areas is None:
        areas = config.SOURCE_AREAS

    frames = []
    row_counts: Dict[str, int] = {}

    for area in areas:
        logger.info(f"Loading area: {area} ...")
        try:
            if area == "PERKOTAAN":
                df = load_perkotaan()
            else:
                df = load_wilayah(area)
            row_counts[area] = len(df)
            frames.append(df)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Area {area} gagal dimuat: {e}")
            row_counts[area] = 0

    if not frames:
        raise RuntimeError("Tidak ada data yang berhasil dimuat dari semua area.")

    master = pd.concat(frames, ignore_index=True)

    # -------------------------------------------------------------------------
    # VALIDASI ROW COUNT
    # -------------------------------------------------------------------------
    if validate_row_count:
        expected = sum(row_counts.values())
        actual   = len(master)
        if actual != expected:
            raise AssertionError(
                f"ROW COUNT MISMATCH! "
                f"Expected {expected:,} rows setelah concat, "
                f"tapi dapat {actual:,} rows. "
                "Periksa duplikasi yang tidak disengaja saat concat."
            )
        logger.info(f"✅ Row count validated: {actual:,} rows (sum = {expected:,})")

    logger.info(f"Master DataFrame: {len(master):,} rows, {master.shape[1]} kolom")
    logger.info(f"Row counts per area: {row_counts}")

    return master, row_counts


# =============================================================================
# INVENTORY HELPER
# =============================================================================

def get_file_inventory() -> pd.DataFrame:
    """
    Buat inventaris file Excel yang tersedia di data/raw/ dan data/reference/.

    Returns
    -------
    pd.DataFrame dengan kolom: area, filepath, exists, size_kb, sheets
    """
    records = []

    for area, filepath in config.RAW_FILES.items():
        record = {
            "role"     : "data_utama",
            "area"     : area,
            "filepath" : str(filepath),
            "exists"   : filepath.exists(),
            "size_kb"  : round(filepath.stat().st_size / 1024, 1) if filepath.exists() else None,
            "sheets"   : None,
        }
        if filepath.exists():
            try:
                xl = pd.ExcelFile(filepath)
                record["sheets"] = ", ".join(xl.sheet_names)
            except Exception:
                record["sheets"] = "ERROR_READING"
        records.append(record)

    # TARIF
    tf = config.TARIF_FILE
    records.append({
        "role"     : "reference",
        "area"     : "TARIF",
        "filepath" : str(tf),
        "exists"   : tf.exists(),
        "size_kb"  : round(tf.stat().st_size / 1024, 1) if tf.exists() else None,
        "sheets"   : None,
    })

    return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    inv = get_file_inventory()
    print(inv.to_string(index=False))

"""
src/database.py — SQLite Database Manager untuk Web Dashboard PDAM.

Tanggung Jawab:
- Mengelola koneksi dan skema tabel SQLite (customers, readings, predictions_anomalies, recommendations, processing_logs).
- Menyimpan dan memelihara data historis pelanggan (Requirement 11).
- Mencegah duplikasi data berdasarkan ID Pelanggan (NoPEL) dan Periode (Requirement 13).
- Menyimpan seluruh hasil pemrosesan ke database (Requirement 14).
- Mencatat log audit proses berhasil/gagal ke tabel database dan file log (Requirement 15).
"""

import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
from . import config

# Setup Logger
logger = logging.getLogger("pdam_database")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    log_dir = config.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "pdam_pipeline.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

DB_PATH = config.DATA_DIR / "database" / "pdam_database.db"

# Cache status inisialisasi dalam memori (Memory Flags) untuk optimasi kecepatan
_DB_INITIALIZED: Dict[str, bool] = {}
_DB_SEEDED: Dict[str, bool] = {}


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Membuka koneksi ke SQLite database dengan timeout dan foreign keys."""
    target_path = Path(db_path) if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging untuk performa web
    return conn


def init_database(db_path: Optional[Path] = None) -> None:
    """Inisialisasi tabel dan indeks database jika belum ada."""
    target_path = Path(db_path) if db_path else DB_PATH
    path_key = str(target_path.resolve())
    if _DB_INITIALIZED.get(path_key, False):
        return

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. Master Customers
    cur.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        NoPEL TEXT PRIMARY KEY,
        Nama TEXT,
        Alamat TEXT,
        RAYON TEXT,
        source_area TEXT,
        tarif REAL,
        tarif_name TEXT,
        latitude REAL,
        longitude REAL,
        cluster_id INTEGER,
        cluster_label TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Monthly Readings (Meter Readings & Usage)
    # UNIQUE(NoPEL, Periode) menjamin TIDAK ADA DUPLIKASI DATA
    cur.execute("""
    CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        NoPEL TEXT NOT NULL,
        Periode TEXT NOT NULL,
        period_dt TEXT,
        Ini REAL NOT NULL,
        Lalu REAL NOT NULL,
        pemakaian_air REAL NOT NULL,
        source_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(NoPEL, Periode)
    );
    """)

    # 3. Predictions & Anomaly Results
    # UNIQUE(NoPEL, Periode) menjamin TIDAK ADA DUPLIKASI PREDIKSI
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        NoPEL TEXT NOT NULL,
        Periode TEXT NOT NULL,
        cluster_id INTEGER,
        cluster_label TEXT,
        actual REAL,
        prediction REAL NOT NULL,
        error_abs REAL,
        pct_diff REAL,
        spike_status TEXT NOT NULL,
        severity_score REAL,
        model_version TEXT DEFAULT 'ANN-S1-k3',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(NoPEL, Periode)
    );
    """)

    # 4. Operational Recommendations
    cur.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        NoPEL TEXT NOT NULL,
        Periode TEXT NOT NULL,
        spike_status TEXT NOT NULL,
        priority TEXT NOT NULL,
        action_type TEXT NOT NULL,
        recommendation_text TEXT NOT NULL,
        status_action TEXT DEFAULT 'OPEN',
        assigned_to TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(NoPEL, Periode)
    );
    """)

    # 5. Audit Processing Logs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS processing_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        process_name TEXT NOT NULL,
        status TEXT NOT NULL,
        records_processed INTEGER DEFAULT 0,
        duplicates_skipped INTEGER DEFAULT 0,
        errors_count INTEGER DEFAULT 0,
        message TEXT,
        user_info TEXT DEFAULT 'SYSTEM'
    );
    """)

    # 6. Sistem Autentikasi Petugas PDAM (Users)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'petugas',
        rayon TEXT DEFAULT 'ALL',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Default Administrator: Admin / PDAM-tirtaperwira12#
    default_users = [
        ("Admin", "9d3de5856828828884f811edf2f44ca28a424dab645b8df5f9183caba46d96ba", "Administrator PDAM", "admin", "ALL"),
        ("petugas", "f34fbe9fd488ff8db237addaf068439963f4e276b24ab06b31dc4ef343a848b4", "Petugas Lapangan Utama", "petugas", "ALL"),
        ("petugas_purbalingga", "f34fbe9fd488ff8db237addaf068439963f4e276b24ab06b31dc4ef343a848b4", "Petugas Rayon Purbalingga", "petugas", "010001"),
        ("petugas_bukateja", "f34fbe9fd488ff8db237addaf068439963f4e276b24ab06b31dc4ef343a848b4", "Petugas Rayon Bukateja", "petugas", "020001"),
    ]
    cur.executemany("""
    INSERT OR IGNORE INTO users (username, password_hash, full_name, role, rayon)
    VALUES (?, ?, ?, ?, ?);
    """, default_users)

    # Indeks untuk query performa tinggi
    cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_nopel ON readings(NoPEL);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_readings_periode ON readings(Periode);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_nopel ON predictions_anomalies(NoPEL);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_spike ON predictions_anomalies(spike_status);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_customers_rayon ON customers(RAYON);")

    conn.commit()
    conn.close()
    _DB_INITIALIZED[path_key] = True
    logger.info("✅ Database initialized successfully: %s", str(db_path or DB_PATH))


def log_process(
    process_name: str,
    status: str,
    records_processed: int = 0,
    duplicates_skipped: int = 0,
    errors_count: int = 0,
    message: str = "",
    user_info: str = "SYSTEM",
    db_path: Optional[Path] = None,
) -> None:
    """Mencatat aktivitas audit logging ke database SQLite dan log file."""
    try:
        conn = get_db_connection(db_path)
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO processing_logs (process_name, status, records_processed, duplicates_skipped, errors_count, message, user_info)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (process_name, status, records_processed, duplicates_skipped, errors_count, message, user_info))
        conn.commit()
        conn.close()
        
        log_msg = f"[{status}] {process_name}: Processed={records_processed}, Duplicates={duplicates_skipped}, Errors={errors_count} — {message}"
        if status.upper() == "SUCCESS":
            logger.info(log_msg)
        elif status.upper() == "WARNING":
            logger.warning(log_msg)
        else:
            logger.error(log_msg)
    except Exception as e:
        logger.error("Failed to write to processing_logs: %s", str(e))


def sync_historical_data_to_db(parquet_path: Optional[Path] = None, db_path: Optional[Path] = None) -> int:
    """
    Mengimpor snapshot historis pelanggan dari master_features.parquet ke SQLite
    agar riwayat pembacaan selalu tersedia untuk feature engineering time-series.
    """
    src_pq = Path(parquet_path) if parquet_path else config.DATA_PROCESSED / "master_features.parquet"
    if not src_pq.exists():
        logger.warning("Historical master parquet not found at %s. Skipping sync.", src_pq)
        return 0

    init_database(db_path)
    logger.info("Syncing historical data from %s to SQLite DB...", src_pq)
    
    df = pd.read_parquet(src_pq)
    df["NoPEL"] = df["NoPEL"].astype(str).str.zfill(8)
    
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. Sync Customers table (latest metadata snapshot per NoPEL)
    cust_cols = ["NoPEL", "Nama", "Alamat", "RAYON", "source_area", "tarif", "latitude", "longitude"]
    avail_cust_cols = [c for c in cust_cols if c in df.columns]
    
    df_latest = df.drop_duplicates(subset=["NoPEL"], keep="last")[avail_cust_cols].copy()
    
    # Cluster assignments if available
    clust_pq = config.OUTPUT_CLUST_SEL / "customer_clusters.parquet"
    if clust_pq.exists():
        df_c = pd.read_parquet(clust_pq)[["NoPEL", "cluster_id", "cluster_label"]].drop_duplicates("NoPEL")
        df_c["NoPEL"] = df_c["NoPEL"].astype(str).str.zfill(8)
        df_latest = df_latest.merge(df_c, on="NoPEL", how="left")
    
    records_cust = []
    for _, r in df_latest.iterrows():
        records_cust.append((
            str(r["NoPEL"]),
            str(r.get("Nama", "-")),
            str(r.get("Alamat", "-")),
            str(r.get("RAYON", "-")),
            str(r.get("source_area", "-")),
            float(r.get("tarif", 10.0)) if pd.notna(r.get("tarif")) else 10.0,
            str(r.get("tarif_name", "-")),
            float(r["latitude"]) if pd.notna(r.get("latitude")) else None,
            float(r["longitude"]) if pd.notna(r.get("longitude")) else None,
            int(r["cluster_id"]) if pd.notna(r.get("cluster_id")) else 1,
            str(r.get("cluster_label", "Klaster 2 (Sedang - Rumah Tangga)"))
        ))

    cur.executemany("""
    INSERT INTO customers (NoPEL, Nama, Alamat, RAYON, source_area, tarif, tarif_name, latitude, longitude, cluster_id, cluster_label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(NoPEL) DO UPDATE SET
        Alamat = excluded.Alamat,
        RAYON = excluded.RAYON,
        tarif = excluded.tarif,
        cluster_id = excluded.cluster_id,
        cluster_label = excluded.cluster_label,
        updated_at = CURRENT_TIMESTAMP;
    """, records_cust)

    # 2. Sync Readings (Historical Monthly Records)
    # Ambil periode dan pemakaian
    df["Periode"] = df["Periode"].astype(str) if "Periode" in df.columns else df["period_dt"].dt.strftime("%Y%m")
    df["period_dt_str"] = df["period_dt"].dt.strftime("%Y-%m-%d") if "period_dt" in df.columns else None
    
    if "pemakaian_air" not in df.columns and "Ini" in df.columns and "Lalu" in df.columns:
        df["pemakaian_air"] = pd.to_numeric(df["Ini"], errors="coerce") - pd.to_numeric(df["Lalu"], errors="coerce")
    
    readings_records = []
    for _, r in df[["NoPEL", "Periode", "period_dt_str", "Ini", "Lalu", "pemakaian_air"]].iterrows():
        readings_records.append((
            str(r["NoPEL"]),
            str(r["Periode"]),
            str(r["period_dt_str"]),
            float(r["Ini"]) if pd.notna(r["Ini"]) else 0.0,
            float(r["Lalu"]) if pd.notna(r["Lalu"]) else 0.0,
            float(r["pemakaian_air"]) if pd.notna(r["pemakaian_air"]) else 0.0,
            "master_features.parquet"
        ))

    cur.executemany("""
    INSERT OR IGNORE INTO readings (NoPEL, Periode, period_dt, Ini, Lalu, pemakaian_air, source_file)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """, readings_records)

    conn.commit()
    count_readings = cur.execute("SELECT COUNT(*) FROM readings;").fetchone()[0]
    conn.close()

    log_process(
        process_name="sync_historical_data_to_db",
        status="SUCCESS",
        records_processed=len(readings_records),
        message=f"Synced {len(records_cust):,} customers and {count_readings:,} readings to SQLite."
    )
    return count_readings


def save_batch_inference_results(df_results: pd.DataFrame, source_file: str = "batch_input", db_path: Optional[Path] = None) -> Tuple[int, int]:
    """
    Menyimpan hasil inferensi lengkap (readings, predictions, recommendations) ke SQLite.
    Menggunakan INSERT OR REPLACE / ON CONFLICT DO UPDATE untuk menghindari duplikasi data (Req 13 & 14).
    
    Returns
    -------
    (saved_count, duplicate_updated_count)
    """
    if df_results is None or len(df_results) == 0:
        return 0, 0

    init_database(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    df_save = df_results.copy()
    df_save["NoPEL"] = df_save["NoPEL"].astype(str).str.zfill(8)
    if "Periode" not in df_save.columns and "periode" in df_save.columns:
        df_save["Periode"] = df_save["periode"]
    df_save["Periode"] = df_save["Periode"].astype(str)
    
    # 1. Simpan/Update ke readings
    readings_data = []
    for _, r in df_save.iterrows():
        pdt = str(r["period_dt"]) if "period_dt" in r and pd.notna(r["period_dt"]) else None
        usage_val = r.get("pemakaian_air")
        if pd.isna(usage_val):
            usage_val = r.get("actual_usage", r.get("usage_current", r.get("usage_original", r.get("actual", 0.0))))
        readings_data.append((
            str(r["NoPEL"]),
            str(r["Periode"]),
            pdt,
            float(r["Ini"]) if pd.notna(r.get("Ini")) else 0.0,
            float(r["Lalu"]) if pd.notna(r.get("Lalu")) else 0.0,
            float(usage_val) if pd.notna(usage_val) else 0.0,
            source_file
        ))

    cur.executemany("""
    INSERT INTO readings (NoPEL, Periode, period_dt, Ini, Lalu, pemakaian_air, source_file)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(NoPEL, Periode) DO UPDATE SET
        Ini = excluded.Ini,
        Lalu = excluded.Lalu,
        pemakaian_air = excluded.pemakaian_air,
        source_file = excluded.source_file;
    """, readings_data)

    # 2. Simpan ke predictions_anomalies
    pred_data = []
    for _, r in df_save.iterrows():
        act_val = r.get("actual", r.get("actual_usage", r.get("usage_current")))
        pred_val = r.get("prediction", r.get("predicted_usage", 0.0))
        err_val = r.get("error_abs", abs(r.get("residual", 0.0)))
        pred_data.append((
            str(r["NoPEL"]),
            str(r["Periode"]),
            int(r["cluster_id"]) if pd.notna(r.get("cluster_id")) else 1,
            str(r.get("cluster_label", "Klaster 2 (Sedang - Rumah Tangga)")),
            float(act_val) if pd.notna(act_val) else None,
            float(pred_val) if pd.notna(pred_val) else 0.0,
            float(err_val) if pd.notna(err_val) else None,
            float(r["pct_diff"]) if pd.notna(r.get("pct_diff")) else 0.0,
            str(r.get("spike_status", "NORMAL")),
            float(r["severity_score"]) if pd.notna(r.get("severity_score")) else 0.0,
            "ANN-S1-k3"
        ))

    cur.executemany("""
    INSERT INTO predictions_anomalies (NoPEL, Periode, cluster_id, cluster_label, actual, prediction, error_abs, pct_diff, spike_status, severity_score, model_version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(NoPEL, Periode) DO UPDATE SET
        cluster_id = excluded.cluster_id,
        cluster_label = excluded.cluster_label,
        actual = excluded.actual,
        prediction = excluded.prediction,
        error_abs = excluded.error_abs,
        pct_diff = excluded.pct_diff,
        spike_status = excluded.spike_status,
        severity_score = excluded.severity_score;
    """, pred_data)

    # 3. Simpan ke recommendations jika kolom tersedia
    if "recommendation_text" in df_save.columns or "recommended_action" in df_save.columns:
        rec_data = []
        for _, r in df_save.iterrows():
            rec_text = r.get("recommendation_text", r.get("recommended_action", "-"))
            rec_prio = r.get("action_priority", r.get("recommendation_priority", "NORMAL"))
            rec_type = r.get("action_type", r.get("anomaly_category", "Billing Rutin"))
            rec_data.append((
                str(r["NoPEL"]),
                str(r["Periode"]),
                str(r.get("spike_status", "NORMAL")),
                str(rec_prio),
                str(rec_type),
                str(rec_text),
                "OPEN"
            ))
        cur.executemany("""
        INSERT INTO recommendations (NoPEL, Periode, spike_status, priority, action_type, recommendation_text, status_action)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(NoPEL, Periode) DO UPDATE SET
            spike_status = excluded.spike_status,
            priority = excluded.priority,
            action_type = excluded.action_type,
            recommendation_text = excluded.recommendation_text;
        """, rec_data)

    conn.commit()
    conn.close()

    log_process(
        process_name="save_batch_inference_results",
        status="SUCCESS",
        records_processed=len(df_save),
        message=f"Saved {len(df_save):,} records to SQLite database without duplicates."
    )
    return len(df_save), 0


def get_customer_historical_readings(nopel: str, limit_months: int = 12, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Mengambil riwayat konsumsi pelanggan per NoPEL dari SQLite untuk feature engineering time-series."""
    nopel_clean = str(nopel).zfill(8)
    conn = get_db_connection(db_path)
    query = """
    SELECT NoPEL, Periode, period_dt, Ini, Lalu, pemakaian_air
    FROM readings
    WHERE NoPEL = ?
    ORDER BY Periode ASC
    """
    df_hist = pd.read_sql_query(query, conn, params=(nopel_clean,))
    conn.close()
    
    if len(df_hist) > limit_months:
        df_hist = df_hist.tail(limit_months)
    return df_hist


def check_existing_keys(
    pairs_nopel_periode: Union[List[Tuple[str, str]], pd.DataFrame],
    db_path: Optional[Path] = None,
) -> set:
    """Mengecek pasangan (NoPEL, Periode) yang sudah ada di database untuk mencegah duplikasi (Req 13)."""
    if isinstance(pairs_nopel_periode, pd.DataFrame):
        if pairs_nopel_periode.empty:
            return set()
        df_sub = pairs_nopel_periode.copy()
        per_col = "periode" if "periode" in df_sub.columns else "Periode"
        pairs = [
            (str(r["NoPEL"]).zfill(8), str(r[per_col]))
            for _, r in df_sub.iterrows()
            if pd.notna(r.get("NoPEL")) and pd.notna(r.get(per_col))
        ]
    else:
        if not pairs_nopel_periode:
            return set()
        pairs = pairs_nopel_periode

    if not pairs:
        return set()

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    
    # Query batched
    existing = set()
    batch_size = 500
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        placeholders = ",".join(["(?, ?)"] * len(chunk))
        flattened = [item for sublist in chunk for item in sublist]
        query = f"SELECT NoPEL, Periode FROM readings WHERE (NoPEL, Periode) IN (VALUES {placeholders});"
        try:
            cur.execute(query, flattened)
            for row in cur.fetchall():
                existing.add((str(row[0]).zfill(8), str(row[1])))
        except Exception:
            # Fallback for simpler SQL syntax
            for nopel, per in chunk:
                cur.execute("SELECT 1 FROM readings WHERE NoPEL = ? AND Periode = ? LIMIT 1;", (nopel, per))
                if cur.fetchone():
                    existing.add((nopel, per))
                    
    conn.close()
    return existing


def get_recent_logs(limit: int = 50, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Mengambil riwayat log audit sistem terbaru dari database SQLite."""
    conn = get_db_connection(db_path)
    query = """
    SELECT id, timestamp, process_name, status, records_processed, duplicates_skipped, errors_count, message, user_info
    FROM processing_logs
    ORDER BY id DESC
    LIMIT ?
    """
    df_logs = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df_logs


def get_database_stats(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Mengambil ringkasan jumlah baris data di setiap tabel SQLite."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    stats = {}
    tables = ["customers", "readings", "predictions_anomalies", "recommendations", "processing_logs", "users"]
    for t in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            stats[t] = cur.fetchone()[0]
        except Exception:
            stats[t] = 0
    conn.close()
    return stats


# =============================================================================
# AUTENTIKASI PETUGAS PDAM
# =============================================================================

def hash_password(password: str) -> str:
    """Hash password menggunakan SHA-256."""
    import hashlib
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def authenticate_user(
    username: str,
    password: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validasi login petugas PDAM.

    Parameters
    ----------
    username : str
    password : str
    db_path : Optional[Path]

    Returns
    -------
    Dict user info jika sukses, None jika gagal.
    """
    init_database(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    pwd_hash = hash_password(password)

    cur.execute("""
    SELECT username, full_name, role, rayon
    FROM users
    WHERE LOWER(username) = LOWER(?) AND password_hash = ?
    """, (username.strip(), pwd_hash))
    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "username" : row[0],
            "full_name": row[1],
            "role"     : row[2],
            "rayon"    : row[3],
        }
    return None


def update_user_credentials(
    current_username: str,
    new_username: str,
    new_password: Optional[str] = None,
    full_name: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Memperbarui username, nama lengkap, dan/atau password akun di tabel users SQLite.
    
    Parameters
    ----------
    current_username : str
        Username pengguna saat ini.
    new_username : str
        Username baru yang diinginkan.
    new_password : Optional[str]
        Kata sandi baru (plain text). Jika None atau kosong, password lama dipertahankan.
    full_name : Optional[str]
        Nama lengkap baru.
    db_path : Optional[Path]
        Jalur ke SQLite database.
        
    Returns
    -------
    Tuple[bool, str]
        (sukses, pesan_status)
    """
    try:
        init_database(db_path)
        conn = get_db_connection(db_path)
        cur = conn.cursor()

        clean_cur_uname = current_username.strip()
        clean_new_uname = new_username.strip()

        # Cek apakah username baru sudah digunakan oleh akun lain
        if clean_new_uname.lower() != clean_cur_uname.lower():
            cur.execute("SELECT id FROM users WHERE username = ?", (clean_new_uname,))
            if cur.fetchone():
                conn.close()
                return False, f"Username '{clean_new_uname}' sudah digunakan akun lain. Silakan pilih username lain."

        if new_password and len(new_password.strip()) > 0:
            pwd_hash = hash_password(new_password.strip())
            if full_name and len(full_name.strip()) > 0:
                cur.execute("""
                    UPDATE users 
                    SET username = ?, password_hash = ?, full_name = ?
                    WHERE username = ?
                """, (clean_new_uname, pwd_hash, full_name.strip(), clean_cur_uname))
            else:
                cur.execute("""
                    UPDATE users 
                    SET username = ?, password_hash = ?
                    WHERE username = ?
                """, (clean_new_uname, pwd_hash, clean_cur_uname))
        else:
            if full_name and len(full_name.strip()) > 0:
                cur.execute("""
                    UPDATE users 
                    SET username = ?, full_name = ?
                    WHERE username = ?
                """, (clean_new_uname, full_name.strip(), clean_cur_uname))
            else:
                cur.execute("""
                    UPDATE users 
                    SET username = ?
                    WHERE username = ?
                """, (clean_new_uname, clean_cur_uname))

        conn.commit()
        conn.close()
        logger.info("Akun '%s' berhasil diperbarui menjadi '%s'.", clean_cur_uname, clean_new_uname)
        return True, "Profil dan kredensial akun berhasil diperbarui."
    except Exception as e:
        logger.error("Gagal memperbarui akun: %s", str(e))
        return False, f"Gagal memperbarui akun: {str(e)}"


# =============================================================================
# AUTOMATIC SEEDING DARI ARTEFAK PENELITIAN
# =============================================================================

def seed_database_if_empty(db_path: Optional[Path] = None) -> int:
    """
    Mengisi database SQLite secara otomatis dari artefak penelitian tersimpan
    jika tabel readings atau predictions_anomalies masih kosong.
    """
    target_path = Path(db_path) if db_path else DB_PATH
    path_key = str(target_path.resolve())
    if _DB_SEEDED.get(path_key, False):
        return 1

    init_database(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions_anomalies")
    count = cur.fetchone()[0]
    if count > 0:
        conn.close()
        _DB_SEEDED[path_key] = True
        return count

    seed_pq = config.BASE_DIR / "app" / "assets" / "streamlit_app_data.parquet"
    if not seed_pq.exists():
        seed_pq = config.OUTPUT_MAPS / "streamlit_app_data.parquet"
    if not seed_pq.exists():
        conn.close()
        return 0

    logger.info(f"Mengisi database otomatis dari {seed_pq} ({seed_pq.stat().st_size / 1024:.1f} KB)...")
    df = pd.read_parquet(seed_pq)

    cust_data = []
    read_data = []
    pred_data = []
    rec_data = []

    for _, r in df.iterrows():
        nopel  = str(r["NoPEL"]).zfill(8)
        rayon  = str(r.get("RAYON", "-"))
        alamat = str(r.get("Alamat", "-"))
        tarif  = float(r.get("tarif", 1.0)) if pd.notna(r.get("tarif")) else 1.0
        lati   = float(r.get("latitude")) if pd.notna(r.get("latitude")) else None
        longi  = float(r.get("longitude")) if pd.notna(r.get("longitude")) else None
        cid    = int(r.get("cluster_id", 1)) if pd.notna(r.get("cluster_id")) else 1
        clabel = str(r.get("cluster_label", f"Klaster {cid+1}"))
        actual = float(r.get("actual", 0.0)) if pd.notna(r.get("actual")) else 0.0
        pred   = float(r.get("prediction", 0.0)) if pd.notna(r.get("prediction")) else 0.0
        pdiff  = float(r.get("pct_diff", 0.0)) if pd.notna(r.get("pct_diff")) else 0.0
        status = str(r.get("spike_status", "NORMAL"))
        err_abs = abs(actual - pred)

        cust_data.append((
            nopel, f"Pelanggan {nopel}", alamat, rayon,
            str(r.get("source_area", "PERKOTAAN")), tarif, lati, longi, cid, clabel
        ))
        read_data.append((
            nopel, "202512", "2025-12-01", pred + actual, pred, actual, "seed_research_data"
        ))
        pred_data.append((
            nopel, "202512", cid, clabel, actual, pred, err_abs, pdiff, status,
            3.0 if status == "INDIKASI_LONJAKAN" else 0.0, "ANN-S1-k3"
        ))

        priority = "TINGGI" if (pdiff > 100 or actual - pred > 50) else (
            "SEDANG" if status == "INDIKASI_LONJAKAN" else ("RENDAH" if status == "ANOMALI_RENDAH" else "NORMAL")
        )
        action_type = "Investigasi Kebocoran" if status == "INDIKASI_LONJAKAN" else (
            "Uji Meter Macet" if status == "ANOMALI_RENDAH" else "Pembacaan Normal"
        )
        rec_text = "Terbitkan SPK Investigasi Lapangan" if priority == "TINGGI" else (
            "Verifikasi Petugas Cater" if priority == "SEDANG" else (
                "Monitoring Pemakaian Rutin" if priority == "RENDAH" else "Tagihan Sesuai Pemakaian"
            )
        )
        rec_data.append((nopel, "202512", status, priority, action_type, rec_text))

    cur.executemany("""
    INSERT OR REPLACE INTO customers (NoPEL, Nama, Alamat, RAYON, source_area, tarif, latitude, longitude, cluster_id, cluster_label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, cust_data)

    cur.executemany("""
    INSERT OR REPLACE INTO readings (NoPEL, Periode, period_dt, Ini, Lalu, pemakaian_air, source_file)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, read_data)

    cur.executemany("""
    INSERT OR REPLACE INTO predictions_anomalies (NoPEL, Periode, cluster_id, cluster_label, actual, prediction, error_abs, pct_diff, spike_status, severity_score, model_version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, pred_data)

    cur.executemany("""
    INSERT OR REPLACE INTO recommendations (NoPEL, Periode, spike_status, priority, action_type, recommendation_text)
    VALUES (?, ?, ?, ?, ?, ?)
    """, rec_data)

    conn.commit()
    conn.close()
    logger.info(f"✅ Berhasil menyuntikkan {len(cust_data):,} data penelitian ke SQLite database.")
    return len(cust_data)


# =============================================================================
# QUERY HELPER UNTUK STREAMLIT DASHBOARD
# =============================================================================

def get_dashboard_data(
    rayon: Optional[Union[str, List[str]]] = None,
    periode: Optional[str] = None,
    cluster_id: Optional[Union[int, List[int]]] = None,
    spike_status: Optional[Union[str, List[str]]] = None,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Ambil dataset gabungan lengkap (customers + readings + predictions + recommendations).
    """
    seed_database_if_empty(db_path)
    conn = get_db_connection(db_path)

    query = """
    SELECT 
        c.NoPEL,
        c.Nama,
        c.Alamat,
        c.RAYON,
        c.source_area,
        c.tarif,
        c.latitude,
        c.longitude,
        p.Periode,
        r.Ini,
        r.Lalu,
        r.pemakaian_air,
        p.cluster_id,
        p.cluster_label,
        p.actual,
        p.prediction,
        p.error_abs,
        p.pct_diff,
        p.spike_status,
        p.severity_score,
        rec.priority AS recommendation_priority,
        rec.action_type,
        rec.recommendation_text
    FROM customers c
    JOIN predictions_anomalies p ON c.NoPEL = p.NoPEL
    LEFT JOIN readings r ON c.NoPEL = r.NoPEL AND p.Periode = r.Periode
    LEFT JOIN recommendations rec ON c.NoPEL = rec.NoPEL AND p.Periode = rec.Periode
    WHERE 1=1
    """
    params = []

    if rayon:
        if isinstance(rayon, list) and len(rayon) > 0 and "ALL" not in rayon:
            ph = ",".join(["?"] * len(rayon))
            query += f" AND c.RAYON IN ({ph})"
            params.extend(rayon)
        elif isinstance(rayon, str) and rayon != "ALL":
            query += " AND c.RAYON = ?"
            params.append(rayon)

    if periode and periode != "ALL":
        query += " AND p.Periode = ?"
        params.append(periode)

    if cluster_id is not None:
        if isinstance(cluster_id, list) and len(cluster_id) > 0:
            ph = ",".join(["?"] * len(cluster_id))
            query += f" AND p.cluster_id IN ({ph})"
            params.extend(cluster_id)
        elif isinstance(cluster_id, int):
            query += " AND p.cluster_id = ?"
            params.append(cluster_id)

    if spike_status:
        if isinstance(spike_status, list) and len(spike_status) > 0 and "ALL" not in spike_status:
            ph = ",".join(["?"] * len(spike_status))
            query += f" AND p.spike_status IN ({ph})"
            params.extend(spike_status)
        elif isinstance(spike_status, str) and spike_status != "ALL":
            query += " AND p.spike_status = ?"
            params.append(spike_status)

    query += " ORDER BY p.pct_diff DESC"
    if limit:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Merge riwayat histori 12 periode konsumsi untuk sparkline peta jika tersedia
    seed_pq = config.BASE_DIR / "app" / "assets" / "streamlit_app_data.parquet"
    if not seed_pq.exists():
        seed_pq = config.OUTPUT_MAPS / "streamlit_app_data.parquet"
    if seed_pq.exists() and not df.empty:
        try:
            df_pq = pd.read_parquet(seed_pq)[["NoPEL", "history_usage_json", "history_periods_json", "keterangan"]]
            df_pq["NoPEL"] = df_pq["NoPEL"].astype(str).str.zfill(8)
            df = df.merge(df_pq, on="NoPEL", how="left")
        except Exception as e:
            logger.warning("Could not merge historical sparkline data from parquet: %s", str(e))

    return df


def get_available_rayons(db_path: Optional[Path] = None) -> List[str]:
    """Daftar rayon unik di database."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT RAYON FROM customers WHERE RAYON IS NOT NULL AND RAYON != '' ORDER BY RAYON")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def get_available_periods(db_path: Optional[Path] = None) -> List[str]:
    """Daftar periode unik di database."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT Periode FROM readings WHERE Periode IS NOT NULL ORDER BY Periode DESC")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

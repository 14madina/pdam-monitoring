"""
src/config.py — Konfigurasi terpusat project PDAM Skripsi.

Semua path, konstanta eksperimen, dan parameter diletakkan di sini.
Notebook dan modul lain mengimpor dari sini agar konsisten.
"""

from pathlib import Path

# =============================================================================
# ROOT PROJECT
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR     = PROJECT_ROOT

# =============================================================================
# DATA PATHS
# =============================================================================
DATA_DIR       = PROJECT_ROOT / "data"
DATA_RAW       = DATA_DIR / "raw"
DATA_REFERENCE = DATA_DIR / "reference"
DATA_INTERIM   = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_DATABASE  = DATA_DIR / "database"
DATABASE_PATH  = DATA_DATABASE / "pdam_database.db"

# =============================================================================
# REPORTS
# =============================================================================
REPORTS_DIR            = PROJECT_ROOT / "reports"
REPORT_DATA_UNDERSTAND = REPORTS_DIR / "data_understanding"
REPORT_DATA_QUALITY    = REPORTS_DIR / "data_quality"
REPORT_IMPUTATION      = REPORTS_DIR / "imputation"
REPORT_FEATURE_SEL     = REPORTS_DIR / "feature_selection"
REPORT_CLUSTERING      = REPORTS_DIR / "clustering"
REPORT_PREDICTION      = REPORTS_DIR / "prediction"
REPORT_EVALUATION      = REPORTS_DIR / "evaluation"

# =============================================================================
# MODELS
# =============================================================================
MODELS_DIR           = PROJECT_ROOT / "models"
MODEL_KMEANS_ALL     = MODELS_DIR / "kmeans" / "all_features"
MODEL_KMEANS_SEL     = MODELS_DIR / "kmeans" / "selected_features"
MODEL_ANN_GLOBAL_ALL = MODELS_DIR / "ann_global" / "all_features"
MODEL_ANN_GLOBAL_SEL = MODELS_DIR / "ann_global" / "selected_features"
MODEL_ANN_CLUST_ALL  = MODELS_DIR / "ann_cluster" / "all_features"
MODEL_ANN_CLUST_SEL  = MODELS_DIR / "ann_cluster" / "selected_features"

# =============================================================================
# OUTPUTS
# =============================================================================
OUTPUTS_DIR          = PROJECT_ROOT / "outputs"
OUTPUT_CLUST_ALL     = OUTPUTS_DIR / "clustering" / "all_features"
OUTPUT_CLUST_SEL     = OUTPUTS_DIR / "clustering" / "selected_features"
OUTPUT_PRED_ALL      = OUTPUTS_DIR / "prediction" / "all_features"
OUTPUT_PRED_SEL      = OUTPUTS_DIR / "prediction" / "selected_features"
OUTPUT_SPIKE         = OUTPUTS_DIR / "spike_detection"
OUTPUT_MAPS          = OUTPUTS_DIR / "maps"
OUTPUT_EXCEL         = OUTPUTS_DIR / "excel"

# =============================================================================
# APP
# =============================================================================
APP_DIR = PROJECT_ROOT / "app"

# =============================================================================
# DATA FILES — Nama file Excel di data/raw/
# Sesuaikan nama file sesuai file Excel yang dimiliki.
# =============================================================================
RAW_FILES = {
    "PERKOTAAN"   : DATA_RAW / "PERKOTAAN.xlsx",
    "BUKATEJA"    : DATA_RAW / "BUKATEJA.xlsx",
    "KEMANGKON-1" : DATA_RAW / "KEMANGKON-1.xlsx",
    "KEMANGKON-2" : DATA_RAW / "KEMANGKON-2.xlsx",
    "KEMANGKON-3" : DATA_RAW / "KEMANGKON-3.xlsx",
}

TARIF_FILE = DATA_REFERENCE / "TARIF.xlsx"

# Sheet names
PERKOTAAN_SHEETS = ["PERKOTAAN_2020_2024", "LANJUTAN_2024_2025"]
WILAYAH_SHEET    = "Table1"   # untuk BUKATEJA dan KEMANGKON-*

# =============================================================================
# AREA CONFIG
# =============================================================================
SOURCE_AREAS = ["PERKOTAAN", "BUKATEJA", "KEMANGKON-1", "KEMANGKON-2", "KEMANGKON-3"]

# =============================================================================
# SCHEMA — Kolom minimal, 15 kolom terpadu, dan kolom terpadu lengkap dengan tarif
# =============================================================================
CONSOLIDATED_COLUMNS = [
    "nopel", "tglbaca", "waktu", "alamat", "lalu", "ini",
    "tarif", "cater", "rayon", "status", "periode", "keterangan",
    "lati", "longi", "alti"
]

CONSOLIDATED_WITH_TARIFF_COLUMNS = [
    "nopel", "tglbaca", "waktu", "alamat", "lalu", "ini",
    "tarif", "golongan_tarif", "estimated_bill_rp",
    "cater", "rayon", "status", "periode", "keterangan",
    "lati", "longi", "alti", "source_area"
]

SCHEMA_COLUMNS = [
    "source_area",
    "source_file",
    "source_sheet",
    "NoPEL",
    "Nama",
    "Alamat",
    "periode",
    "period_dt",
    "tglBaca",
    "Waktu",
    "Lalu",
    "Ini",
    "usage_original",
    "RAYON",
    "Status",
    "keterangan",
    "Cater",
    "latitude",
    "longitude",
    "lati",
    "longi",
    "alti",
    "time",
    "updatedDate",
    "updatedBy",
    "tarif",
    "golongan_tarif",
    "estimated_bill_rp",
]

# Kolom NoPEL disimpan sebagai string
NOPEL_DTYPE = str

# =============================================================================
# EXPERIMENT CONSTANTS
# =============================================================================
RANDOM_STATE = 42

# Time split rasio (kronologis, unique months) — 80:10:10
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# K-Means: range K yang diuji
KMEANS_K_RANGE = list(range(2, 9))  # K = 2, 3, 4, 5, 6, 7, 8

# Feature selection: minimum vote count untuk dipilih
FEATURE_SELECTION_MIN_VOTES = 3

# =============================================================================
# SPIKE DETECTION
# =============================================================================
SPIKE_THRESHOLD_PCT  =  30.0   # > +30% → INDIKASI_LONJAKAN
ANOMALY_THRESHOLD_PCT = -30.0  # < -30% → ANOMALI_RENDAH
SPIKE_EPSILON        = 1e-6    # Hindari pembagian dengan nol

SPIKE_STATUS_LONJAKAN = "INDIKASI_LONJAKAN"
SPIKE_STATUS_RENDAH   = "ANOMALI_RENDAH"
SPIKE_STATUS_NORMAL   = "NORMAL"

# =============================================================================
# SCENARIO NAMES — Gunakan nama ini secara konsisten di seluruh project
# =============================================================================
SCENARIO_ALL      = "S0_ALL_FEATURES"
SCENARIO_SELECTED = "S1_SELECTED_FEATURES"

SCENARIO_LABELS = {
    SCENARIO_ALL      : "Without Feature Selection (All Features)",
    SCENARIO_SELECTED : "With Feature Selection (Selected Features)",
}

# =============================================================================
# IMPUTASI
# =============================================================================
IMPUTATION_METHODS = ["historical_mean", "historical_median", "historical_mode", "historical_min_positive"]

# =============================================================================
# FEATURE NAMES — Fitur HANYA diambil dari data gabungan + tarif terhitung
# (TIDAK menyertakan fitur tambahan seperti lag, rolling, delta, hist, dsb.)
# =============================================================================
TARGET_COL = "target_next_usage"

MERGED_DATASET_FEATURES = [
    "NoPEL",
    "Nama",
    "Alamat",
    "tglBaca",
    "Waktu",
    "Lalu",
    "Ini",
    "tarif",
    "Cater",
    "RAYON",
    "Status",
    "periode",
    "keterangan",
    "lati",
    "longi",
    "alti",
    "time",
    "updatedDate",
    "updatedBy",
]

ALL_CANDIDATE_FEATURES = MERGED_DATASET_FEATURES

# Model Features untuk S0 dan S1
ALL_FEATURES = ["Ini", "Lalu", "tarif", "lati", "longi", "alti", "time"]
SELECTED_FEATURES = ["Ini", "Lalu", "tarif"]

# =============================================================================
# ANN HYPERPARAMETERS (sama untuk S0 dan S1 agar perbandingan adil)
# =============================================================================
ANN_CONFIG = {
    "hidden_layers"    : [128, 64, 32],
    "activation"       : "relu",
    "dropout_rate"     : 0.05,
    "learning_rate"    : 0.005,
    "batch_size"       : 4096,
    "max_epochs"       : 100,
    "use_early_stopping": False,  # Belajar penuh 100 epoch tanpa berhenti di tengah jalan
    "early_stopping_patience": None,
    "reduce_lr_patience": 5,
    "optimizer"        : "adam",
    "loss"             : "mse",
}

# =============================================================================
# SCALER NAMES
# =============================================================================
SCALER_KMEANS = "MinMaxScaler"
SCALER_ANN    = "StandardScaler"

# =============================================================================
# UTILITY: Buat semua direktori jika belum ada
# =============================================================================
def ensure_dirs():
    """Pastikan semua direktori output tersedia."""
    dirs = [
        DATA_RAW, DATA_REFERENCE, DATA_INTERIM, DATA_PROCESSED,
        REPORT_DATA_UNDERSTAND, REPORT_DATA_QUALITY, REPORT_IMPUTATION,
        REPORT_FEATURE_SEL, REPORT_CLUSTERING, REPORT_PREDICTION, REPORT_EVALUATION,
        MODEL_KMEANS_ALL, MODEL_KMEANS_SEL,
        MODEL_ANN_GLOBAL_ALL, MODEL_ANN_GLOBAL_SEL,
        MODEL_ANN_CLUST_ALL, MODEL_ANN_CLUST_SEL,
        OUTPUT_CLUST_ALL, OUTPUT_CLUST_SEL,
        OUTPUT_PRED_ALL, OUTPUT_PRED_SEL,
        OUTPUT_SPIKE, OUTPUT_MAPS, OUTPUT_EXCEL,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("✅ Config loaded. PROJECT_ROOT =", PROJECT_ROOT)
    print("   RAW_FILES:")
    for k, v in RAW_FILES.items():
        print(f"     {k}: {v}")

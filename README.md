

---

## Deskripsi
Project ini mengimplementasikan kerangka **CRISP-DM** untuk analisis dan prediksi pemakaian air PDAM dari lima wilayah:
- PERKOTAAN
- BUKATEJA
- KEMANGKON-1, KEMANGKON-2, KEMANGKON-3

Terdapat **dua skenario eksperimen utama**:
- **S0 (All Features)**: model menggunakan seluruh candidate feature hasil feature extraction
- **S1 (Selected Features)**: model menggunakan subset fitur yang lolos 5 metode feature selection

---

## Struktur Folder

```
pdam_skripsi_project/
├── data/
│   ├── raw/          ← Letakkan file Excel PDAM di sini
│   ├── reference/    ← TARIF.xlsx
│   ├── interim/      ← master_merged.parquet
│   └── processed/    ← master_clean.parquet, master_features.parquet
├── notebooks/        ← 12 notebook berurutan (00–11)
├── src/              ← Modul Python reusable
├── models/           ← Model tersimpan (K-Means, ANN)
├── reports/          ← CSV dan grafik laporan
├── outputs/          ← Hasil akhir (clustering, prediksi, spike)
└── app/              ← Streamlit dashboard
```

---

## Setup Environment

### 1. Buat virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Siapkan data
Letakkan file Excel data PDAM ke folder `data/raw/`:
```
data/raw/
├── PERKOTAAN.xlsx        # Sheet: PERKOTAAN_2020_2024, LANJUTAN_2024_2025
├── BUKATEJA.xlsx         # Sheet: Table1
├── KEMANGKON-1.xlsx      # Sheet: Table1
├── KEMANGKON-2.xlsx      # Sheet: Table1
└── KEMANGKON-3.xlsx      # Sheet: Table1

data/reference/
└── TARIF.xlsx            # Referensi tarif (BUKAN data pelanggan)
```

> **PENTING**: Nama file Excel bisa disesuaikan di `src/config.py` pada variabel `RAW_FILES`.

---

## Urutan Eksekusi Notebook

```
00_business_understanding.ipynb
        ↓
01_data_understanding_eda.ipynb
        ↓ [CHECKPOINT DU — berhenti, review schema]
02_data_preparation_merge_clean.ipynb
        ↓
03_imputation_experiment.ipynb
        ↓ [CHECKPOINT IMPUTATION — pilih metode]
04_post_imputation_distribution_check.ipynb
        ↓
05_feature_extraction.ipynb
        ↓
06_feature_selection.ipynb
        ↓ [CHECKPOINT FEATURES — review S0 vs S1]
07_modeling_kmeans_feature_comparison.ipynb
        ↓ [CHECKPOINT K-MEANS — approve K final]
08_modeling_ann_feature_comparison.ipynb
        ↓
09_evaluation_final_comparison.ipynb
        ↓ [CHECKPOINT MODEL — approve pipeline final]
10_spike_detection.ipynb
        ↓
11_deployment_export.ipynb
```

---

## Menjalankan Dashboard

```bash
streamlit run app/app.py
```

---

## Konfigurasi Utama

Semua konstanta ada di `src/config.py`:
- Path folder data, model, report, output
- `RANDOM_STATE = 42`
- Rasio split: Train 70% / Validation 15% / Test 15%
- Wilayah data: `SOURCE_AREAS`
- Nama file Excel: `RAW_FILES`
- Threshold spike detection: 30%
- K range K-Means: 2–8

---

## Metrik Evaluasi

### K-Means
| Metrik | Arah |
|---|---|
| Silhouette Score | ↑ lebih baik |
| Davies-Bouldin Index | ↓ lebih baik |
| Calinski-Harabasz Index | ↑ lebih baik |
| Inertia/WCSS | Elbow analysis |

### ANN Regression
| Metrik | Arah |
|---|---|
| RMSE | ↓ lebih baik |
| MAE | ↓ lebih baik |
| MARE | ↓ lebih baik |
| NSC/NSE | ↑ lebih baik |
| R² | ↑ lebih baik |

---

## Critical Rules
1. Tidak boleh sampling data modeling
2. Tidak boleh edit file raw
3. TARIF tidak dicampur sebagai record pelanggan
4. Feature extraction hanya satu kali
5. Feature selection di-fit pada Train saja
6. Scaling/encoding di-fit pada Train saja
7. Test set tidak untuk tuning/hyperparameter selection
8. Semua output eksperimen disimpan untuk reproducibility
9. `random_state = 42` untuk semua eksperimen

---

## Reproducibility
- `random_state = 42` di semua eksperimen
- Feature list, encoder, scaler, model, hyperparameter, periode split, dan metrik semuanya disimpan
- Model tidak di-overwrite tanpa versi

---

## Dependencies Utama
- Python 3.9+
- pandas, numpy, scipy
- scikit-learn (K-Means, feature selection)
- TensorFlow/Keras (ANN)
- matplotlib, seaborn, plotly (visualisasi)
- streamlit (dashboard)
- openpyxl, pyarrow (I/O)

"""
src/pipeline_runner.py — Master Orchestration Pipeline PDAM Dashboard.

Menyatukan seluruh tahapan inferensi model dan integrasi data:
1. Validasi & Preprocessing (Excel Input / DataFrame)
2. Pengecekan duplikasi ke Database SQLite
3. Pengelompokan K-Means (K=3, Skenario S1 3 fitur)
4. Prediksi Pemakaian Air ANN Regression (Model Per-Klaster & Global)
5. Deteksi Lonjakan & Anomali Pemakaian (Threshold ±30%)
6. Rekomendasi Operasional & Surat Perintah Kerja (SPK)
7. Penyimpanan otomatis ke Database SQLite & Audit Logging

Data baru diproses langsung on-the-fly tanpa perlu menjalankan ulang notebook riset.
"""

import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import (
    clustering,
    config,
    database,
    feature_engineering,
    prediction,
    preprocessing,
    recommendation,
    spike_detection,
)

logger = logging.getLogger("pdam_pipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(ch)


def run_batch_pipeline(
    file_or_df: Union[str, Path, pd.DataFrame, Any],
    source_name: Optional[str] = None,
    save_to_db: bool = True,
    prediction_mode: str = "cluster_specific",
    scenario: str = "selected_features",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Eksekusi pipeline terpadu dari file Excel atau DataFrame mentah hingga tersimpan di database.

    Parameters
    ----------
    file_or_df : Union[str, Path, pd.DataFrame, Any]
        Data masukan (path file Excel, buffer file upload Streamlit/FastAPI, atau DataFrame).
    source_name : Optional[str]
        Nama file atau identitas sumber batch data (default: nama file atau 'Web_Upload').
    save_to_db : bool
        Apakah hasil inferensi disimpan otomatis ke database SQLite (default True).
    prediction_mode : str
        'cluster_specific' (default) atau 'global'.
    scenario : str
        'selected_features' (S1 - 3 fitur) atau 'all_features' (S0 - 7 fitur).
    db_path : Optional[Path]
        Path SQLite database.

    Returns
    -------
    Dict[str, Any] berisi:
        - success : bool
        - total_records : int
        - df_result : pd.DataFrame lengkap siap divisualisasikan
        - cluster_summary : dict jumlah per klaster
        - spike_summary : dict status anomali
        - recommendation_summary : dict prioritas SPK
        - execution_time_sec : float
        - warnings : list
        - errors : list
    """
    start_time = time.time()
    errors = []
    warnings = []

    # Penentuan nama sumber data
    if source_name:
        src_label = source_name
    elif isinstance(file_or_df, (str, Path)):
        src_label = Path(file_or_df).name
    else:
        src_label = f"Upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"🚀 Memulai Pipeline PDAM untuk sumber: {src_label}...")

    # Log proses start ke database
    try:
        database.log_process(
            process_name=f"batch_pipeline_{src_label}",
            status="RUNNING",
            records_processed=0,
            message="Pipeline batch execution started",
            db_path=db_path,
        )
    except Exception as e:
        logger.warning(f"Gagal mencatat log awal ke DB: {e}")

    try:
        # ---------------------------------------------------------------------
        # TAHAP 1: VALIDASI & PREPROCESSING DATA INPUT
        # ---------------------------------------------------------------------
        logger.info("Tahap 1: Validasi & Preprocessing input...")
        df_clean, prep_summary = preprocessing.preprocess_new_batch(
            file_or_df,
            check_db_duplicates=True,
            db_path=db_path,
        )
        warnings.extend(prep_summary["validation"]["warnings"])

        if df_clean.empty:
            raise ValueError("Data bersih kosong setelah tahap preprocessing.")

        # ---------------------------------------------------------------------
        # TAHAP 2: K-MEANS CLUSTERING (K=3)
        # ---------------------------------------------------------------------
        logger.info("Tahap 2: Pengelompokan Klaster K-Means...")
        df_clustered = clustering.predict_cluster(
            df_clean,
            scenario=scenario,
        )

        # ---------------------------------------------------------------------
        # TAHAP 3: ANN REGRESSION PREDICTION
        # ---------------------------------------------------------------------
        logger.info(f"Tahap 3: Prediksi ANN Regression (Mode: {prediction_mode}, Skenario: {scenario})...")
        df_predicted = prediction.predict_water_usage(
            df_clustered,
            mode=prediction_mode,
            scenario=scenario,
        )

        # ---------------------------------------------------------------------
        # TAHAP 4: DETEKSI LONJAKAN & ANOMALI
        # ---------------------------------------------------------------------
        logger.info("Tahap 4: Deteksi Lonjakan & Anomali Pemakaian Air (Threshold ±30%)...")
        df_spikes = spike_detection.detect_spikes_from_predictions(df_predicted)

        # ---------------------------------------------------------------------
        # TAHAP 5: GENERATE REKOMENDASI & SPK
        # ---------------------------------------------------------------------
        logger.info("Tahap 5: Pembuatan Rekomendasi Operasional & SPK...")
        df_final = recommendation.generate_recommendations(df_spikes)

        # ---------------------------------------------------------------------
        # TAHAP 6: PENYIMPANAN KE DATABASE (ANTI DUPLIKASI NoPEL + PERIODE)
        # ---------------------------------------------------------------------
        n_saved = 0
        if save_to_db:
            logger.info("Tahap 6: Menyimpan hasil ke SQLite database...")
            n_saved, _ = database.save_batch_inference_results(
                df_results=df_final,
                source_file=src_label,
                db_path=db_path,
            )

        elapsed_time = round(time.time() - start_time, 2)

        # Ringkasan hasil untuk UI Dashboard
        clust_counts = df_final["cluster_label"].value_counts().to_dict()
        spike_counts = df_final["spike_status"].value_counts().to_dict()
        rec_summary = recommendation.get_recommendation_summary(df_final)

        # Log proses sukses ke DB
        try:
            database.log_process(
                process_name=f"batch_pipeline_{src_label}",
                status="SUCCESS",
                records_processed=len(df_final),
                message=f"Pipeline batch completed in {elapsed_time}s. Saved {n_saved:,} records.",
                db_path=db_path,
            )
        except Exception as e:
            logger.warning(f"Gagal mencatat log akhir ke DB: {e}")

        logger.info(
            f"✅ Pipeline selesai dengan sukses dalam {elapsed_time}s! "
            f"Total baris: {len(df_final):,}, Lonjakan={spike_counts.get(config.SPIKE_STATUS_LONJAKAN, 0):,}"
        )

        return {
            "success"               : True,
            "total_records"         : len(df_final),
            "df_result"             : df_final,
            "cluster_summary"       : clust_counts,
            "spike_summary"         : spike_counts,
            "recommendation_summary": rec_summary,
            "execution_time_sec"    : elapsed_time,
            "warnings"              : warnings,
            "errors"                : [],
        }

    except Exception as e:
        elapsed_time = round(time.time() - start_time, 2)
        err_msg = str(e)
        logger.error(f"❌ Pipeline gagal setelah {elapsed_time}s: {err_msg}", exc_info=True)
        errors.append(err_msg)

        try:
            database.log_process(
                process_name=f"batch_pipeline_{src_label}",
                status="FAILED",
                records_processed=0,
                message=f"Pipeline failed: {err_msg}",
                db_path=db_path,
            )
        except Exception:
            pass

        return {
            "success"               : False,
            "total_records"         : 0,
            "df_result"             : pd.DataFrame(),
            "cluster_summary"       : {},
            "spike_summary"         : {},
            "recommendation_summary": {},
            "execution_time_sec"    : elapsed_time,
            "warnings"              : warnings,
            "errors"                : errors,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("pipeline_runner.py loaded OK")


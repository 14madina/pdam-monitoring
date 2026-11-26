"""
src/prediction.py — Modul Prediksi Pemakaian Air PDAM (ANN Regression).

Modul ini mengintegrasikan model ANN Regression final hasil riset:
1. Memuat model ANN Global dan ANN Per Klaster dari folder models/ tanpa re-training.
2. Menggunakan skenario terpilih S1 (3 fitur: ['Ini', 'Lalu', 'tarif']) sebagai default.
3. Mendukung prediksi batch (seluruh pelanggan/file) dan prediksi individu (single customer).
4. Menyediakan fallback ke model global jika model klaster spesifik tidak tersedia.
5. Menjamin nilai prediksi non-negatif (pemakaian air fisik >= 0 m³).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("pdam_prediction")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(ch)

# Set KERAS_HOME ke data/interim/.keras untuk menghindari PermissionError di environment Windows
keras_dir = (config.DATA_INTERIM / ".keras").resolve()
keras_dir.mkdir(parents=True, exist_ok=True)
os.environ["KERAS_HOME"] = str(keras_dir)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Cache model agar tidak dibaca ulang dari disk setiap kali ada request
_MODEL_CACHE: Dict[str, Any] = {}


# =============================================================================
# MODEL LOADERS (LOAD TANPA TRAINING ULANG)
# =============================================================================

def load_global_ann_model(
    scenario: str = "selected_features",
) -> Tuple[Any, Any, List[str]]:
    """
    Muat model ANN Regression Global hasil riset tanpa pelatihan ulang.

    Parameters
    ----------
    scenario : str
        'selected_features' (S1 - 3 fitur) atau 'all_features' (S0 - 7 fitur).

    Returns
    -------
    (model, scaler, feature_list)
    """
    cache_key = f"ann_global_{scenario}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    import joblib
    import tensorflow as tf

    save_dir = config.MODEL_ANN_GLOBAL_SEL if scenario == "selected_features" else config.MODEL_ANN_GLOBAL_ALL
    model_path = save_dir / "ann_global.keras"
    scaler_path = save_dir / "scaler.pkl"
    features_path = save_dir / "features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model ANN Global tidak ditemukan di: {model_path}")

    logger.info(f"Loading ANN Global model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(features_path, "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    _MODEL_CACHE[cache_key] = (model, scaler, feature_list)
    logger.info(f"✅ ANN Global model ({scenario}) loaded successfully with features: {feature_list}")
    return model, scaler, feature_list


def load_cluster_ann_models(
    scenario: str = "selected_features",
    k: int = 3,
) -> Dict[int, Tuple[Any, Any, List[str]]]:
    """
    Muat seluruh model ANN Regression per-klaster (Cluster 0, Cluster 1, Cluster 2).

    Parameters
    ----------
    scenario : str
        'selected_features' (S1) atau 'all_features' (S0).
    k : int
        Jumlah klaster (default = 3).

    Returns
    -------
    Dict[cluster_id, (model, scaler, feature_list)]
    """
    cache_key = f"ann_clusters_{scenario}_{k}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    import joblib
    import tensorflow as tf

    save_dir = config.MODEL_ANN_CLUST_SEL if scenario == "selected_features" else config.MODEL_ANN_CLUST_ALL
    cluster_models = {}

    for c_id in range(k):
        model_path = save_dir / f"ann_global_cluster{c_id}.keras"
        scaler_path = save_dir / f"scaler_cluster{c_id}.pkl"
        features_path = save_dir / f"features_cluster{c_id}.json"

        if model_path.exists():
            logger.info(f"Loading ANN Cluster {c_id} model from {model_path} ...")
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            if features_path.exists():
                with open(features_path, "r", encoding="utf-8") as f:
                    feature_list = json.load(f)
            else:
                feature_list = config.SELECTED_FEATURES if scenario == "selected_features" else config.ALL_FEATURES

            cluster_models[c_id] = (model, scaler, feature_list)
            logger.info(f"✅ Cluster {c_id} ANN model loaded.")
        else:
            logger.warning(f"File model klaster {c_id} tidak ditemukan di {model_path}.")

    _MODEL_CACHE[cache_key] = cluster_models
    return cluster_models


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_water_usage(
    df: pd.DataFrame,
    mode: str = "cluster_specific",
    scenario: str = "selected_features",
) -> pd.DataFrame:
    """
    Eksekusi prediksi pemakaian air (m³) untuk sekumpulan data pelanggan.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang telah melalui tahap preprocessing.
        Harus memiliki fitur: ['Ini', 'Lalu', 'tarif'].
    mode : str
        'cluster_specific' (gunakan model khusus tiap klaster) atau 'global' (gunakan satu model global).
    scenario : str
        'selected_features' (default S1) atau 'all_features' (S0).

    Returns
    -------
    pd.DataFrame dengan kolom tambahan:
        - cluster_id (jika belum ada)
        - cluster_label
        - cluster_color
        - usage_current (Ini - Lalu)
        - predicted_usage (hasil prediksi ANN untuk periode berikutnya, non-negatif)
        - diff_predicted_vs_current (predicted_usage - usage_current)
        - diff_pct_projected (%)
    """
    df_out = df.copy()

    # Pastikan cluster_id tersedia
    if "cluster_id" not in df_out.columns:
        from . import clustering
        logger.info("Kolom cluster_id belum tersedia, menjalankan clustering K-Means...")
        df_out = clustering.predict_cluster(df_out, scenario=scenario)

    # Pastikan usage_current tersedia (Ini - Lalu)
    if "usage_current" not in df_out.columns:
        if "Ini" in df_out.columns and "Lalu" in df_out.columns:
            df_out["usage_current"] = pd.to_numeric(df_out["Ini"], errors="coerce") - pd.to_numeric(df_out["Lalu"], errors="coerce")
        elif "usage_original" in df_out.columns:
            df_out["usage_current"] = df_out["usage_original"]
        else:
            df_out["usage_current"] = 0.0

    predictions = np.zeros(len(df_out), dtype=float)

    if mode == "cluster_specific":
        cluster_models = load_cluster_ann_models(scenario=scenario, k=3)
        global_model, global_scaler, global_features = load_global_ann_model(scenario=scenario)

        for cid in df_out["cluster_id"].unique():
            idx_mask = df_out["cluster_id"] == cid
            n_rows = idx_mask.sum()
            if n_rows == 0:
                continue

            sub_df = df_out.loc[idx_mask]

            if cid in cluster_models:
                c_model, c_scaler, c_features = cluster_models[cid]
            else:
                logger.warning(f"Model klaster {cid} tidak ditemukan, fallback ke model Global.")
                c_model, c_scaler, c_features = global_model, global_scaler, global_features

            # Ambil fitur sesuai urutan eksak
            X = sub_df[c_features].copy()
            for col in c_features:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

            X_scaled = c_scaler.transform(X) if c_scaler else X.values
            y_pred = c_model.predict(X_scaled, verbose=0).flatten()

            # Pastikan non-negatif
            y_pred = np.maximum(y_pred, 0.0)
            predictions[idx_mask.values] = y_pred

    else:
        # Mode Global
        global_model, global_scaler, global_features = load_global_ann_model(scenario=scenario)
        X = df_out[global_features].copy()
        for col in global_features:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

        X_scaled = global_scaler.transform(X) if global_scaler else X.values
        y_pred = global_model.predict(X_scaled, verbose=0).flatten()
        predictions = np.maximum(y_pred, 0.0)

    df_out["predicted_usage"] = np.round(predictions, 2)
    df_out["diff_predicted_vs_current"] = np.round(df_out["predicted_usage"] - df_out["usage_current"], 2)

    # Persentase perubahan proyeksi
    denom = np.maximum(np.abs(df_out["usage_current"]), 1.0)
    df_out["diff_pct_projected"] = np.round((df_out["diff_predicted_vs_current"] / denom) * 100.0, 2)

    logger.info(
        f"Prediksi ANN ({mode}, {scenario}) selesai untuk {len(df_out):,} baris. "
        f"Rata-rata prediksi: {df_out['predicted_usage'].mean():.2f} m³"
    )

    return df_out


def predict_single_customer(
    ini: float,
    lalu: float,
    tarif: float = 1.0,
    cluster_id: Optional[int] = None,
    mode: str = "cluster_specific",
    scenario: str = "selected_features",
) -> Dict[str, Any]:
    """
    Prediksi instan pemakaian air untuk satu pelanggan (cocok untuk form input web).

    Parameters
    ----------
    ini : float
        Angka meter bulan ini.
    lalu : float
        Angka meter bulan lalu.
    tarif : float, optional (default 1.0)
        Kode/kelompok tarif.
    cluster_id : Optional[int]
        Jika None, klaster akan diprediksi otomatis menggunakan K-Means.
    mode : str
        'cluster_specific' atau 'global'.
    scenario : str
        'selected_features' (default S1) atau 'all_features'.

    Returns
    -------
    Dict[str, Any] berisi:
        - ini, lalu, usage_current, tarif
        - cluster_id, cluster_label, cluster_color
        - predicted_next_usage (m³)
        - projected_change (m³)
        - projected_change_pct (%)
        - status_trend (NAIK, TURUN, STABIL)
    """
    ini_val = float(ini)
    lalu_val = float(lalu)
    tarif_val = float(tarif)
    usage_curr = max(0.0, ini_val - lalu_val)

    # 1. Tentukan klaster jika belum ditentukan
    from . import clustering
    if cluster_id is None:
        df_single = pd.DataFrame([{
            "Ini": ini_val,
            "Lalu": lalu_val,
            "tarif": tarif_val,
        }])
        df_clust = clustering.predict_cluster(df_single, scenario=scenario)
        cid = int(df_clust["cluster_id"].iloc[0])
        clabel = str(df_clust["cluster_label"].iloc[0])
        ccolor = str(df_clust["cluster_color"].iloc[0])
    else:
        cid = int(cluster_id)
        clabel = clustering.CLUSTER_METADATA.get(cid, {}).get("cluster_label", f"Klaster {cid+1}")
        ccolor = clustering.CLUSTER_METADATA.get(cid, {}).get("cluster_color", "#808080")

    # 2. Siapkan input fitur
    features = config.SELECTED_FEATURES if scenario == "selected_features" else config.ALL_FEATURES
    row_features = {
        "Ini": ini_val,
        "Lalu": lalu_val,
        "tarif": tarif_val,
        "lati": 0.0,
        "longi": 0.0,
        "alti": 0.0,
        "time": 0.0,
    }
    X_df = pd.DataFrame([[row_features[f] for f in features]], columns=features, dtype=float)

    # 3. Pilih model
    if mode == "cluster_specific":
        cluster_models = load_cluster_ann_models(scenario=scenario, k=3)
        if cid in cluster_models:
            model, scaler, _ = cluster_models[cid]
        else:
            model, scaler, _ = load_global_ann_model(scenario=scenario)
    else:
        model, scaler, _ = load_global_ann_model(scenario=scenario)

    X_scaled = scaler.transform(X_df) if scaler else X_df.values
    y_pred_val = float(model.predict(X_scaled, verbose=0).flatten()[0])
    y_pred_val = max(0.0, round(y_pred_val, 2))

    change = round(y_pred_val - usage_curr, 2)
    denom = max(usage_curr, 1.0)
    change_pct = round((change / denom) * 100.0, 2)

    if change_pct > 15.0:
        trend = "PROYEKSI MENINGKAT"
    elif change_pct < -15.0:
        trend = "PROYEKSI MENURUN"
    else:
        trend = "PROYEKSI STABIL"

    return {
        "Ini"                  : ini_val,
        "Lalu"                 : lalu_val,
        "usage_current"        : usage_curr,
        "tarif"                : tarif_val,
        "cluster_id"           : cid,
        "cluster_label"        : clabel,
        "cluster_color"        : ccolor,
        "predicted_next_usage" : y_pred_val,
        "projected_change"     : change,
        "projected_change_pct" : change_pct,
        "trend_summary"        : trend,
        "scenario"             : scenario,
        "mode"                 : mode,
    }


def clear_model_cache():
    """Mengosongkan cache model di memori."""
    _MODEL_CACHE.clear()
    logger.info("Cache model telah dikosongkan.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing prediction.py ...")
    res = predict_single_customer(ini=150, lalu=120, tarif=1.0)
    print("Single customer prediction test result:", res)

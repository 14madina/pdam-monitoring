"""
src/clustering.py — K-Means clustering untuk skenario S0 dan S1.

Rules:
- K-Means memakai latest training snapshot per NoPEL (satu baris per pelanggan).
- RobustScaler sebagai default scaler.
- Uji K=2..8 dan simpan semua metrik.
- random_state = 42 untuk reproducibility.
- Model disimpan dengan joblib.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# PREPARE CLUSTERING DATA
# =============================================================================

def prepare_clustering_data(
    df: pd.DataFrame,
    feature_list: List[str],
    train_months: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Siapkan data untuk K-Means: ambil snapshot terbaru per NoPEL
    dari training period.

    K-Means clustering menggunakan agregat per pelanggan,
    bukan data time-series longitudinal.

    Parameters
    ----------
    df : pd.DataFrame
        Master feature DataFrame.
    feature_list : List[str]
        Fitur yang akan digunakan.
    train_months : list
        Bulan-bulan training period.

    Returns
    -------
    (df_cluster_input, df_meta)
        df_cluster_input : features untuk clustering (satu baris per NoPEL)
        df_meta          : metadata (NoPEL, source_area, RAYON, dsb)
    """
    # Filter ke training period
    df_train = df[df["period_dt"].isin(train_months)].copy()

    # Ambil snapshot terbaru per NoPEL (baris terakhir dalam training period)
    df_latest = (
        df_train
        .sort_values(["NoPEL", "period_dt"])
        .groupby("NoPEL")
        .last()
        .reset_index()
    )

    # Validasi fitur tersedia
    missing_features = [f for f in feature_list if f not in df_latest.columns]
    if missing_features:
        logger.warning(f"Fitur tidak ditemukan di data: {missing_features}")
        feature_list = [f for f in feature_list if f in df_latest.columns]

    # Ambil fitur dan metadata
    df_cluster_input = df_latest[["NoPEL"] + feature_list].copy()

    meta_cols = ["NoPEL"] + [c for c in ["source_area", "RAYON", "latitude", "longitude"]
                              if c in df_latest.columns]
    df_meta = df_latest[meta_cols].copy()

    logger.info(
        f"Clustering input: {len(df_cluster_input):,} pelanggan, "
        f"{len(feature_list)} features"
    )

    return df_cluster_input, df_meta


# =============================================================================
# K SEARCH
# =============================================================================

def run_kmeans_k_search(
    df_cluster_input: pd.DataFrame,
    feature_list: List[str],
    k_range: List[int] = config.KMEANS_K_RANGE,
    random_state: int = config.RANDOM_STATE,
    scaler_type: str = config.SCALER_KMEANS,
) -> pd.DataFrame:
    """
    Uji K-Means untuk berbagai nilai K dan kumpulkan metrik.

    Parameters
    ----------
    df_cluster_input : pd.DataFrame
        Input clustering dengan kolom 'NoPEL' dan features.
    feature_list : List[str]
    k_range : List[int]
    random_state : int
    scaler_type : str

    Returns
    -------
    pd.DataFrame dengan kolom: K, inertia, silhouette, dbi, chi, cluster_sizes
    """
    X = df_cluster_input[feature_list].copy()

    # Handle NaN — impute dengan median kolom
    X = X.fillna(X.median())

    # Scaling
    if scaler_type == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler_type == "RobustScaler":
        p995 = np.percentile(X, 99.5, axis=0)
        X_clipped = np.clip(X, 0, p995)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_clipped)
    else:
        p995 = np.percentile(X, 99.5, axis=0)
        X_clipped = np.clip(X, 0, p995)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clipped)

    results = []
    for k in k_range:
        logger.info(f"  Trying K={k} ...")
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=500,
            random_state=random_state,
        )
        labels = km.fit_predict(X_scaled)

        inertia   = km.inertia_
        sil       = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)),
                                     random_state=random_state)
        dbi       = davies_bouldin_score(X_scaled, labels)
        chi       = calinski_harabasz_score(X_scaled, labels)

        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        min_size      = min(cluster_sizes.values())
        max_size      = max(cluster_sizes.values())
        balance_ratio = min_size / max_size  # ideal mendekati 1

        results.append({
            "K"            : k,
            "inertia"      : round(inertia, 4),
            "silhouette"   : round(sil,     6),
            "dbi"          : round(dbi,     6),
            "chi"          : round(chi,     4),
            "min_cluster"  : min_size,
            "max_cluster"  : max_size,
            "balance_ratio": round(balance_ratio, 4),
            "cluster_sizes": str(cluster_sizes),
        })
        logger.info(
            f"    K={k}: inertia={inertia:.1f}, sil={sil:.4f}, "
            f"dbi={dbi:.4f}, chi={chi:.1f}"
        )

    df_results = pd.DataFrame(results)
    logger.info(f"✅ K-Search complete for {len(k_range)} K values.")
    return df_results


# =============================================================================
# FIT FINAL K-MEANS
# =============================================================================

def fit_kmeans(
    df_cluster_input: pd.DataFrame,
    feature_list: List[str],
    k: int,
    random_state: int = config.RANDOM_STATE,
    scaler_type: str = config.SCALER_KMEANS,
) -> Tuple[KMeans, object, pd.DataFrame]:
    """
    Fit K-Means final dengan K terpilih.

    Returns
    -------
    (kmeans_model, scaler, df_with_cluster_id)
    """
    X = df_cluster_input[feature_list].copy().fillna(df_cluster_input[feature_list].median())

    if scaler_type == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler_type == "RobustScaler":
        p995 = np.percentile(X, 99.5, axis=0)
        X_clipped = np.clip(X, 0, p995)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_clipped)
    else:
        p995 = np.percentile(X, 99.5, axis=0)
        X_clipped = np.clip(X, 0, p995)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clipped)

    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=random_state,
    )
    raw_labels = km.fit_predict(X_scaled)

    # Urutkan cluster agar konsisten: 0 (RENDAH), 1 (SEDANG), 2 (TINGGI)
    # berdasarkan rata-rata fitur pemakaian/meter index
    ref_col = 'Ini' if 'Ini' in X.columns else feature_list[0]
    mean_per_cluster = pd.Series(X[ref_col]).groupby(raw_labels).mean()
    sorted_clusters = mean_per_cluster.sort_values().index.tolist()
    label_mapping = {old: new for new, old in enumerate(sorted_clusters)}
    ordered_labels = np.array([label_mapping[l] for l in raw_labels])

    tier_names = {0: "RENDAH", 1: "SEDANG", 2: "TINGGI", 3: "SANGAT_TINGGI"}

    df_result = df_cluster_input.copy()
    df_result["cluster_id"] = ordered_labels
    df_result["cluster_name"] = [tier_names.get(cid, 'CLUSTER_' + str(cid)) for cid in ordered_labels]
    df_result["cluster_label"] = [f"{cid} ({tier_names.get(cid, 'CLUSTER_' + str(cid))})" for cid in ordered_labels]

    cluster_dist = pd.Series(ordered_labels).value_counts().sort_index()
    logger.info(f"✅ K-Means final fit: K={k}, ordered cluster distribution:\n{cluster_dist.to_dict()}")

    return km, scaler, df_result


# =============================================================================
# CLUSTER PROFILES
# =============================================================================

def get_cluster_profiles(
    df_with_clusters: pd.DataFrame,
    feature_list: List[str],
) -> pd.DataFrame:
    """
    Analisis profil centroid per cluster.

    Returns
    -------
    pd.DataFrame dengan statistik per cluster per fitur.
    """
    agg_funcs = {f: ["mean", "median", "std", "min", "max"] for f in feature_list}
    profile = (
        df_with_clusters.groupby("cluster_id")
        .agg({**{f: ["mean", "median", "std"] for f in feature_list if f in df_with_clusters.columns},
              "NoPEL": "count"})
    )
    profile.columns = ["_".join(c).strip("_") for c in profile.columns]
    profile = profile.rename(columns={"NoPEL_count": "n_members"})
    profile = profile.reset_index()
    return profile


# =============================================================================
# SAVE / LOAD MODEL
# =============================================================================

def save_kmeans_model(
    km: KMeans,
    scaler: object,
    feature_list: List[str],
    save_dir: Path,
    k: int,
    scenario: str,
) -> None:
    """Simpan K-Means model, scaler, dan feature list."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(km,     save_dir / f"kmeans_k{k}.pkl")
    joblib.dump(scaler, save_dir / f"scaler_k{k}.pkl")

    with open(save_dir / f"features_k{k}.json", "w") as f:
        json.dump(feature_list, f, indent=2)

    meta = {
        "K"        : k,
        "scenario" : scenario,
        "n_features": len(feature_list),
        "random_state": config.RANDOM_STATE,
        "scaler"   : type(scaler).__name__,
    }
    with open(save_dir / f"meta_k{k}.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"K-Means model saved to {save_dir} (K={k}, scenario={scenario})")


def load_kmeans_model(
    save_dir: Path,
    k: int,
) -> Tuple[KMeans, object, List[str]]:
    """Load K-Means model, scaler, dan feature list."""
    import joblib
    save_dir = Path(save_dir)

    km     = joblib.load(save_dir / f"kmeans_k{k}.pkl")
    scaler = joblib.load(save_dir / f"scaler_k{k}.pkl")

    with open(save_dir / f"features_k{k}.json") as f:
        feature_list = json.load(f)

    logger.info(f"K-Means model loaded from {save_dir} (K={k})")
    return km, scaler, feature_list


def plot_kmeans_evaluation_linecharts(df_results: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Visualisasi evaluasi K-Means menggunakan GRAFIK GARIS SAJA (Line Charts Only):
    - Subplot 1: Silhouette Score vs K (Grafik Garis)
    - Subplot 2: Davies-Bouldin Index (DBI) vs K (Grafik Garis)
    - Subplot 3: Inertia (Elbow Method) vs K (Grafik Garis)
    - Subplot 4: Calinski-Harabasz Index (CHI) vs K (Grafik Garis)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Silhouette Score (Line Plot — Biru Tirta Perwira)
    axes[0, 0].plot(df_results["K"], df_results["silhouette"], marker="o", color="#0284C7", linewidth=2.5, markersize=8)
    axes[0, 0].axhline(0.70, color="#16A34A", linestyle="--", alpha=0.8, label="Ambang Batas Kuat > 0.70")
    axes[0, 0].set_title("A. Silhouette Score vs K (Tinggi Lebih Baik)", fontsize=11.5, fontweight="bold", color="#0369A1", pad=10)
    axes[0, 0].set_xlabel("Jumlah Klaster (K)", fontsize=10)
    axes[0, 0].set_ylabel("Silhouette Score", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend(fontsize=9)

    # 2. Davies-Bouldin Index (Line Plot — Merah Aksen Tirta Perwira)
    axes[0, 1].plot(df_results["K"], df_results["dbi"], marker="s", color="#DC2626", linewidth=2.5, markersize=8)
    axes[0, 1].set_title("B. Davies-Bouldin Index vs K (Rendah Lebih Baik)", fontsize=11.5, fontweight="bold", color="#0369A1", pad=10)
    axes[0, 1].set_xlabel("Jumlah Klaster (K)", fontsize=10)
    axes[0, 1].set_ylabel("Davies-Bouldin Index", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # 3. Inertia / Elbow (Line Plot — Hijau Gunung Slamet)
    axes[1, 0].plot(df_results["K"], df_results["inertia"], marker="^", color="#16A34A", linewidth=2.5, markersize=8)
    axes[1, 0].set_title("C. Inersia / Elbow Method vs K", fontsize=11.5, fontweight="bold", color="#0369A1", pad=10)
    axes[1, 0].set_xlabel("Jumlah Klaster (K)", fontsize=10)
    axes[1, 0].set_ylabel("Inersia (WCSS)", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # 4. Calinski-Harabasz Index (Line Plot — Biru Dongker Gelombang Logo)
    axes[1, 1].plot(df_results["K"], df_results["chi"], marker="D", color="#0F4C81", linewidth=2.5, markersize=8)
    axes[1, 1].set_title("D. Calinski-Harabasz Index vs K (Tinggi Lebih Baik)", fontsize=11.5, fontweight="bold", color="#0369A1", pad=10)
    axes[1, 1].set_xlabel("Jumlah Klaster (K)", fontsize=10)
    axes[1, 1].set_ylabel("Calinski-Harabasz Score", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Evaluasi Penentuan Nilai Klaster Optimal K-Means (K = 2 s.d 8)", fontsize=13.5, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"K-Means line charts saved to {save_path}")

    return fig


# =============================================================================
# WEB DASHBOARD INTEGRATION: CLUSTER PREDICTION & METADATA
# =============================================================================

CLUSTER_METADATA = {
    0: {
        "cluster_name" : "Klaster 1",
        "cluster_label": "Klaster 1 (Rendah - ABRI/Khusus)",
        "cluster_color": "#16A34A",  # Hijau Gunung Slamet (Pemakaian Rendah/Sosial)
        "keterangan"   : "Kelompok pelanggan dengan volume pemakaian air rendah atau tarif khusus/sosial.",
    },
    1: {
        "cluster_name" : "Klaster 2",
        "cluster_label": "Klaster 2 (Sedang - Rumah Tangga)",
        "cluster_color": "#0284C7",  # Biru Tirta Perwira (Pemakaian Sedang/Rumah Tangga)
        "keterangan"   : "Kelompok pelanggan rumah tangga reguler dengan volume pemakaian stabil menengah.",
    },
    2: {
        "cluster_name" : "Klaster 3",
        "cluster_label": "Klaster 3 (Tinggi - Niaga/Industri)",
        "cluster_color": "#0F4C81",  # Biru Gelombang Dongker (Pemakaian Tinggi/Niaga/Industri)
        "keterangan"   : "Kelompok pelanggan komersial, niaga, dan industri dengan intensitas pemakaian tinggi.",
    },
}


def load_final_kmeans_model(
    k: int = 3,
    scenario: str = "selected_features",
) -> Tuple[KMeans, object, List[str]]:
    """
    Muat model K-Means final hasil riset tanpa pelatihan ulang.

    Parameters
    ----------
    k : int
        Jumlah klaster (default = 3).
    scenario : str
        'selected_features' (S1 - 3 fitur) atau 'all_features' (S0 - 7 fitur).

    Returns
    -------
    (kmeans_model, scaler, feature_list)
    """
    if scenario == "selected_features":
        model_dir = config.MODEL_KMEANS_SEL
    else:
        model_dir = config.MODEL_KMEANS_ALL

    return load_kmeans_model(save_dir=model_dir, k=k)


def predict_cluster(
    df: pd.DataFrame,
    km_model: Optional[KMeans] = None,
    scaler: Optional[object] = None,
    feature_list: Optional[List[str]] = None,
    scenario: str = "selected_features",
) -> pd.DataFrame:
    """
    Prediksi klaster pelanggan untuk DataFrame baru menggunakan model K-Means hasil riset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang minimal memiliki kolom fitur yang sesuai (misal: Ini, Lalu, tarif).
    km_model : KMeans, optional
    scaler : object, optional
    feature_list : List[str], optional
    scenario : str, optional

    Returns
    -------
    pd.DataFrame dengan kolom tambahan:
        cluster_id (int: 0, 1, 2)
        cluster_label (str: nama klaster deskriptif)
        cluster_color (str: hex color code untuk visualisasi)
    """
    df_out = df.copy()

    # Load model jika belum disediakan
    if km_model is None or scaler is None or feature_list is None:
        km_model, scaler, feature_list = load_final_kmeans_model(k=3, scenario=scenario)

    # Validasi keberadaan fitur
    missing = [f for f in feature_list if f not in df_out.columns]
    if missing:
        raise ValueError(f"Fitur untuk K-Means tidak lengkap: {missing}. Memerlukan: {feature_list}")

    # Ekstrak matriks fitur sesuai urutan eksak saat training
    X = df_out[feature_list].copy()
    for col in feature_list:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # Scaling menggunakan scaler training tersimpan
    X_scaled = scaler.transform(X)

    # Prediksi klaster
    cluster_preds = km_model.predict(X_scaled)
    df_out["cluster_id"] = cluster_preds.astype(int)

    # Tambahkan label deskriptif dan kode warna
    df_out["cluster_label"] = df_out["cluster_id"].map(
        lambda cid: CLUSTER_METADATA.get(cid, {}).get("cluster_label", f"Klaster {cid+1}")
    )
    df_out["cluster_color"] = df_out["cluster_id"].map(
        lambda cid: CLUSTER_METADATA.get(cid, {}).get("cluster_color", "#808080")
    )

    logger.info(
        f"K-Means prediction complete: {len(df_out):,} baris diklasifikasikan ke {km_model.n_clusters} klaster."
    )
    return df_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("clustering.py loaded OK")


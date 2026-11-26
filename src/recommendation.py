"""
src/recommendation.py — Modul Rekomendasi Operasional & Surat Perintah Kerja (SPK) PDAM.

Menghasilkan rekomendasi tindakan bisnis dan teknis yang dapat ditindaklanjuti
berdasarkan hasil deteksi lonjakan/anomali dan pengelompokan profil pelanggan.
"""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger("pdam_recommendation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(ch)


# =============================================================================
# REKOMENDASI RULES ENGINE
# =============================================================================

ACTION_RULES = {
    "LONJAKAN_EKSTREM": {
        "priority"       : "TINGGI",
        "action_title"   : "Penerbitan SPK Investigasi Kebocoran / Tera Meter",
        "action_detail"  : (
            "Konsumsi air melonjak drastis (>100% atau >50 m³ di atas prediksi model). "
            "Segera terbitkan SPK untuk inspeksi fisik ke lokasi: "
            "1. Cek kebocoran pipa persil di belakang meter pelanggan. "
            "2. Periksa kemungkinan kerusakan/loncatan roda gigi meter air. "
            "3. Lakukan konfirmasi pola pemakaian kepada pemilik persil."
        ),
        "target_division": "Bagian Transmisi & Distribusi / Bagian Hubungan Langganan",
        "sla_days"       : 3,
    },
    "LONJAKAN_MODERAT": {
        "priority"       : "SEDANG",
        "action_title"   : "Verifikasi Catat Meter & Konfirmasi Pemakaian",
        "action_detail"  : (
            "Konsumsi air berada 30% s.d 100% di atas prediksi normal. "
            "Lakukan validasi data foto angka stand meter oleh pengawas cater. "
            "Jika foto sesuai, konfirmasi kemungkinan penambahan penghuni atau aktivitas renovasi."
        ),
        "target_division": "Bagian Pembaca Meter (Cater) & Customer Service",
        "sla_days"       : 7,
    },
    "METER_MACET_ATAU_KOSONG": {
        "priority"       : "TINGGI",
        "action_title"   : "Inspeksi Meter Macet / Verifikasi Rumah Kosong",
        "action_detail"  : (
            "Konsumsi air mendekati 0 m³ dengan penurunan ekstrem (>80% di bawah prediksi). "
            "1. Periksa apakah jarum register meter macet saat keran dibuka. "
            "2. Pastikan apakah persil dalam kondisi kosong/tidak berpenghuni. "
            "3. Jika meter macet, jadwalkan penggantian meter air baru."
        ),
        "target_division": "Bagian Pemeliharaan Meter / Penggantian Meter",
        "sla_days"       : 5,
    },
    "PENURUNAN_SIGNIFIKAN": {
        "priority"       : "RENDAH",
        "action_title"   : "Monitoring Administratif & Cek Sumber Air Alternatif",
        "action_detail"  : (
            "Konsumsi air turun signifikan (30% s.d 80% di bawah prediksi normal). "
            "Lakukan pemantauan pada siklus bulan berikutnya dan periksa apakah pelanggan "
            "menggunakan sumber air alternatif (sumur bor/pompa)."
        ),
        "target_division": "Bagian Administrasi Langganan",
        "sla_days"       : 14,
    },
    "NORMAL": {
        "priority"       : "NORMAL",
        "action_title"   : "Pembacaan Normal Terverifikasi",
        "action_detail"  : "Pola pemakaian air stabil dan konsisten dengan prediksi model. Tagihan dapat diterbitkan langsung.",
        "target_division": "Bagian Penagihan & Keuangan",
        "sla_days"       : 0,
    },
}


def generate_recommendations(df_spikes: pd.DataFrame) -> pd.DataFrame:
    """
    Hasilkan rekomendasi operasional terperinci untuk seluruh baris data pelanggan.

    Parameters
    ----------
    df_spikes : pd.DataFrame
        DataFrame yang telah diproses oleh detect_spikes_from_predictions.
        Wajib memiliki kolom: 'anomaly_category', 'pct_diff', 'residual'.

    Returns
    -------
    pd.DataFrame dengan kolom tambahan rekomendasi:
        - recommendation_priority ('TINGGI', 'SEDANG', 'RENDAH', 'NORMAL')
        - recommended_action
        - action_details
        - target_division
        - sla_days
    """
    df_out = df_spikes.copy()

    priorities = []
    actions = []
    details = []
    divisions = []
    slas = []

    for _, row in df_out.iterrows():
        cat = row.get("anomaly_category", "NORMAL")
        rule = ACTION_RULES.get(cat, ACTION_RULES["NORMAL"])

        priorities.append(rule["priority"])
        actions.append(rule["action_title"])
        details.append(rule["action_detail"])
        divisions.append(rule["target_division"])
        slas.append(rule["sla_days"])

    df_out["recommendation_priority"] = priorities
    df_out["recommended_action"] = actions
    df_out["action_details"] = details
    df_out["target_division"] = divisions
    df_out["sla_days"] = slas

    n_tinggi = (df_out["recommendation_priority"] == "TINGGI").sum()
    n_sedang = (df_out["recommendation_priority"] == "SEDANG").sum()
    logger.info(
        f"Generated recommendations: {len(df_out):,} pelanggan. "
        f"Prioritas TINGGI={n_tinggi:,}, SEDANG={n_sedang:,}"
    )

    return df_out


def get_recommendation_summary(df_recs: pd.DataFrame) -> Dict[str, Any]:
    """Ringkasan statistik rekomendasi tindakan untuk kartu dashboard."""
    if df_recs.empty or "recommendation_priority" not in df_recs.columns:
        return {}

    n_total = len(df_recs)
    p_counts = df_recs["recommendation_priority"].value_counts().to_dict()

    return {
        "total_pelanggan"   : n_total,
        "spk_urgent_count"  : int(p_counts.get("TINGGI", 0)),
        "verifikasi_count"  : int(p_counts.get("SEDANG", 0)),
        "monitoring_count"  : int(p_counts.get("RENDAH", 0)),
        "normal_count"      : int(p_counts.get("NORMAL", 0)),
        "pct_butuh_tindakan": round((p_counts.get("TINGGI", 0) + p_counts.get("SEDANG", 0)) / n_total * 100, 2),
    }


def export_spk_work_orders(
    df_recs: pd.DataFrame,
    priority_filter: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Ekspor daftar Surat Perintah Kerja (SPK) untuk investigasi tim lapangan PDAM.

    Parameters
    ----------
    df_recs : pd.DataFrame
        DataFrame hasil generate_recommendations.
    priority_filter : Optional[List[str]]
        Filter prioritas, default ['TINGGI', 'SEDANG'].
    output_path : Optional[Union[str, Path]]
        Jika diberikan, simpan ke file Excel (.xlsx).

    Returns
    -------
    pd.DataFrame format format siap cetak/unduh SPK.
    """
    if priority_filter is None:
        priority_filter = ["TINGGI", "SEDANG"]

    mask = df_recs["recommendation_priority"].isin(priority_filter)
    spk_df = df_recs[mask].copy().reset_index(drop=True)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d")
    year_prefix = datetime.datetime.now().strftime("%Y%m")

    # Format kolom SPK
    spk_export = pd.DataFrame()
    spk_export["No_SPK"] = [f"SPK-PDAM/{year_prefix}/{i+1:05d}" for i in range(len(spk_df))]
    spk_export["Tanggal_Terbit"] = now_str
    spk_export["Prioritas"] = spk_df.get("recommendation_priority", "SEDANG")
    spk_export["NoPEL"] = spk_df["NoPEL"].astype(str).str.zfill(8)
    spk_export["Nama"] = spk_df.get("Nama", "-")
    spk_export["Alamat"] = spk_df.get("Alamat", "-")
    spk_export["RAYON"] = spk_df.get("RAYON", "-")
    spk_export["Klaster"] = spk_df.get("cluster_label", "-")
    spk_export["Pemakaian_Aktual_m3"] = spk_df.get("actual_usage", spk_df.get("actual", spk_df.get("usage_current", 0.0)))
    spk_export["Prediksi_Model_m3"] = spk_df.get("predicted_usage", spk_df.get("prediction", 0.0))
    spk_export["Selisih_m3"] = spk_df.get("residual", spk_df.get("error_abs", 0.0))
    spk_export["Persentase_Diff_%"] = spk_df.get("pct_diff", 0.0)
    spk_export["Kategori_Anomali"] = spk_df.get("anomaly_category", spk_df.get("spike_status", "-"))
    spk_export["Instruksi_Tindakan"] = spk_df.get("recommended_action", spk_df.get("recommendation_text", "-"))
    spk_export["Detail_Pekerjaan"] = spk_df.get("action_details", spk_df.get("action_type", "-"))
    spk_export["Divisi_Tujuan"] = spk_df.get("target_division", "Bagian Distribusi / Pelayanan Langganan")
    spk_export["Batas_Waktu_Hari"] = spk_df.get("sla_days", 3)
    spk_export["Petugas_Pelaksana"] = ""
    spk_export["Hasil_Pemeriksaan_Lapangan"] = ""
    spk_export["Status_SPK"] = "OPEN"

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        spk_export.to_excel(out_p, index=False, engine="openpyxl")
        logger.info(f"SPK work orders exported to Excel: {out_p} ({len(spk_export)} items)")

    return spk_export


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("recommendation.py loaded OK")


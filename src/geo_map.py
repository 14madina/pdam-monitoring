"""
src/geo_map.py — Modul Peta Geospasial Interaktif PDAM Purbalingga.

Menghasilkan peta berlapis berwarna (Google Hybrid Satellite, OpenStreetMap, Esri World Imagery, OpenTopoMap)
tanpa watermark API KEY, dengan MarkerCluster canggih, kompas mata angin (Compass Rose),
dispersi titik bertumpuk (golden spiral) agar seluruh titik dapat diklik,
dan kartu pop-up interaktif lengkap dengan grafik sparkline SVG riwayat 12 periode konsumsi air (Aktual vs Prediksi ANN vs Ambang +30%).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import folium
from folium.plugins import Fullscreen, MarkerCluster
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# Konfigurasi Warna & Label Klaster K-Means (Harmonisasi Palet Logo Tirta Perwira)
CLUSTER_COLORS = {
    0: "#16A34A",  # Klaster 1: Hijau Gunung Slamet (Rendah - ABRI/Khusus)
    1: "#0284C7",  # Klaster 2: Biru Tirta Perwira (Sedang - Rumah Tangga)
    2: "#0F4C81",  # Klaster 3: Biru Gelombang Dongker (Tinggi - Niaga/Industri)
}

CLUSTER_LABELS = {
    0: "Klaster 1 (Rendah - ABRI/Khusus)",
    1: "Klaster 2 (Sedang - Rumah Tangga)",
    2: "Klaster 3 (Tinggi - Niaga/Industri)",
}

# Konfigurasi Warna & Label Status Lonjakan (Harmonisasi Logo Tirta Perwira)
SPIKE_COLORS = {
    "INDIKASI_LONJAKAN": "#DC2626",  # Merah Perwira (Kenaikan > +30%)
    "ANOMALI_RENDAH"   : "#D97706",  # Amber / Titik Emas (Penurunan < -30% / Potensi Meter Macet)
    "NORMAL"           : "#16A34A",  # Hijau Alami (Konsumsi Wajar -30% s/d +30%)
}

SPIKE_LABELS = {
    "INDIKASI_LONJAKAN": "Indikasi Lonjakan (> +30%)",
    "ANOMALI_RENDAH"   : "Anomali Rendah (< -30%)",
    "NORMAL"           : "Konsumsi Normal (Wajar)",
}


def disperse_duplicate_coordinates(
    df_geo: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    radius_step: float = 0.000045,
) -> pd.DataFrame:
    """
    Terapkan dispersi mikro spiral emas (golden spiral) deterministik pada koordinat duplikat.
    Memisahkan titik bertumpuk sejauh 4 s/d 35 meter agar setiap marker dapat diklik satu per satu.
    """
    df_out = df_geo.copy()
    if df_out.empty:
        return df_out

    rounded_lat = df_out[lat_col].round(6)
    rounded_lon = df_out[lon_col].round(6)
    df_out["_coord_group"] = list(zip(rounded_lat, rounded_lon))
    df_out["_dup_idx"] = df_out.groupby("_coord_group").cumcount()
    golden_angle = 2.399963229728653
    mask = df_out["_dup_idx"] > 0
    if mask.any():
        indices = df_out.loc[mask, "_dup_idx"].values
        r = radius_step * np.sqrt(indices)
        theta = indices * golden_angle
        lat_rad = np.radians(df_out.loc[mask, lat_col].values)
        cos_lat = np.cos(lat_rad)
        cos_lat = np.where(np.abs(cos_lat) < 1e-5, 1.0, cos_lat)
        df_out.loc[mask, lat_col] += r * np.cos(theta)
        df_out.loc[mask, lon_col] += (r * np.sin(theta)) / cos_lat
    df_out.drop(columns=["_coord_group", "_dup_idx"], inplace=True)
    return df_out


def generate_svg_chart(
    history_vals: Any,
    period_labels: Any = None,
    pred_val: Optional[float] = None,
    actual_val: Optional[float] = None,
) -> str:
    """
    Generate 12-period SVG sparkline chart for embedding inside marker popup cards.
    Format:
    - Garis Biru padat + titik biru: Pemakaian Aktual historis bulanan
    - Garis Oranye putus-putus + segitiga: Prediksi Model ANN
    - Garis Kuning putus-putus: Ambang batas lonjakan (+30% dari rata-rata)
    - Label periode bulan pada sumbu X
    """
    # Parsing history_vals
    if isinstance(history_vals, str):
        try:
            history_vals = json.loads(history_vals)
        except Exception:
            history_vals = []
    elif isinstance(history_vals, np.ndarray):
        history_vals = history_vals.tolist()
    elif not isinstance(history_vals, list):
        history_vals = []

    # Parsing period_labels
    if isinstance(period_labels, str):
        try:
            period_labels = json.loads(period_labels)
        except Exception:
            period_labels = []
    elif isinstance(period_labels, np.ndarray):
        period_labels = period_labels.tolist()
    elif not isinstance(period_labels, list):
        period_labels = []

    if not history_vals or len(history_vals) < 2:
        return (
            "<div style='color:#757575; font-size:10.5px; padding:6px; background:#F8FAFC; "
            "border:1px dashed #CBD5E1; border-radius:4px; text-align:center; font-style:italic; margin-top:5px;'>"
            "Riwayat 12 periode konsumsi air tidak tersedia</div>"
        )

    vals = []
    for v in history_vals[-12:]:
        try:
            val_f = float(v)
            vals.append(0.0 if np.isnan(val_f) else val_f)
        except Exception:
            vals.append(0.0)

    if actual_val is not None and not np.isnan(actual_val) and actual_val > 0:
        vals[-1] = float(actual_val)

    p_val = None
    if pred_val is not None:
        try:
            pred_f = float(pred_val)
            if not np.isnan(pred_f) and pred_f > 0:
                p_val = pred_f
        except Exception:
            p_val = None

    all_scale_vals = vals.copy()
    if p_val is not None:
        all_scale_vals.append(p_val)

    max_val = max(max(all_scale_vals), 1.0)
    avg_val = float(np.mean(vals)) if vals else 1.0
    thresh_val = avg_val * 1.3
    max_scale = max(max_val, thresh_val) * 1.18

    w, h = 280, 108
    pad_left, pad_right, pad_top, pad_bottom = 24, 12, 22, 20

    n = len(vals)
    pts = []
    th_pts = []

    for i, val in enumerate(vals):
        x = pad_left + (i / max(1, n - 1)) * (w - pad_left - pad_right)
        y = pad_top + (1 - val / max_scale) * (h - pad_top - pad_bottom)
        pts.append((x, y))

        y_th = pad_top + (1 - thresh_val / max_scale) * (h - pad_top - pad_bottom)
        th_pts.append((x, y_th))

    pts_str = " ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in pts])
    th_pts_str = " ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in th_pts])

    # Titik data aktual (Biru Tirta Perwira)
    dots_blue = "".join([f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="2.8" fill="#0284C7"/>' for p in pts])

    # Marker proyeksi prediksi ANN di periode terakhir (Hijau Slamet)
    pred_elements = ""
    if p_val is not None and len(pts) >= 2:
        x_last = pts[-1][0]
        y_pred = pad_top + (1 - p_val / max_scale) * (h - pad_top - pad_bottom)
        x_prev = pts[-2][0]
        y_prev = pts[-2][1]

        pred_line = f'<line x1="{x_prev:.1f}" y1="{y_prev:.1f}" x2="{x_last:.1f}" y2="{y_pred:.1f}" stroke="#16A34A" stroke-width="2" stroke-dasharray="3,3"/>'
        pred_marker = f'<polygon points="{x_last:.1f},{y_pred-4:.1f} {x_last-4:.1f},{y_pred+4:.1f} {x_last+4:.1f},{y_pred+4:.1f}" fill="#16A34A"/>'
        pred_elements = f"{pred_line} {pred_marker}"

    # Label sumbu X periode (awal, tengah, akhir)
    x_labels = ""
    if period_labels and len(period_labels) >= 2:
        plabels = period_labels[-n:]
        indices = [0, len(plabels) // 2, len(plabels) - 1]
        for idx in sorted(list(set(indices))):
            if idx < len(plabels):
                x = pad_left + (idx / max(1, n - 1)) * (w - pad_left - pad_right)
                align = "start" if idx == 0 else ("end" if idx == len(plabels) - 1 else "middle")
                x_labels += f'<text x="{x}" y="{h-4}" font-size="7.5" fill="#64748B" font-family="Arial,sans-serif" text-anchor="{align}">{plabels[idx]}</text>'

    legend_header = f"""
    <text x="{pad_left}" y="12" font-size="8" fill="#0284C7" font-weight="bold" font-family="Arial,sans-serif">● Aktual</text>
    <text x="{pad_left + 70}" y="12" font-size="8" fill="#16A34A" font-weight="bold" font-family="Arial,sans-serif">▲ Prediksi ANN</text>
    <text x="{pad_left + 165}" y="12" font-size="8" fill="#D97706" font-family="Arial,sans-serif">-- Batas +30%</text>
    """

    svg = f"""
    <svg width="{w}" height="{h}" style="background:#ffffff; border:1px solid #E2E8F0; border-radius:6px; margin-top:5px; box-shadow:inset 0 1px 2px rgba(0,0,0,0.03);">
        {legend_header}
        <line x1="{pad_left}" y1="{pad_top}" x2="{w-pad_right}" y2="{pad_top}" stroke="#F1F5F9" stroke-width="1"/>
        <line x1="{pad_left}" y1="{h-pad_bottom}" x2="{w-pad_right}" y2="{h-pad_bottom}" stroke="#E2E8F0" stroke-width="1"/>
        <polyline fill="none" stroke="#D97706" stroke-width="1.3" stroke-dasharray="3,2" points="{th_pts_str}" />
        <polyline fill="none" stroke="#0284C7" stroke-width="2.2" stroke-linecap="round" points="{pts_str}" />
        {dots_blue}
        {pred_elements}
        {x_labels}
    </svg>
    """
    return svg


def build_pdam_interactive_map(
    df_customers: pd.DataFrame,
    center: Optional[List[float]] = None,
    zoom_start: int = 13,
    color_by: str = "spike_status",  # "spike_status" atau "cluster"
) -> folium.Map:
    """
    Bangun peta geospasial interaktif PDAM dengan:
    1. Multi-layer base maps berwarna (Google Hybrid Satellite, OpenStreetMap, Esri World Imagery, OpenTopoMap) bebas watermark API KEY.
    2. Marker cluster dengan declustering otomatis saat diperbesar.
    3. Dispersi koordinat bertumpuk (golden spiral) agar semua titik dapat diklik.
    4. Pop-up informatif lengkap dengan grafik 12 periode konsumsi air (SVG sparkline).
    5. Kompas mata angin (Compass Rose N/S/E/W).
    6. Kontrol layer dan fullscreen.
    """
    valid_geo = df_customers[
        df_customers["latitude"].notna() &
        df_customers["longitude"].notna() &
        (df_customers["latitude"] != 0) &
        (df_customers["longitude"] != 0)
    ].copy()

    if valid_geo.empty:
        m = folium.Map(location=[-7.388, 109.363], zoom_start=13, tiles=None)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            attr="Google Hybrid Satellite",
            name="Google Hybrid Satellite (Satelit & Jalan)",
        ).add_to(m)
        folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap (Peta Jalan Berwarna)").add_to(m)
        folium.LayerControl().add_to(m)
        return m

    # Terapkan dispersi mikro golden spiral untuk koordinat bertumpuk
    valid_geo = disperse_duplicate_coordinates(valid_geo)

    if center is None:
        c_lat = float(valid_geo["latitude"].median())
        c_lon = float(valid_geo["longitude"].median())
    else:
        c_lat, c_lon = center

    pdam_map = folium.Map(
        location=[c_lat, c_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    # ===============================================================
    # 1. BASEMAPS BERLAPIS BERWARNA (BEBAS WATERMARK API KEY)
    # ===============================================================
    # Layer 1: Google Hybrid Satellite (Citra Satelit Berwarna Asli + Jalan & Label)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid Satellite",
        name="Google Hybrid Satellite (Satelit & Jalan)",
        overlay=False,
        control=True,
    ).add_to(pdam_map)

    # Layer 2: OpenStreetMap (Peta Jalan Standar Berwarna)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Peta Jalan Berwarna)",
        overlay=False,
        control=True,
    ).add_to(pdam_map)

    # Layer 3: Esri World Imagery (Citra Satelit Resolusi Tinggi)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite (Citra Satelit Resolusi Tinggi)",
        overlay=False,
        control=True,
    ).add_to(pdam_map)

    # Layer 4: OpenTopoMap (Peta Topografi Kontur Berwarna)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="OpenTopoMap (Topografi Berwarna)",
        overlay=False,
        control=True,
    ).add_to(pdam_map)

    # ===============================================================
    # 2. KOMPAS MATA ANGIN & LOGO BADGE PDAM
    # ===============================================================
    compass_html = """
    <div style="position: fixed; 
         top: 80px; left: 16px; width: 52px; height: 52px; 
         z-index: 999; background: rgba(255,255,255,0.95); border: 2px solid #1E293B; border-radius: 50%;
         display: flex; align-items: center; justify-content: center;
         box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-family: Arial, sans-serif;">
         <svg width="44" height="44" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="46" fill="none" stroke="#64748B" stroke-width="1.5"/>
            <text x="50" y="24" text-anchor="middle" font-size="20" font-weight="bold" fill="#DC2626">N</text>
            <text x="50" y="88" text-anchor="middle" font-size="18" font-weight="bold" fill="#334155">S</text>
            <text x="82" y="56" text-anchor="middle" font-size="18" font-weight="bold" fill="#334155">E</text>
            <text x="18" y="56" text-anchor="middle" font-size="18" font-weight="bold" fill="#334155">W</text>
            <polygon points="50,28 45,48 55,48" fill="#DC2626"/>
            <polygon points="50,72 45,52 55,52" fill="#334155"/>
         </svg>
    </div>
    """
    pdam_map.get_root().html.add_child(folium.Element(compass_html))

    # ===============================================================
    # 3. MARKER CLUSTERING DENGAN DECLUSTER OTOMATIS
    # ===============================================================
    cluster_group_name = (
        "Pelanggan PDAM (Warna Status Lonjakan)"
        if color_by == "spike_status"
        else "Pelanggan PDAM (Warna Klaster K-Means)"
    )
    customer_layer = folium.FeatureGroup(name=cluster_group_name)

    mc = MarkerCluster(
        name="MarkerCluster Pelanggan",
        options={
            "maxClusterRadius": 35,
            "spiderfyOnMaxZoom": True,
            "spiderfyDistanceMultiplier": 2.0,
            "zoomToBoundsOnClick": True,
            "disableClusteringAtZoom": 17,
            "removeOutsideVisibleBounds": True,
        },
    ).add_to(customer_layer)

    for _, row in valid_geo.iterrows():
        # Parsing Klaster
        raw_cid = row.get("cluster_id")
        try:
            cid = int(raw_cid) if (pd.notna(raw_cid) and str(raw_cid).strip() != "") else 1
        except Exception:
            cid = 1
        clust_color = CLUSTER_COLORS.get(cid, "#0288D1")
        clust_label = CLUSTER_LABELS.get(cid, f"Klaster {cid+1}")

        # Parsing Status Lonjakan
        spike_status = str(row.get("spike_status", "NORMAL")).upper()
        if "LONJAKAN" in spike_status:
            status_color = "#DC2626"
            status_label = "Indikasi Lonjakan (> +30%)"
            st_key = "INDIKASI_LONJAKAN"
        elif "RENDAH" in spike_status or "ANOMALI" in spike_status:
            status_color = "#D97706"
            status_label = "Anomali Rendah (< -30%)"
            st_key = "ANOMALI_RENDAH"
        else:
            status_color = "#16A34A"
            status_label = "Konsumsi Normal (Wajar)"
            st_key = "NORMAL"

        # Tentukan warna marker utama berdasarkan skema terpilih
        if color_by == "cluster":
            marker_color = clust_color
        else:
            marker_color = status_color

        nopel = str(row["NoPEL"]).zfill(8)
        alamat_val = str(row.get("Alamat", "-")).replace("\\", "/").replace("`", "'").replace("$", "&#36;")
        rayon_val = str(row.get("RAYON", "-")).replace("\\", "/").replace("`", "'").replace("$", "&#36;")
        keterangan_val = str(row.get("keterangan", "-")).strip()

        # Riwayat konsumsi & nilai numerik
        hist_vals = row.get("history_usage_json", [])
        period_labs = row.get("history_periods_json", [])

        # Nilai pemakaian aktual dengan fallback ke histori terakhir
        actual_val = None
        if pd.notna(row.get("actual")):
            try:
                actual_val = float(row.get("actual"))
            except Exception:
                pass
        if actual_val is None or actual_val < 0:
            if isinstance(hist_vals, list) and hist_vals:
                actual_val = float(hist_vals[-1])
            else:
                actual_val = 0.0

        # Nilai prediksi ANN dengan fallback logis
        pred_val = None
        if pd.notna(row.get("prediction")):
            try:
                pred_val = float(row.get("prediction"))
            except Exception:
                pass
        if pred_val is None or pred_val < 0:
            pred_val = actual_val

        # Selisih persen
        pct_diff = 0.0
        if pd.notna(row.get("pct_diff")):
            try:
                pct_diff = float(row.get("pct_diff"))
            except Exception:
                pct_diff = 0.0
        elif pred_val and pred_val > 0:
            pct_diff = ((actual_val - pred_val) / pred_val) * 100.0

        err_abs = abs(actual_val - pred_val)

        # Golongan tarif
        tarif_val = str(row.get("tarif", "1.0"))
        try:
            tarif_str = f"Tarif {int(float(tarif_val))}"
        except Exception:
            tarif_str = f"Tarif {tarif_val}"

        # SVG Chart riwayat 12 periode
        chart_svg = generate_svg_chart(hist_vals, period_labs, pred_val=pred_val, actual_val=actual_val)

        # Catatan kondisi lapangan jika ada
        kondisi_badge = ""
        if keterangan_val and keterangan_val.lower() not in ["-", "nan", "normal", "none"]:
            kondisi_badge = (
                f'<div style="margin-top: 3px; margin-bottom: 3px;">'
                f'<span style="background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 600;">'
                f'Catatan Lapangan: {keterangan_val}</span></div>'
            )

        # Palet Warna Badge Klaster (Harmonisasi Lembut & Bebas Tabrakan Warna)
        if cid == 0:
            c_bg, c_tx, c_bd = "#DCFCE7", "#166534", "#86EFAC"  # Hijau Slamet (Rendah)
        elif cid == 1:
            c_bg, c_tx, c_bd = "#E0F2FE", "#0369A1", "#7DD3FC"  # Biru Tirta (Sedang)
        else:
            c_bg, c_tx, c_bd = "#E0E7FF", "#1E3A8A", "#A5B4FC"  # Biru Dongker Gelombang Logo (Tinggi)

        # Palet Warna Badge Status Lonjakan
        if "LONJAKAN" in spike_status:
            s_bg, s_tx, s_bd = "#FEE2E2", "#991B1B", "#FCA5A5"  # Merah Perwira Lembut
        elif "RENDAH" in spike_status or "ANOMALI" in spike_status:
            s_bg, s_tx, s_bd = "#FEF3C7", "#92400E", "#FCD34D"  # Amber Lembut
        else:
            s_bg, s_tx, s_bd = "#DCFCE7", "#166534", "#86EFAC"  # Hijau Alami Lembut

        # Kartu Pop-up Lengkap Sesuai Standar PDAM & Palet Logo
        popup_content = f"""
        <div style="font-family: Arial, Helvetica, sans-serif; font-size: 11px; line-height: 1.5; color: #1E293B; width: 285px; padding: 4px;">
            <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 6px; color: #0F172A; border-bottom: 2px solid #0284C7; padding-bottom: 4px; display: flex; align-items: center; justify-content: space-between;">
                <span>Detail Pelanggan PDAM</span>
                <span style="font-size: 9.5px; color: #0284C7; font-weight: 700; letter-spacing: 0.5px;">TIRTA PERWIRA</span>
            </div>
            <div style="margin-bottom: 2px;"><b>ID Pelanggan (NoPEL):</b> <span style="font-family: monospace; font-weight: 600;">{nopel}</span></div>
            <div style="margin-bottom: 2px;"><b>Alamat:</b> {alamat_val}</div>
            <div style="margin-bottom: 2px;"><b>Rayon / Cabang:</b> {rayon_val} | <b>Golongan:</b> {tarif_str}</div>
            {kondisi_badge}

            <div style="margin-top: 6px; margin-bottom: 5px;">
                <b style="color: #475569;">Segmentasi Profil Klaster:</b><br/>
                <span style="display:inline-block; margin-top:3px; background-color: {c_bg}; color: {c_tx}; border: 1px solid {c_bd}; padding: 3px 9px; border-radius: 5px; font-weight: 700; font-size: 10.5px; letter-spacing: 0.2px;">
                    {clust_label}
                </span>
            </div>

            <div style="margin-top: 6px; margin-bottom: 5px; border-top: 1px dashed #E2E8F0; padding-top: 5px;">
                <b style="color: #475569;">Status Deteksi Lonjakan (±30%):</b><br/>
                <span style="display:inline-block; margin-top:3px; background-color: {s_bg}; color: {s_tx}; border: 1px solid {s_bd}; padding: 3px 9px; border-radius: 5px; font-weight: 700; font-size: 10.5px; letter-spacing: 0.2px;">
                    {status_label}
                </span>
            </div>

            <div style="margin-bottom: 2px; margin-top: 5px;"><b>Konsumsi Aktual:</b> <span style="color: #0284C7; font-weight: 700;">{actual_val:.1f} m³</span></div>
            <div style="margin-bottom: 2px;"><b>Prediksi Model ANN:</b> <span style="color: #16A34A; font-weight: 700;">{pred_val:.1f} m³</span></div>
            <div style="margin-bottom: 2px;"><b>Selisih Error:</b> <span style="color: {status_color}; font-weight: 700;">{err_abs:.1f} m³ ({pct_diff:+.1f}%)</span></div>

            <div style="margin-top: 8px; margin-bottom: 3px; font-weight: 600; color: #0369A1;">
                Riwayat 12 Periode Konsumsi Air (Aktual vs Prediksi):
            </div>
            {chart_svg}
        </div>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5.5,
            color="#FFFFFF",
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.92,
            weight=1.2,
            tooltip=f"NoPEL: {nopel} | {clust_label} | {status_label}",
            popup=folium.Popup(popup_content, max_width=320),
        ).add_to(mc)

    customer_layer.add_to(pdam_map)
    folium.LayerControl(collapsed=False, position="topright").add_to(pdam_map)
    Fullscreen(position="topright").add_to(pdam_map)

    return pdam_map

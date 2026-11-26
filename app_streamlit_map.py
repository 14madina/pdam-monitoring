import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import folium
from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Peta Interaktif PDAM Purbalingga | Dual Model",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 800;
        color: #0D47A1;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #546E7A;
        margin-bottom: 16px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #1976D2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .kpi-num {
        font-size: 22px;
        font-weight: 800;
        color: #1A365D;
    }
    .kpi-lbl {
        font-size: 11px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .resume-box {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .badge-normal {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        border: 1px solid #A5D6A7;
    }
    .badge-spike {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        border: 1px solid #EF9A9A;
    }
    .badge-anomaly {
        background-color: #FFF8E1;
        color: #F57F17;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        border: 1px solid #FFE082;
    }
    .badge-cluster-0 {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        border: 1px solid #90CAF9;
    }
    .badge-cluster-1 {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        border: 1px solid #FFCC80;
    }
    .badge-cluster-2 {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 11px;
        border: 1px solid #A5D6A7;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADER (CACHED)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Memuat data spasial & model...")
def load_data():
    base_paths = [
        Path("pdam_skripsi_project/outputs/maps/streamlit_app_data.parquet"),
        Path("outputs/maps/streamlit_app_data.parquet"),
        Path("../outputs/maps/streamlit_app_data.parquet")
    ]
    data_path = None
    for bp in base_paths:
        if bp.exists():
            data_path = bp
            break
            
    if data_path is None:
        st.error("File dataset `streamlit_app_data.parquet` tidak ditemukan!")
        return pd.DataFrame()
        
    df = pd.read_parquet(data_path)
    df['NoPEL'] = df['NoPEL'].astype(str)
    
    # Pre-parse history json strings
    def safe_json_load(val):
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except:
            return []
            
    df['history_vals'] = df['history_usage_json'].apply(safe_json_load)
    df['period_labels'] = df['history_periods_json'].apply(safe_json_load)
    return df

df_all = load_data()

if df_all.empty:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR: FILTER & PENCARIAN PELANGGAN
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/water.png", width=64)
st.sidebar.markdown("### 🎛️ Kontrol & Filter Peta")

# 1. Filter Rayon
rayons = ["Semua Rayon"] + sorted([r for r in df_all['RAYON'].dropna().unique() if str(r).strip() != ''])
sel_rayon = st.sidebar.selectbox("Pilih Wilayah / Rayon:", rayons, index=0)

# 2. Filter K-Means Cluster
clusters = ["Semua Cluster"] + sorted(df_all['cluster_label'].dropna().unique().tolist())
sel_cluster = st.sidebar.selectbox("Filter Segmentasi K-Means:", clusters, index=0)

# 3. Filter Spike Status
spike_statuses = ["Semua Status"] + sorted(df_all['spike_status'].dropna().unique().tolist())
sel_status = st.sidebar.selectbox("Filter Status Spike (ANN):", spike_statuses, index=0)

# Apply filters
df_filtered = df_all.copy()
if sel_rayon != "Semua Rayon":
    df_filtered = df_filtered[df_filtered['RAYON'] == sel_rayon]
if sel_cluster != "Semua Cluster":
    df_filtered = df_filtered[df_filtered['cluster_label'] == sel_cluster]
if sel_status != "Semua Status":
    df_filtered = df_filtered[df_filtered['spike_status'] == sel_status]

st.sidebar.markdown(f"**Data tersaring:** {len(df_filtered):,} dari {len(df_all):,} pelanggan")

# 4. Search Customer by NoPEL / Alamat
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Cari Pelanggan Spesifik")

# Create searchable options
search_options = ["-- Pilih untuk Fokus Titik --"] + [
    f"{r['NoPEL']} | {r['Alamat']} ({r['spike_status']})" 
    for _, r in df_all.head(5000).iterrows()
]
selected_search = st.sidebar.selectbox("Cari NoPEL / Alamat:", search_options, index=0)

# 5. Marker rendering limit slider
st.sidebar.markdown("---")
max_markers = st.sidebar.slider(
    "Maksimal Titik Ditampilkan di Peta:", 
    min_value=200, 
    max_value=min(5000, len(df_filtered)), 
    value=min(1500, len(df_filtered)),
    step=100,
    help="Batasan jumlah titik untuk menjaga responsivitas interaksi geser & zoom peta."
)

color_mode = st.sidebar.radio(
    "Skema Pewarnaan Titik:",
    ["Segmentasi K-Means", "Status Spike Detection"],
    index=0
)

use_clustering = st.sidebar.checkbox("Gunakan Marker Clustering", value=True)

# ══════════════════════════════════════════════════════════════════════════
# RESOLVE ACTIVE SELECTED CUSTOMER
# ══════════════════════════════════════════════════════════════════════════
active_nopel = None

if selected_search != "-- Pilih untuk Fokus Titik --":
    active_nopel = selected_search.split(" | ")[0].strip()

# Check session state for clicked marker
if 'last_clicked_nopel' not in st.session_state:
    st.session_state['last_clicked_nopel'] = None

# If user clicked marker or searched, pick that customer
if active_nopel is None and st.session_state['last_clicked_nopel'] is not None:
    active_nopel = st.session_state['last_clicked_nopel']

# Fallback default customer if none selected yet
if active_nopel is None or active_nopel not in df_all['NoPEL'].values:
    # Pick a spike customer as default showcase
    spike_subset = df_all[df_all['spike_status'] == 'INDIKASI_LONJAKAN']
    if not spike_subset.empty:
        active_nopel = spike_subset.iloc[0]['NoPEL']
    else:
        active_nopel = df_all.iloc[0]['NoPEL']

active_customer = df_all[df_all['NoPEL'] == active_nopel].iloc[0]

# ══════════════════════════════════════════════════════════════════════════
# HEADER & KPI METRICS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">💧 Dashboard Peta Interaktif PDAM Tirta Perwira Kab. Purbalingga</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Integrasi Spasial Dua Model: <b>K-Means Clustering</b> (Segmentasi Konsumsi) & <b>ANN Regression</b> (Prediksi & Deteksi Lonjakan / Spike)</div>', unsafe_allow_html=True)

# Top KPI Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-lbl">Total Pelanggan Terpetakan</div>
        <div class="kpi-num">{len(df_all):,}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    mean_act = df_all['actual'].mean()
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: #00ACC1;">
        <div class="kpi-lbl">Rata-Rata Pemakaian</div>
        <div class="kpi-num">{mean_act:.1f} <span style="font-size:13px;">m³</span></div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    n_spike = (df_all['spike_status'] == 'INDIKASI_LONJAKAN').sum()
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: #E53935;">
        <div class="kpi-lbl">Indikasi Lonjakan (Spike)</div>
        <div class="kpi-num" style="color:#C62828;">{n_spike:,}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    n_drop = (df_all['spike_status'] == 'ANOMALI_RENDAH').sum()
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: #FB8C00;">
        <div class="kpi-lbl">Anomali Rendah (Drop)</div>
        <div class="kpi-num" style="color:#E65100;">{n_drop:,}</div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: #43A047;">
        <div class="kpi-lbl">Akurasi ANN (R² / RMSE)</div>
        <div class="kpi-num" style="color:#2E7D32;">81.87% <span style="font-size:12px;color:#666;">/ 29.69</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT LAYOUT: MAP (LEFT) & RESUME CARD (RIGHT)
# ══════════════════════════════════════════════════════════════════════════
col_map, col_resume = st.columns([7, 5])

# Prepare Folium Map
with col_map:
    st.markdown("#### 🗺️ Peta Spasial Interaktif Pelanggan")
    st.caption("💡 *Peta dapat digerakkan (pan/zoom) secara bebas. Klik sembarang titik untuk membuka resume detail.*")
    
    # Map center (center on active customer if valid, else Purbalingga median)
    center_lat = float(active_customer['latitude']) if pd.notna(active_customer['latitude']) else df_filtered['latitude'].median()
    center_lon = float(active_customer['longitude']) if pd.notna(active_customer['longitude']) else df_filtered['longitude'].median()
    
    pdam_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14 if active_nopel else 13,
        tiles=None,
        control_scale=True
    )
    
    # Base layers
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Hybrid Satellite',
        overlay=False,
        control=True
    ).add_to(pdam_map)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False,
        control=True
    ).add_to(pdam_map)
    
    folium.TileLayer(
        tiles='CartoDB positron',
        name='CartoDB Light',
        overlay=False,
        control=True
    ).add_to(pdam_map)
    
    # Floating Compass Rose (Arah Mata Angin) SVG
    compass_html = """
    <div style="position: fixed; 
         top: 80px; left: 20px; width: 60px; height: 60px; 
         z-index:9999; background: rgba(255,255,255,0.95); border: 2px solid #37474F; border-radius: 50%;
         display: flex; align-items: center; justify-content: center;
         box-shadow: 0 2px 8px rgba(0,0,0,0.35);">
         <svg width="50" height="50" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#37474F" stroke-width="2"/>
            <polygon points="50,8 43,40 57,40" fill="#E53935" stroke="#B71C1C" stroke-width="0.5"/>
            <polygon points="50,92 43,60 57,60" fill="#90A4AE" stroke="#546E7A" stroke-width="0.5"/>
            <polygon points="92,50 60,43 60,57" fill="#90A4AE" stroke="#546E7A" stroke-width="0.5"/>
            <polygon points="8,50 40,43 40,57" fill="#90A4AE" stroke="#546E7A" stroke-width="0.5"/>
            <circle cx="50" cy="50" r="5" fill="#37474F"/>
            <text x="50" y="7" text-anchor="middle" font-size="10" font-weight="bold" fill="#E53935">N</text>
            <text x="50" y="99" text-anchor="middle" font-size="9" font-weight="bold" fill="#546E7A">S</text>
            <text x="97" y="53" text-anchor="middle" font-size="9" font-weight="bold" fill="#546E7A">E</text>
            <text x="4" y="53" text-anchor="middle" font-size="9" font-weight="bold" fill="#546E7A">W</text>
         </svg>
    </div>
    """
    pdam_map.get_root().html.add_child(folium.Element(compass_html))
    
    # Legend HTML
    legend_html = """
    <div style="position: fixed; 
         bottom: 25px; left: 20px; width: 230px;
         border: 2px solid #455A64; z-index:9999; font-size:11px; font-family: 'Segoe UI', Arial, sans-serif;
         background-color: rgba(255,255,255,0.96); padding: 10px 12px; border-radius: 8px; 
         box-shadow: 0 3px 10px rgba(0,0,0,0.3);">
         <div style="font-weight:bold; color:#0D47A1; margin-bottom:6px; border-bottom:1px solid #90CAF9; padding-bottom:3px;">
            🎨 Legenda Titik
         </div>
         <b>Segmentasi K-Means:</b><br>
         <span style="color:#1976D2; font-weight:bold;">●</span> Cluster 0: Rendah<br>
         <span style="color:#FB8C00; font-weight:bold;">●</span> Cluster 1: Sedang<br>
         <span style="color:#43A047; font-weight:bold;">●</span> Cluster 2: Tinggi<br>
         <hr style="margin:5px 0; border:0.5px solid #CFD8DC;">
         <b>Deteksi Spike (ANN):</b><br>
         <span style="color:#43A047; font-weight:bold;">●</span> NORMAL<br>
         <span style="color:#E53935; font-weight:bold;">●</span> INDIKASI LONJAKAN<br>
         <span style="color:#FF9800; font-weight:bold;">●</span> ANOMALI RENDAH
    </div>
    """
    pdam_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Render Points
    marker_target = MarkerCluster(name="Klaster Titik Pelanggan").add_to(pdam_map) if use_clustering else pdam_map
    
    # Sample points to render
    if len(df_filtered) > max_markers:
        # stratified sample by cluster if possible
        df_display = df_filtered.groupby('cluster_id', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(max_markers * len(x) / len(df_filtered))), random_state=42)
        )
    else:
        df_display = df_filtered.copy()
        
    # Ensure active customer is included in display
    if active_customer['NoPEL'] not in df_display['NoPEL'].values:
        df_display = pd.concat([df_display, pd.DataFrame([active_customer])], ignore_index=True)
        
    # Colors
    cluster_color_map = {0: '#1976D2', 1: '#FB8C00', 2: '#43A047'}
    spike_color_map = {'NORMAL': '#43A047', 'INDIKASI_LONJAKAN': '#E53935', 'ANOMALI_RENDAH': '#FF9800'}
    
    def generate_svg_chart_mini(history_vals, period_labels=None):
        if len(history_vals) < 2:
            return "<div style='color:grey;font-size:10px;'>Data historis terbatas</div>"
        vals = [float(v) for v in history_vals[-12:]]
        max_val = max(max(vals), 1.0)
        avg_val = np.mean(vals)
        thresh_val = avg_val * 1.3
        max_scale = max(max_val, thresh_val) * 1.15
        w, h = 260, 95
        pad_left, pad_right, pad_top, pad_bottom = 20, 10, 15, 20
        n = len(vals)
        pts, th_pts = [], []
        for i, val in enumerate(vals):
            x = pad_left + (i / (n - 1)) * (w - pad_left - pad_right)
            y = pad_top + (1 - val / max_scale) * (h - pad_top - pad_bottom)
            pts.append((x, y))
            y_th = pad_top + (1 - thresh_val / max_scale) * (h - pad_top - pad_bottom)
            th_pts.append((x, y_th))
        pts_str = ' '.join([f'{p[0]:.1f},{p[1]:.1f}' for p in pts])
        th_pts_str = ' '.join([f'{p[0]:.1f},{p[1]:.1f}' for p in th_pts])
        dots_blue = ''.join([f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="2.5" fill="#1976D2"/>' for p in pts])
        dots_orange = ''.join([f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="2" fill="#FB8C00"/>' for p in th_pts])
        x_labels = ''
        if period_labels and len(period_labels) >= 2:
            plabels = period_labels[-n:]
            indices = [0, len(plabels) // 2, len(plabels) - 1]
            for idx in set(indices):
                if idx < len(plabels):
                    x = pad_left + (idx / (n - 1)) * (w - pad_left - pad_right)
                    align = "start" if idx == 0 else ("end" if idx == len(plabels) - 1 else "middle")
                    x_labels += f'<text x="{x}" y="{h-4}" font-size="7.5" fill="#757575" font-family="Arial,sans-serif" text-anchor="{align}">{plabels[idx]}</text>'
        return f"""<svg width="{w}" height="{h}" style="background:#ffffff;border:1px solid #e0e0e0;border-radius:3px;margin-top:4px;">
            <line x1="{pad_left}" y1="{pad_top}" x2="{w-pad_right}" y2="{pad_top}" stroke="#f5f5f5" stroke-width="1"/>
            <line x1="{pad_left}" y1="{h-pad_bottom}" x2="{w-pad_right}" y2="{h-pad_bottom}" stroke="#e0e0e0" stroke-width="1"/>
            <polyline fill="none" stroke="#FB8C00" stroke-width="1.8" stroke-dasharray="3,3" points="{th_pts_str}" />
            {dots_orange}
            <polyline fill="none" stroke="#1976D2" stroke-width="2" points="{pts_str}" />
            {dots_blue}
            {x_labels}
        </svg>"""

    for _, row in df_display.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        nopel = str(row['NoPEL'])
        cid = int(row['cluster_id']) if pd.notna(row['cluster_id']) else 0
        s_status = str(row['spike_status'])
        
        if color_mode == "Segmentasi K-Means":
            dot_color = cluster_color_map.get(cid, '#1976D2')
        else:
            dot_color = spike_color_map.get(s_status, '#43A047')
            
        is_active = (nopel == active_nopel)
        
        actual_val = float(row.get('actual', 0))
        pred_val = float(row.get('prediction', 0))
        err_abs = abs(actual_val - pred_val)
        
        tarif_disp = str(row.get('tarif_name', ''))
        if not tarif_disp or tarif_disp.upper() in ['NAN', 'NONE', '']:
            t_num = row.get('tarif')
            tarif_disp = str(int(t_num)) if pd.notna(t_num) else 'Rumah Tangga C'

        kode_tarif_disp = str(row.get('tarif_symbol', 'R3'))
        if kode_tarif_disp.upper() in ['NAN', 'NONE', '']:
            kode_tarif_disp = 'R3'

        cluster_disp = str(row.get('cluster_label', f'Cluster {cid}'))

        status_color = '#C62828' if 'LONJAKAN' in s_status else ('#EF6C00' if ('RENDAH' in s_status or 'ANOMALI' in s_status) else '#2E7D32')

        h_vals = row.get('history_vals', [])
        p_labs = row.get('period_labels', [])
        chart_svg = generate_svg_chart_mini(h_vals, p_labs)
        
        popup_html = f"""
        <div style="font-family: Arial, Helvetica, sans-serif; font-size: 11px; line-height: 1.5; color: #212121; width: 275px; padding: 3px;">
            <div style="font-weight: bold; font-size: 13px; margin-bottom: 6px; color: #0D47A1; border-bottom: 1.5px solid #1976D2; padding-bottom: 3px;">
                Detail Pelanggan PDAM
            </div>
            <div style="margin-bottom: 2px;"><b>NoPEL:</b> {nopel}</div>
            <div style="margin-bottom: 2px;"><b>Alamat:</b> {row.get('Alamat', '-')}</div>
            <div style="margin-bottom: 2px;"><b>Rayon:</b> {row.get('RAYON', '-')}</div>
            <div style="margin-bottom: 2px;"><b>Tarif numerik:</b> {int(round(float(row['tarif']))) if pd.notna(row.get('tarif')) else '-'}</div>
            <div style="margin-top: 4px; margin-bottom: 4px;"><b>Cluster:</b> {cluster_disp}</div>
            <div style="margin-top: 6px; margin-bottom: 4px; font-weight: bold; color: #0D47A1; border-top: 1px dashed #B0BEC5; padding-top: 4px;">
                ANN Regression
            </div>
            <div style="margin-bottom: 2px;"><b>Status:</b> <span style="font-weight: bold; color: {status_color};">{s_status}</span></div>
            <div style="margin-bottom: 2px;"><b>Aktual:</b> {actual_val:.2f} m&sup3;</div>
            <div style="margin-bottom: 2px;"><b>Prediksi:</b> {pred_val:.2f} m&sup3;</div>
            <div style="margin-bottom: 2px;"><b>Error:</b> {err_abs:.2f} m&sup3;</div>
            <div style="margin-top: 8px; margin-bottom: 2px; font-weight: bold; color: #212121;">
                Grafik 12 Periode Konsumsi
            </div>
            {chart_svg}
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=9 if is_active else 5,
            color='#D50000' if is_active else dot_color,
            weight=3 if is_active else 1.0,
            fill=True,
            fill_color=dot_color,
            fill_opacity=0.95 if is_active else 0.85,
            tooltip=f"NoPEL: {nopel} | {row['Alamat'][:25]} | {s_status}",
            popup=folium.Popup(popup_html, max_width=320)
        ).add_to(marker_target)
        
    # Layer control and fullscreen
    folium.LayerControl(collapsed=False).add_to(pdam_map)
    Fullscreen(position='topright').add_to(pdam_map)
    
    # Render Folium inside Streamlit
    map_output = st_folium(
        pdam_map,
        width=None,
        height=620,
        returned_objects=["last_object_clicked", "last_clicked"]
    )
    
    # Check if a marker was clicked
    if map_output and map_output.get("last_object_clicked"):
        clicked_coord = map_output["last_object_clicked"]
        if clicked_coord and "lat" in clicked_coord and "lng" in clicked_coord:
            clat = clicked_coord["lat"]
            clng = clicked_coord["lng"]
            # Find nearest customer by euclidean distance
            dists = ((df_display['latitude'] - clat)**2 + (df_display['longitude'] - clng)**2)
            nearest_idx = dists.idxmin()
            if dists[nearest_idx] < 0.0001:  # within ~10 meters
                clicked_nopel = df_display.loc[nearest_idx, 'NoPEL']
                if st.session_state.get('last_clicked_nopel') != clicked_nopel:
                    st.session_state['last_clicked_nopel'] = clicked_nopel
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL: RESUME DETAIL PELANGGAN TERPILIH
# ══════════════════════════════════════════════════════════════════════════
with col_resume:
    st.markdown("#### 📋 Resume Titik Pelanggan Terpilih")
    
    cust = active_customer
    cid = int(cust['cluster_id']) if pd.notna(cust['cluster_id']) else 0
    s_status = str(cust['spike_status'])
    
    # Status badges
    cluster_badge_class = f"badge-cluster-{cid}"
    if s_status == 'INDIKASI_LONJAKAN':
        spike_badge_class = "badge-spike"
        spike_icon = "🔴"
    elif s_status == 'ANOMALI_RENDAH':
        spike_badge_class = "badge-anomaly"
        spike_icon = "🟡"
    else:
        spike_badge_class = "badge-normal"
        spike_icon = "🟢"
        
    st.markdown(f"""
    <div class="resume-box">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="font-size: 18px; font-weight: 800; color: #0D47A1; font-family: monospace;">
                NoPEL: {cust['NoPEL']}
            </div>
            <div>
                <span class="{cluster_badge_class}">{cust['cluster_label']}</span>
                &nbsp;
                <span class="{spike_badge_class}">{spike_icon} {s_status}</span>
            </div>
        </div>
        
        <table style="width: 100%; font-size: 13px; line-height: 1.6; border-collapse: collapse; margin-bottom: 8px;">
            <tr>
                <td style="color:#546E7A; width: 32%;"><b>Alamat</b></td>
                <td style="color:#263238;">: {cust['Alamat']}</td>
            </tr>
            <tr>
                <td style="color:#546E7A;"><b>Wilayah / Rayon</b></td>
                <td style="color:#263238;">: {cust['RAYON']} ({cust.get('source_area', 'Purbalingga')})</td>
            </tr>
            <tr>
                <td style="color:#546E7A;"><b>Golongan Tarif</b></td>
                <td style="color:#263238;">: {cust.get('golongan_tarif', 'Rumah Tangga C')} (Kode: {cust.get('tarif', '-')})</td>
            </tr>
            <tr>
                <td style="color:#546E7A;"><b>Keterangan Meter</b></td>
                <td style="color:#263238;">: {cust.get('keterangan', 'Normal')}</td>
            </tr>
            <tr>
                <td style="color:#546E7A;"><b>Koordinat Spasial</b></td>
                <td style="color:#263238; font-family: monospace;">: ({cust['latitude']:.5f}, {cust['longitude']:.5f})</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Evaluation Metrics Sub-card
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Aktual", f"{cust['actual']:.1f} m³")
    with m2:
        st.metric("Prediksi ANN", f"{cust['prediction']:.1f} m³")
    with m3:
        err = cust['err_abs']
        st.metric("Error Abs", f"{err:.1f} m³")
    with m4:
        pct = cust['pct_diff']
        st.metric("Selisih %", f"{pct:+.1f}%")
        
    # Operational Action Item
    if s_status == 'INDIKASI_LONJAKAN':
        st.error(f"""
        🚨 **Rekomendasi Tindak Lanjut Lapangan:**  
        Konsumsi air pelanggan melonjak **{pct:+.1f}%** melampaui batas ambang model (+30%).  
        • Lakukan inspeksi meteran air di lokasi pelanggan.  
        • Periksa potensi kebocoran pipa instalasi dalam rumah tangga atau komersial.
        """)
    elif s_status == 'ANOMALI_RENDAH':
        st.warning(f"""
        ⚠️ **Rekomendasi Tindak Lanjut Lapangan:**  
        Konsumsi air pelanggan mengalami penurunan drastis **{pct:+.1f}%** di bawah ambang model (-30%).  
        • Periksa kondisi meteran air (indikasi meter macet/rusak).  
        • Konfirmasi status hunian (apakah rumah dalam keadaan kosong/ditinggalkan).
        """)
    else:
        st.success("""
        ✅ **Status Operasional Aman:**  
        Konsumsi air berada dalam rentang toleransi wajar (-30% s/d +30%). Pola konsumsi stabil.
        """)
        
    # Interactive Plotly Chart: 12-Month Consumption Trend
    st.markdown("##### 📈 Tren Konsumsi 12 Periode Historis")
    history_vals = cust['history_vals']
    period_labels = cust['period_labels']
    
    if len(history_vals) >= 2:
        fig = go.Figure()
        
        # Average line
        avg_val = np.mean(history_vals)
        spike_thresh = avg_val * 1.3
        drop_thresh = avg_val * 0.7
        
        # Actual history line
        fig.add_trace(go.Scatter(
            x=period_labels,
            y=history_vals,
            mode='lines+markers',
            name='Konsumsi Aktual (m³)',
            line=dict(color='#1976D2', width=2.5),
            marker=dict(size=6, color='#0D47A1')
        ))
        
        # Highlight last point (current month)
        fig.add_trace(go.Scatter(
            x=[period_labels[-1]],
            y=[history_vals[-1]],
            mode='markers',
            name='Bulan Ini',
            marker=dict(size=11, color='#D50000', symbol='diamond')
        ))
        
        # Threshold line (+30% Spike)
        fig.add_trace(go.Scatter(
            x=period_labels,
            y=[spike_thresh] * len(period_labels),
            mode='lines',
            name='Batas Lonjakan (+30%)',
            line=dict(color='#E53935', width=1.5, dash='dash')
        ))
        
        # Average line
        fig.add_trace(go.Scatter(
            x=period_labels,
            y=[avg_val] * len(period_labels),
            mode='lines',
            name='Rata-Rata Historis',
            line=dict(color='#78909C', width=1, dash='dot')
        ))
        
        fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=15, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(showgrid=True, gridcolor='#ECEFF1', tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor='#ECEFF1', title="m³"),
            plot_bgcolor='rgba(248,250,252,0.6)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data historis 12 periode tidak cukup untuk menampilkan grafik tren.")

# ══════════════════════════════════════════════════════════════════════════
# FOOTER: SCIENTIFIC SUMMARY & MODEL SPECIFICATION
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("🔬 Metodologi Ilmiah & Spesifikasi Model (Standar Publikasi Jurnal)"):
    st.markdown(r"""
    ### Ringkasan Rancangan Model & Validasi Ilmiah:
    1. **Segmentasi Pelanggan (K-Means Clustering):**
       - **Skenario:** `S1_SELECTED_FEATURES` (Hasil seleksi 5 metode consensus: Spearman, Mutual Info, F-Regression, RF, RFE).
       - **K Optimal:** $K=3$ (Cluster 0: Konsumsi Rendah, Cluster 1: Konsumsi Sedang, Cluster 2: Konsumsi Tinggi).
       - **Metrik Klaster:** *Silhouette Score* = **0.9331**, *Davies-Bouldin Index* = **0.1822** (Lebih unggul dibandingkan S0 All Features: Silhouette=0.9280, DBI=0.3567).
    2. **Prediksi Deret Waktu & Regresi (Artificial Neural Network):**
       - **Arsitektur:** Feedforward Neural Network dengan Regularisasi Dropout & Early Stopping.
       - **Split Data:** *Temporal Train-Validation-Test Split* (tanpa kebocoran data masa depan).
       - **Hasil Evaluasi Test:** $R^2 = 0.8187$, $RMSE = 29.69$ m³ (Unggul terhadap baseline lag-1 $R^2 = 0.8160$ dan S0 $R^2 = 0.8181$).
    3. **Deteksi Lonjakan (Spike Detection):**
       - Mengukur deviasi persentase antara realisasi pemakaian aktual terhadap hasil prediksi model ANN.
       - Formula: $\Delta\% = \\frac{y_{\\text{aktual}} - \\hat{y}_{\\text{prediksi}}}{\\hat{y}_{\\text{prediksi}}} \\times 100\%$.
       - Klasifikasi: $\Delta\% > +30\%$ (**Indikasi Lonjakan**), $\Delta\% < -30\%$ (**Anomali Rendah**), lainnya (**Normal**).
    """)

# Expandable Data Table
with st.expander(f"📊 Tabel Data Pelanggan Tersaring ({len(df_filtered):,} Data)"):
    display_cols = ['NoPEL', 'Alamat', 'RAYON', 'cluster_label', 'actual', 'prediction', 'pct_diff', 'spike_status', 'latitude', 'longitude']
    st.dataframe(df_filtered[display_cols].head(500), use_container_width=True)
    csv_data = df_filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Tersaring (CSV)",
        data=csv_data,
        file_name="pdam_pelanggan_spasial_filtered.csv",
        mime="text/csv"
    )

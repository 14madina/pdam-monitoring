"""
app/app.py — Dashboard Sistem Smart Water Monitoring PDAM Tirta Perwira Purbalingga
Portal Khusus Administrator PDAM Berbasis K-Means Clustering & ANN Regression.

Fitur Modul Administrator:
1. Login Sistem Administrator PDAM
2. Dashboard Utama (KPI, Distribusi Klaster, Grafik Aktual vs Prediksi, Sebaran Anomali Rayon)
3. Data Pelanggan (Upload Stand Meteran Excel, Filter Rayon & Periode, Riwayat)
4. Prediksi Pemakaian (Estimasi ANN Regression, Tren Konsumsi, Simulator Mandiri)
5. Deteksi Lonjakan (Ambang Batas ±30%, Skor Keparahan, Besaran Kenaikan)
6. Peta Sebaran Pelanggan (Folium Web-GIS, Multi-layer Satellite, Pop-up Sparkline 12 Periode)
7. Rekomendasi PDAM (Tindakan Operasional, Pelanggan Prioritas, Ekspor SPK Excel)
"""

import datetime
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

# Setup paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Logo path & Base64 encoder helper
LOGO_PATH = APP_DIR / "assets" / "logo_tirta_perwira.png"
if not LOGO_PATH.exists():
    LOGO_PATH = PROJECT_DIR / "app" / "assets" / "logo_tirta_perwira.png"


def get_logo_base64() -> str:
    """Mengembalikan logo dalam format Data URI Base64 untuk rendering mulus di HTML/SVG."""
    if LOGO_PATH.exists():
        import base64
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    return ""

import folium
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# Import modular ML & DB components
import importlib
from src import (
    clustering,
    config,
    database,
    feature_engineering,
    geo_map,
    pipeline_runner,
    prediction,
    preprocessing,
    recommendation,
    spike_detection,
)

# Selalu reload geo_map & clustering agar style terbaru langsung aktif tanpa terhalang cache
try:
    importlib.reload(geo_map)
    importlib.reload(clustering)
except Exception:
    pass

# Inisialisasi Database jika belum terisi
try:
    database.seed_database_if_empty()
except Exception as e:
    logging.warning(f"Database init check: {e}")

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="PERUMDA AIR MINUM TIRTA PERWIRA — Smart Water Monitoring Purbalingga",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# MODERN CUSTOM STYLING (Harmonisasi Logo Tirta Perwira - Flat, Clean, Non-Gradient)
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1E293B;
}

/* Background & Body */
.main {
    background-color: #F8FAFC;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
}

/* Sub-Bab & Headings — Estetik Biru Tirta Perwira (User Friendly & Nyaman) */
h2, .stMarkdown h2, [data-testid="stMarkdownContainer"] h2 {
    color: #0369A1 !important; /* Biru Tirta Pekat */
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px;
    margin-top: 1.2rem !important;
    margin-bottom: 0.8rem !important;
}
h3, .stMarkdown h3, [data-testid="stMarkdownContainer"] h3 {
    color: #0284C7 !important; /* Biru Tirta Primer */
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.6rem !important;
}
h4, .stMarkdown h4, [data-testid="stMarkdownContainer"] h4 {
    color: #0F4C81 !important; /* Biru Dongker Gelombang Logo */
    font-size: 0.98rem !important;
    font-weight: 600 !important;
}

/* Header Banner — Biru Tirta Perwira (Flat Solid, Sesuai Color Palette Logo, Tanpa Gradasi) */
.pdam-header {
    background-color: #0284C7;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 22px;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.2);
}
.pdam-header h1 {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0;
    color: #FFFFFF !important;
    letter-spacing: -0.3px;
}
.pdam-header p {
    margin: 4px 0 0 0;
    font-size: 0.88rem;
    color: #E0F2FE !important;
    font-weight: 500;
}

/* Metric KPI Cards — Flat Elegant Cards */
.kpi-container {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.kpi-card {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 16px 18px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    transition: all 0.2s ease;
    flex: 1;
}
.kpi-card:hover {
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    transform: translateY(-1px);
}
.kpi-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 5px;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-danger {
    background-color: #FEE2E2;
    color: #DC2626;
    border: 1px solid #FECACA;
}
.badge-warning {
    background-color: #FEF3C7;
    color: #D97706;
    border: 1px solid #FDE68A;
}
.badge-success {
    background-color: #DCFCE7;
    color: #16A34A;
    border: 1px solid #BBF7D0;
}
.badge-info {
    background-color: #E0F2FE;
    color: #0284C7;
    border: 1px solid #BAE6FD;
}

/* Primary Button (Biru Logo Tirta Perwira — Solid Tanpa Gradasi) */
button[data-testid="stBaseButton-primary"],
div.stButton > button[kind="primary"] {
    background-color: #0284C7 !important;
    border: 1px solid #0284C7 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(2, 132, 199, 0.15) !important;
    transition: all 0.15s ease !important;
}
button[data-testid="stBaseButton-primary"]:hover,
div.stButton > button[kind="primary"]:hover {
    background-color: #0369A1 !important;
    border-color: #0369A1 !important;
    box-shadow: 0 3px 8px rgba(2, 132, 199, 0.25) !important;
}

/* Secondary Button */
button[data-testid="stBaseButton-secondary"],
div.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
    color: #334155 !important;
    background-color: #FFFFFF !important;
    font-weight: 500 !important;
}
button[data-testid="stBaseButton-secondary"]:hover,
div.stButton > button:not([kind="primary"]):hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}

/* Streamlit Built-in Metric Card */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase;
}
div[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* Sidebar Customization — Light Clean Theme */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label {
    color: #334155 !important;
}

/* Sidebar Navigation Header */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 8px !important;
}

/* Sidebar Radio Navigation Items */
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    margin-bottom: 5px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.15s ease !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"]:hover {
    background: #F1F5F9 !important;
    border-color: #CBD5E1 !important;
    transform: translateX(2px) !important;
}

/* Radio Button Circle */
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] div:first-child {
    border-color: #94A3B8 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] input:checked + div {
    background-color: #0284C7 !important;
    border-color: #0284C7 !important;
}

/* Active Navigation Item Highlight */
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) {
    background: #EFF6FF !important;
    border: 1px solid #BAE6FD !important;
    border-left: 4px solid #0284C7 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) span {
    color: #0369A1 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] p {
    color: #334155 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Sidebar Primary Buttons (Simpan Akun dsb — Tetap Biru Estetik) */
section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background-color: #0284C7 !important;
    color: #FFFFFF !important;
    border: 1px solid #0284C7 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    box-shadow: 0 1px 3px rgba(2, 132, 199, 0.2) !important;
    transition: all 0.15s ease !important;
}

section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {
    background-color: #0369A1 !important;
    border-color: #0369A1 !important;
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span {
    color: #FFFFFF !important;
}

/* Logout Button (Keluar - Secondary) */
section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    color: #DC2626 !important;
    border: 1px solid #FECACA !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    box-shadow: 0 1px 2px rgba(220, 38, 38, 0.05) !important;
    transition: all 0.15s ease !important;
}

section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]):hover {
    background: #FEF2F2 !important;
    border-color: #F87171 !important;
    color: #B91C1C !important;
}

section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) p,
section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) span {
    color: #DC2626 !important;
    font-weight: 600 !important;
}

/* Tables and Dataframes */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
}

/* Tabs styling — Satu Garis Biru Bersih Tanpa Dobel Merah */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E2E8F0 !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #64748B;
    border-radius: 6px 6px 0 0;
    border-bottom: none !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #0284C7 !important;
    height: 3px !important;
}
.stTabs [aria-selected="true"] {
    color: #0284C7 !important;
    border-bottom: none !important;
}

/* Slider BaseWeb Styling — Bulatan Geser Merah Diubah Menjadi Biru Tirta Perwira */
div[data-baseweb="slider"] [role="slider"],
div[data-testid="stSlider"] [role="slider"],
div[data-testid="stSliderThumb"],
div[role="slider"] {
    background-color: #0284C7 !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 4px rgba(2, 132, 199, 0.4) !important;
}
div[data-baseweb="slider"] [role="slider"]:hover,
div[data-baseweb="slider"] [role="slider"]:active,
div[data-testid="stSlider"] [role="slider"]:hover {
    background-color: #0369A1 !important;
    box-shadow: 0 0 0 6px rgba(2, 132, 199, 0.2) !important;
}
/* Slider Track Bar Aktif */
div[data-baseweb="slider"] > div > div > div:first-child,
div[data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"],
div[data-baseweb="slider"] div[style*="#ff4b4b"],
div[data-baseweb="slider"] div[style*="#FF4B4B"] {
    background-color: #0284C7 !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# AUTENTIKASI & MANAJEMEN SESI PETUGAS
# =============================================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = None


def render_login_screen():
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.4, 1])

    with col2:
        logo_b64 = get_logo_base64()
        logo_img_tag = (
            f'<img src="{logo_b64}" style="width: 125px; height: 125px; object-fit: contain; margin-bottom: 12px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.12));"/><br/>'
            if logo_b64 else '<div style="font-size: 2.5rem; margin-bottom: 8px; font-weight: 800; color: #0288D1;">PDAM</div>'
        )

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 24px;">
            {logo_img_tag}
            <h2 style="color: #0F172A; font-weight: 800; margin: 0; letter-spacing: -0.4px;">PERUMDA AIR MINUM TIRTA PERWIRA</h2>
            <div style="color: #0284C7; font-weight: 700; font-size: 0.95rem; margin-top: 4px; letter-spacing: 0.5px;">KABUPATEN PURBALINGGA</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<h4 style='color: #0F172A; font-weight: 700; margin-bottom: 16px;'>Portal Administrator PDAM</h4>", unsafe_allow_html=True)
            username_input = st.text_input("Nama Pengguna (Username)", placeholder="Masukkan username admin")
            password_input = st.text_input("Kata Sandi (Password)", type="password", placeholder="Masukkan kata sandi")
            
            submit_btn = st.button("Masuk Sistem", type="primary", use_container_width=True)

            if submit_btn:
                if not username_input or not password_input:
                    st.error("Silakan isi nama pengguna dan kata sandi.")
                else:
                    user = database.authenticate_user(username_input, password_input)
                    if user and user.get("role") == "admin":
                        st.session_state["authenticated"] = True
                        st.session_state["user_info"] = user
                        st.success(f"Login berhasil! Selamat datang, {user['full_name']}.")
                        st.rerun()
                    elif user and user.get("role") != "admin":
                        st.error("Akses ditolak: Portal ini dikonfigurasi khusus untuk Administrator PDAM.")
                    else:
                        st.error("Nama pengguna atau kata sandi tidak valid.")

            st.markdown("""
            <div style='text-align: center; font-size: 0.76rem; color: #94A3B8; margin-top: 14px;'>
                🔒 Sistem Terenkripsi & Terintegrasi PDAM Tirta Perwira
            </div>
            """, unsafe_allow_html=True)


if not st.session_state["authenticated"]:
    render_login_screen()
    st.stop()


# =============================================================================
# SIDEBAR NAVIGASI & PROFIL PETUGAS
# =============================================================================
current_user = st.session_state["user_info"]

with st.sidebar:
    logo_b64 = get_logo_base64()
    logo_sb_img = (
        f'<img src="{logo_b64}" style="width: 48px; height: 48px; object-fit: contain;"/>'
        if logo_b64 else '<div style="font-size: 1.2rem; font-weight: 800; color: #0284C7;">PDAM</div>'
    )

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px; padding: 4px 0; border-bottom: 1px solid #E2E8F0; padding-bottom: 14px;">
        {logo_sb_img}
        <div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #0F172A; line-height: 1.2;">TIRTA PERWIRA</div>
            <div style="font-size: 0.72rem; color: #0284C7; letter-spacing: 0.5px; font-weight: 700;">PURBALINGGA</div>
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 500; margin-top: 2px;">Smart Water Monitoring</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Info Akun Administrator — Clean Light Card
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);">
        <div style="font-size: 0.68rem; color: #64748B; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">LOGIN SEBAGAI:</div>
        <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-top: 3px; letter-spacing: -0.2px;">{current_user['full_name']}</div>
        <div style="display: flex; gap: 8px; margin-top: 8px;">
            <span style="background: #0284C7; color: #FFFFFF; font-size: 0.72rem; padding: 3px 10px; border-radius: 5px; font-weight: 700; letter-spacing: 0.4px;">ADMIN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Menu Navigasi Modul Administrator (6 Menu Standar)
    menu_options = [
        "Dashboard Utama",
        "Data Pelanggan",
        "Prediksi Pemakaian",
        "Deteksi Lonjakan",
        "Peta Sebaran Pelanggan",
        "Rekomendasi PDAM",
    ]

    # Inisialisasi navigasi sesi jika belum ada
    if "nav_radio" not in st.session_state or st.session_state["nav_radio"] not in menu_options:
        st.session_state["nav_radio"] = menu_options[0]

    # Handler jika ada perpindahan halaman terprogram via tombol alur kerja
    if "target_page" in st.session_state:
        st.session_state["nav_radio"] = st.session_state.pop("target_page")

    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px;">
        Panel Navigasi Administrator
    </div>
    """, unsafe_allow_html=True)

    selected_menu = st.radio(
        "Panel Navigasi Administrator",
        menu_options,
        key="nav_radio",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    if st.button("Keluar (Logout)", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user_info"] = None
        st.rerun()


# =============================================================================
# HELPER KOMPONEN: PENJELASAN GRAFIK & PETUNJUK LANGKAH BERIKUTNYA
# =============================================================================
def render_chart_caption(explanation_text: str, border_color: str = "#0284C7") -> str:
    """Helper untuk menampilkan penjelasan grafik yang singkat, estetik, dan informatif."""
    return f"""
    <div style="font-size: 0.81rem; color: #475569; margin-top: 8px; margin-bottom: 12px; padding: 7px 14px; background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3.5px solid {border_color}; border-radius: 6px; line-height: 1.5;">
        <b style="color: #0F172A;">Keterangan Visual:</b> {explanation_text}
    </div>
    """


def render_next_step_guide(current_page_num: int):
    guide_info = {
        1: ("Data Pelanggan", "Kelola atau unggah data stand meteran pelanggan bulanan"),
        2: ("Prediksi Pemakaian", "Jalankan model ANN Regression untuk mengestimasi konsumsi air wajar"),
        3: ("Deteksi Lonjakan", "Deteksi lonjakan konsumsi air (> +30%) dan anomali meter macet (< -30%)"),
        4: ("Peta Sebaran Pelanggan", "Pantau persebaran spasial pelanggan anomali dan sebaran rayon secara geografis"),
        5: ("Rekomendasi PDAM", "Susun rekomendasi tindakan operasional dan unduh Surat Perintah Kerja (SPK)"),
        6: (None, "Seluruh tahapan monitoring pemantauan air dari analisis hingga SPK telah selesai"),
    }
    next_target, desc = guide_info.get(current_page_num, (None, ""))
    if next_target:
        st.markdown(f"""
        <div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-left: 4px solid #0284C7; border-radius: 8px; padding: 14px 18px; margin-top: 28px; margin-bottom: 10px;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.5px;">Petunjuk Langkah Kerja Berikutnya:</div>
            <div style="font-size: 0.88rem; color: #334155; margin-top: 3px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        btn_c1, btn_c2 = st.columns([2.8, 1.2])
        with btn_c2:
            if st.button(
                f"Lanjut ke: {next_target}",
                key=f"btn_next_step_{current_page_num}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["target_page"] = next_target
                st.rerun()
    else:
        st.markdown(f"""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; border-radius: 8px; padding: 14px 18px; margin-top: 28px;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #16A34A; text-transform: uppercase; letter-spacing: 0.5px;">Alur Kerja Monitoring Selesai:</div>
            <div style="font-size: 0.88rem; color: #334155; margin-top: 3px;">{desc}. Seluruh data dan berkas SPK siap digunakan oleh divisi teknis/lapangan PDAM Tirta Perwira.</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# HELPER DATA LOADER DENGAN FILTER GLOBAL (TEROPTIMASI MEMORY CACHING)
# =============================================================================
@st.cache_data(ttl=300)
def load_cached_rayons():
    """Mengambil daftar rayon dengan caching memori agar query instan."""
    return ["ALL"] + database.get_available_rayons()

@st.cache_data(ttl=300)
def load_cached_periods():
    """Mengambil daftar periode dengan caching memori agar query instan."""
    return ["ALL"] + database.get_available_periods()

@st.cache_data(ttl=300)
def load_cached_dashboard_data(rayon="ALL", periode="ALL"):
    return database.get_dashboard_data(
        rayon=None if rayon == "ALL" else rayon,
        periode=None if periode == "ALL" else periode,
    )

@st.cache_data(show_spinner=False, ttl=300)
def generate_optimized_map_html(sub_df: pd.DataFrame, color_mode: str) -> str:
    """Cache rendering HTML peta leaflet agar navigasi instan tanpa merusak tipe data."""
    m = geo_map.build_pdam_interactive_map(sub_df, color_by=color_mode)
    return m.get_root().render()


# =============================================================================
# HALAMAN 1: DASHBOARD UTAMA
# =============================================================================
if selected_menu == "Dashboard Utama":
    st.markdown("""
    <div class="pdam-header">
        <h1>Dashboard Utama Konsumsi Air</h1>
    </div>
    """, unsafe_allow_html=True)

    # Filter Bar
    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 3])
    with f_col1:
        rayons = load_cached_rayons()
        selected_rayon = st.selectbox("Wilayah / Rayon", rayons, index=0)
    with f_col2:
        periods = load_cached_periods()
        selected_period = st.selectbox("Periode Tagihan", periods, index=0)

    df_dash = load_cached_dashboard_data(selected_rayon, selected_period)

    if df_dash.empty:
        st.info("Belum ada data pelanggan untuk filter yang dipilih. Silakan unggah data di menu 'Data Pelanggan'.")
    else:
        # Metrik KPI Utama
        total_cust = len(df_dash)
        avg_usage = df_dash["actual"].mean() if "actual" in df_dash.columns else 0.0
        n_spikes = (df_dash["spike_status"] == "INDIKASI_LONJAKAN").sum()
        pct_spikes = (n_spikes / total_cust * 100) if total_cust > 0 else 0.0
        n_low = (df_dash["spike_status"] == "ANOMALI_RENDAH").sum()

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card" style="border-top: 3px solid #0284C7;">
                <div class="kpi-title">Total Pelanggan Terdaftar</div>
                <div class="kpi-value">{total_cust:,}</div>
                <div class="kpi-sub">Pelanggan Aktif Termonitor</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #16A34A;">
                <div class="kpi-title">Rata-Rata Pemakaian Air</div>
                <div class="kpi-value">{avg_usage:.1f} <span style="font-size:1.05rem; font-weight:500; color:#64748B;">m³</span></div>
                <div class="kpi-sub">Volume Konsumsi Bulanan</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #DC2626;">
                <div class="kpi-title">Pelanggan Terindikasi Lonjakan</div>
                <div class="kpi-value" style="color: #DC2626;">{n_spikes:,}</div>
                <div class="kpi-sub"><span style="color: #DC2626; font-weight:600;">{pct_spikes:.1f}%</span> di atas threshold (+30%)</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #D97706;">
                <div class="kpi-title">Anomali Rendah / Potensi Macet</div>
                <div class="kpi-value" style="color: #D97706;">{n_low:,}</div>
                <div class="kpi-sub">Konsumsi turun &lt; -30%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns([1.65, 1.15])

        with col_g1:
            st.subheader("Perbandingan Pemakaian Aktual vs Prediksi ANN")
            tab_comp1, tab_comp2 = st.tabs([
                "Garis Waktu Lengkap (2020-2026)",
                "Sampel Pelanggan Interaktif",
            ])

            with tab_comp1:
                # Menampilkan grafik Full Timeline hasil riset (N = 103,175)
                p_full_timeline = APP_DIR / "assets" / "ann_full_timeline_actual_vs_predicted.png"
                if not p_full_timeline.exists():
                    p_full_timeline = config.REPORTS_DIR / "prediction" / "ann_full_timeline_actual_vs_predicted.png"
                if not p_full_timeline.exists():
                    p_full_timeline = config.PROJECT_ROOT / "outputs" / "prediction" / "ann_full_timeline_actual_vs_predicted.png"

                if p_full_timeline.exists():
                    st.image(
                        str(p_full_timeline),
                        caption="Full Timeline: Pemakaian Air Aktual (Seluruh Periode) vs Prediksi ANN (Periode Uji: Agu 2025 - Apr 2026 | N = 103.175 observasi)",
                        use_container_width=True,
                    )
                    st.markdown(render_chart_caption("Garis biru menggambarkan rata-rata konsumsi air pelanggan per bulan (2020–2026). Garis oranye putus-putus menunjukkan hasil estimasi model ANN pada periode uji yang terbukti konsisten mengikuti tren konsumsi riil."), unsafe_allow_html=True)
                else:
                    st.info("Grafik Full Timeline sedang diproses atau belum tersedia di direktori reports.")

            with tab_comp2:
                # Sample agregat untuk visualisasi interaktif pelanggan individual
                sample_chart = df_dash.head(60).copy().reset_index(drop=True)
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=list(range(len(sample_chart))),
                    y=sample_chart["actual"],
                    mode="lines+markers",
                    name="Pemakaian Aktual (m³)",
                    line=dict(color="#0284C7", width=2.5),
                    marker=dict(size=5),
                ))
                fig_comp.add_trace(go.Scatter(
                    x=list(range(len(sample_chart))),
                    y=sample_chart["prediction"],
                    mode="lines+markers",
                    name="Prediksi Model ANN (m³)",
                    line=dict(color="#16A34A", width=2.2, dash="dash"),
                    marker=dict(size=5, symbol="circle"),
                ))
                fig_comp.update_layout(
                    xaxis_title="Sampel Pelanggan",
                    yaxis_title="Volume Air (m³)",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=340,
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                st.markdown(render_chart_caption("Perbandingan langsung konsumsi aktual (biru) vs estimasi ANN (hijau putus-putus) pada 60 sampel pelanggan. Keselarasan titik menunjukkan akurasi model."), unsafe_allow_html=True)

        with col_g2:
            st.subheader("Distribusi Klaster Profil Pelanggan")

            def map_clean_cluster(row):
                cid = row.get("cluster_id")
                clab = str(row.get("cluster_label", ""))
                if pd.notna(cid):
                    try:
                        cid_int = int(cid)
                        if cid_int == 0:
                            return "Klaster 1: Rendah (ABRI/Khusus)"
                        elif cid_int == 1:
                            return "Klaster 2: Sedang (Rumah Tangga)"
                        elif cid_int == 2:
                            return "Klaster 3: Tinggi (Niaga/Industri)"
                    except Exception:
                        pass
                if "0" in clab or "Rendah" in clab:
                    return "Klaster 1: Rendah (ABRI/Khusus)"
                elif "1" in clab or "Sedang" in clab:
                    return "Klaster 2: Sedang (Rumah Tangga)"
                elif "2" in clab or "Tinggi" in clab:
                    return "Klaster 3: Tinggi (Niaga/Industri)"
                return "Pelanggan Baru (Belum Terklaster)"

            df_dash_copy = df_dash.copy()
            df_dash_copy["clean_klaster"] = df_dash_copy.apply(map_clean_cluster, axis=1)

            clust_summary = df_dash_copy["clean_klaster"].value_counts().reset_index()
            clust_summary.columns = ["Klaster", "Jumlah"]

            color_map = {
                "Klaster 1: Rendah (ABRI/Khusus)": "#16A34A",
                "Klaster 2: Sedang (Rumah Tangga)": "#0284C7",
                "Klaster 3: Tinggi (Niaga/Industri)": "#0F4C81",
                "Pelanggan Baru (Belum Terklaster)": "#94A3B8",
            }
            colors = [color_map.get(k, "#64748B") for k in clust_summary["Klaster"]]

            fig_donut = go.Figure(data=[go.Pie(
                labels=clust_summary["Klaster"],
                values=clust_summary["Jumlah"],
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="percent",
                hoverinfo="label+percent+value",
            )])
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.08,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10, family="Plus Jakarta Sans, sans-serif"),
                ),
                margin=dict(l=10, r=10, t=10, b=40),
                height=340,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown(render_chart_caption("Proporsi komposisi 3 profil pelanggan: Klaster 1 (Rendah - ABRI/Khusus), Klaster 2 (Sedang - Rumah Tangga, mayoritas 90%+), dan Klaster 3 (Tinggi - Niaga/Industri)."), unsafe_allow_html=True)

        # Bar chart sebaran per rayon
        st.subheader("Sebaran Status Anomali Berdasarkan Rayon Wilayah")
        top_rayons = df_dash["RAYON"].value_counts().head(12).index.tolist()
        df_rayon_spikes = df_dash[df_dash["RAYON"].isin(top_rayons)].groupby(["RAYON", "spike_status"]).size().reset_index(name="Jumlah")

        fig_rayon = px.bar(
            df_rayon_spikes,
            x="RAYON",
            y="Jumlah",
            color="spike_status",
            color_discrete_map={
                "INDIKASI_LONJAKAN": "#DC2626",
                "NORMAL": "#16A34A",
                "ANOMALI_RENDAH": "#D97706",
            },
            barmode="stack",
            height=320,
        )
        fig_rayon.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Kode Rayon Wilayah",
            yaxis_title="Jumlah Pelanggan",
            legend_title="Status Anomali",
        )
        st.plotly_chart(fig_rayon, use_container_width=True)
        st.markdown(render_chart_caption("Sebaran status meter per rayon wilayah: Hijau menandakan konsumsi normal, Merah menunjukkan lonjakan (>+30%), dan Oranye mendeteksi penurunan drastis (<-30% / meter macet)."), unsafe_allow_html=True)

        render_next_step_guide(1)


# =============================================================================
# HALAMAN 2: DATA PELANGGAN (UPLOAD & TABEL)
# =============================================================================
elif selected_menu == "Data Pelanggan":
    st.markdown("""
    <div class="pdam-header">
        <h1>Data Pelanggan & Pemrosesan Batch</h1>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Unggah File Excel Baru", "Database Pelanggan", "Riwayat Pemrosesan"])

    with tab1:
        st.subheader("Unggah File Stand Meteran Pelanggan (Excel / CSV)")

        uploaded_file = st.file_uploader(
            "Pilih file Excel (.xlsx, .xls) atau CSV",
            type=["xlsx", "xls", "csv"],
            help="File harus memiliki kolom minimum: NoPEL, Lalu, Ini. Kolom tarif dan periode bersifat opsional.",
        )

        if uploaded_file is not None:
            col_u1, col_u2 = st.columns([2, 1])
            with col_u1:
                st.write(f"**File Terpilih:** `{uploaded_file.name}` ({round(uploaded_file.size / 1024, 1)} KB)")
            with col_u2:
                btn_process = st.button("Jalankan Pemrosesan Otomatis", type="primary", use_container_width=True)

            if btn_process:
                progress_bar = st.progress(10)
                status_placeholder = st.empty()

                status_placeholder.info("Tahap 1/5: Memvalidasi schema dan integritas kolom input...")
                time.sleep(0.4)
                progress_bar.progress(30)

                status_placeholder.info("Tahap 2/5: Ekstraksi fitur fisik & inferensi klaster K-Means (K=3)...")
                time.sleep(0.4)
                progress_bar.progress(55)

                status_placeholder.info("Tahap 3/5: Prediksi konsumsi air menggunakan model ANN Regression...")
                time.sleep(0.4)
                progress_bar.progress(75)

                status_placeholder.info("Tahap 4/5: Deteksi lonjakan residual (±30%) & penyusunan rekomendasi...")
                
                # Eksekusi pipeline
                result = pipeline_runner.run_batch_pipeline(
                    file_or_df=uploaded_file,
                    source_name=uploaded_file.name,
                    save_to_db=True,
                    prediction_mode="cluster_specific",
                    scenario="selected_features",
                )

                progress_bar.progress(100)
                status_placeholder.empty()

                if result["success"]:
                    st.success(
                        f"Pemrosesan Selesai dalam {result['execution_time_sec']} detik! "
                        f"Total {result['total_records']:,} data berhasil diproses dan disimpan ke database."
                    )
                    
                    # Ringkasan hasil pemrosesan
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric("Total Data Diproses", f"{result['total_records']:,} baris")
                    with res_col2:
                        n_spike_up = result["spike_summary"].get("INDIKASI_LONJAKAN", 0)
                        st.metric("Terindikasi Lonjakan", f"{n_spike_up:,} pelanggan")
                    with res_col3:
                        spk_urgent = result["recommendation_summary"].get("spk_urgent_count", 0)
                        st.metric("SPK Prioritas Tinggi", f"{spk_urgent:,} tindakan")

                    # Preview hasil
                    st.subheader("Pratinjau Hasil Pemrosesan")
                    st.dataframe(result["df_result"].head(20), use_container_width=True)
                    st.cache_data.clear()
                else:
                    st.error(f"Gagal memproses file: {result['errors']}")

    with tab2:
        st.subheader("Tabel Data Master Pelanggan & Hasil Prediksi")

        # Filter bar
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            r_opts = load_cached_rayons()
            f_rayon = st.selectbox("Filter Rayon", r_opts, key="tab2_rayon")
        with col_f2:
            p_opts = load_cached_periods()
            f_period = st.selectbox("Filter Periode", p_opts, key="tab2_period")
        with col_f3:
            s_opts = ["ALL", "INDIKASI_LONJAKAN", "NORMAL", "ANOMALI_RENDAH"]
            f_status = st.selectbox("Filter Status Lonjakan", s_opts, key="tab2_status")
        with col_f4:
            search_nopel = st.text_input("Cari NoPEL / Alamat", placeholder="Ketik NoPEL...")

        df_cust = load_cached_dashboard_data(f_rayon, f_period)

        if f_status != "ALL":
            df_cust = df_cust[df_cust["spike_status"] == f_status]

        if search_nopel:
            q = search_nopel.strip().lower()
            df_cust = df_cust[
                df_cust["NoPEL"].str.lower().str.contains(q) |
                df_cust["Alamat"].str.lower().str.contains(q)
            ]

        st.write(f"Menampilkan **{len(df_cust):,}** data pelanggan")

        # Tampilkan tabel interaktif
        cols_display = [
            "NoPEL", "Nama", "Alamat", "RAYON", "tarif", "Ini", "Lalu",
            "actual", "prediction", "pct_diff", "spike_status", "recommendation_priority"
        ]
        cols_available = [c for c in cols_display if c in df_cust.columns]
        
        st.dataframe(
            df_cust[cols_available].head(1000),
            use_container_width=True,
            height=450,
        )

        # Tombol download
        csv_buffer = df_cust[cols_available].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh Data Terfilter (.csv)",
            data=csv_buffer,
            file_name=f"pdam_data_pelanggan_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    with tab3:
        st.subheader("Riwayat Log Pemrosesan Data (Audit Trail)")
        logs_df = database.get_recent_logs(limit=50)
        if logs_df.empty:
            st.info("Belum ada riwayat aktivitas pemrosesan.")
        else:
            st.dataframe(logs_df, use_container_width=True)

    render_next_step_guide(2)


# =============================================================================
# HALAMAN 3: PREDIKSI PEMAKAIAN (ANN REGRESSION)
# =============================================================================
elif selected_menu == "Prediksi Pemakaian":
    st.markdown("""
    <div class="pdam-header">
        <h1>Prediksi Pemakaian Air (ANN Regression)</h1>
    </div>
    """, unsafe_allow_html=True)

    tab_pred1, tab_pred2, tab_pred3 = st.tabs(["Analisis Sebaran Prediksi", "Simulator Prediksi Mandiri", "Prediksi Berdasarkan Klaster"])

    df_pred_all = load_cached_dashboard_data("ALL", "ALL")

    with tab_pred1:
        st.subheader("Analisis Sebaran Konsumsi Air: Aktual vs Prediksi ANN")

        if not df_pred_all.empty:
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                # Scatter plot actual vs predicted
                sample_p = df_pred_all.sample(min(1200, len(df_pred_all)), random_state=42)
                fig_scat = px.scatter(
                    sample_p,
                    x="actual",
                    y="prediction",
                    color="cluster_label",
                    opacity=0.6,
                    title="Scatter Plot: Aktual vs Prediksi (Garis Ideal y = x)",
                    labels={"actual": "Pemakaian Aktual (m³)", "prediction": "Prediksi ANN (m³)"},
                    template="plotly_white",
                    color_discrete_map={
                        "Klaster 1 (Rendah - ABRI/Khusus)": "#16A34A",
                        "Klaster 2 (Sedang - Rumah Tangga)": "#0284C7",
                        "Klaster 3 (Tinggi - Niaga/Industri)": "#0F4C81",
                    },
                )
                max_v = max(sample_p["actual"].max(), sample_p["prediction"].max())
                fig_scat.add_trace(go.Scatter(
                    x=[0, max_v], y=[0, max_v], mode="lines",
                    line=dict(color="#DC2626", dash="dash"), name="Ideal (y = x)"
                ))
                st.plotly_chart(fig_scat, use_container_width=True)

            with col_sc2:
                # Error residual distribution
                residuals = df_pred_all["actual"] - df_pred_all["prediction"]
                res_filtered = residuals[residuals.between(residuals.quantile(0.01), residuals.quantile(0.99))]
                fig_hist = px.histogram(
                    res_filtered,
                    nbins=40,
                    title="Distribusi Residual Error (y_true - y_pred)",
                    labels={"value": "Residual Error (m³)"},
                    color_discrete_sequence=["#0284C7"],
                    template="plotly_white",
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="#DC2626", annotation_text="Zero Error")
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown(render_chart_caption("Pada scatter plot (kiri), titik yang mendekati garis diagonal merah (y = x) menunjukkan akurasi model ANN. Pada histogram (kanan), selisih error terpusat di titik nol membuktikan estimasi konsumsi tidak bias."), unsafe_allow_html=True)

    with tab_pred2:
        st.subheader("Simulator Prediksi Konsumsi Air Mandiri")

        sim_col1, sim_col2 = st.columns([1.2, 1.5])

        with sim_col1:
            with st.container(border=True):
                st.markdown("#### Input Parameter Meteran")
                sim_lalu = st.number_input("Stand Meter Lalu (Lalu)", min_value=0.0, value=120.0, step=1.0)
                sim_ini = st.number_input("Stand Meter Terkini (Ini)", min_value=0.0, value=150.0, step=1.0)
                tarif_options = [1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0]
                tarif_labels = {
                    1.0: "Tarif 1 - ABRI / TNI POLRI",
                    3.0: "Tarif 3 - Industri Kecil",
                    4.0: "Tarif 4 - Industri Besar",
                    5.0: "Tarif 5 - Inst. Pemerintah",
                    6.0: "Tarif 6 - Niaga Kecil",
                    7.0: "Tarif 7 - Niaga Besar",
                    9.0: "Tarif 9 - Rumah Tangga B",
                    10.0: "Tarif 10 - Rumah Tangga C",
                    11.0: "Tarif 11 - Sosial Umum",
                    12.0: "Tarif 12 - Sosial Khusus",
                }
                sim_tarif = st.selectbox(
                    "Golongan Tarif",
                    tarif_options,
                    format_func=lambda x: tarif_labels.get(x, f"Tarif {int(x)}"),
                    index=7,
                )
                
                curr_usage = max(0.0, sim_ini - sim_lalu)
                st.info(f"Konsumsi Bulan Berjalan: **{curr_usage:.1f} m³** (Ini - Lalu)")

                btn_sim = st.button("Hitung Prediksi ANN", type="primary", use_container_width=True)

        with sim_col2:
            if btn_sim or "sim_result" not in st.session_state:
                pred_res = prediction.predict_single_customer(
                    ini=sim_ini,
                    lalu=sim_lalu,
                    tarif=sim_tarif,
                    mode="cluster_specific",
                    scenario="selected_features",
                )
                st.session_state["sim_result"] = pred_res

            res = st.session_state["sim_result"]
            with st.container(border=True):
                st.markdown("#### Hasil Prediksi & Klasifikasi Klaster")
                
                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.metric("Prediksi Bulan Depan", f"{res['predicted_next_usage']:.2f} m³")
                    cid = res.get('cluster_id', 1)
                    if cid == 0:
                        c_bg, c_tx, c_bd = "#DCFCE7", "#166534", "#86EFAC"
                    elif cid == 1:
                        c_bg, c_tx, c_bd = "#E0F2FE", "#0369A1", "#7DD3FC"
                    else:
                        c_bg, c_tx, c_bd = "#E0E7FF", "#1E3A8A", "#A5B4FC"
                    st.markdown(f"<div style='margin-top:6px;'><b>Klaster Profil:</b><br/><span style='display:inline-block; margin-top:3px; background:{c_bg}; color:{c_tx}; border:1px solid {c_bd}; padding:3px 9px; border-radius:5px; font-weight:700; font-size:0.82rem;'>{res['cluster_label']}</span></div>", unsafe_allow_html=True)
                with res_c2:
                    change_val = res['projected_change']
                    pct_val = res['projected_change_pct']
                    st.metric("Proyeksi Perubahan", f"{change_val:+.2f} m³", delta=f"{pct_val:+.1f}%")
                    st.markdown(f"<div style='margin-top:6px;'><b>Pola Tren:</b><br/><span style='display:inline-block; margin-top:3px; background:#F1F5F9; color:#334155; padding:3px 9px; border-radius:5px; font-weight:600; font-size:0.82rem; border:1px solid #E2E8F0;'>{res['trend_summary']}</span></div>", unsafe_allow_html=True)

    with tab_pred3:
        st.subheader("Komparasi Hasil Prediksi Berdasarkan Profil Klaster")
        if not df_pred_all.empty:
            fig_box = px.box(
                df_pred_all,
                x="cluster_label",
                y="prediction",
                color="cluster_label",
                title="Distribusi Sebaran Prediksi Pemakaian Air per Klaster Pelanggan",
                labels={"prediction": "Prediksi Pemakaian Air (m³)", "cluster_label": "Klaster"},
                template="plotly_white",
                color_discrete_map={
                    "Klaster 1 (Rendah - ABRI/Khusus)": "#16A34A",
                    "Klaster 2 (Sedang - Rumah Tangga)": "#0284C7",
                    "Klaster 3 (Tinggi - Niaga/Industri)": "#0F4C81",
                }
            )
            fig_box.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown(render_chart_caption("Rentang konsumsi air (m³) yang diprediksi untuk masing-masing klaster: Klaster 3 memiliki konsumsi tertinggi dan variabilitas terbesar, sedangkan Klaster 1 memiliki konsumsi paling rendah."), unsafe_allow_html=True)

    render_next_step_guide(3)


# =============================================================================
# HALAMAN 4: DETEKSI LONJAKAN
# =============================================================================
elif selected_menu == "Deteksi Lonjakan":
    st.markdown("""
    <div class="pdam-header">
        <h1>Deteksi Lonjakan & Anomali Pemakaian Air</h1>
    </div>
    """, unsafe_allow_html=True)

    df_spikes_page = load_cached_dashboard_data("ALL", "ALL")

    if df_spikes_page.empty:
        st.info("Belum ada data pemakaian untuk deteksi lonjakan.")
    else:
        # Metrik
        n_extreme = (df_spikes_page["pct_diff"] > 100.0).sum()
        n_moderate = ((df_spikes_page["pct_diff"] > 30.0) & (df_spikes_page["pct_diff"] <= 100.0)).sum()
        n_normal = (df_spikes_page["spike_status"] == "NORMAL").sum()
        n_low_drop = (df_spikes_page["spike_status"] == "ANOMALI_RENDAH").sum()

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card" style="border-top: 3px solid #991B1B;">
                <div class="kpi-title">Lonjakan Ekstrem (&gt; 100%)</div>
                <div class="kpi-value" style="color: #991B1B;">{n_extreme:,}</div>
                <div class="kpi-sub">Prioritas Investigasi Segera</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #DC2626;">
                <div class="kpi-title">Lonjakan Moderat (+30% s.d +100%)</div>
                <div class="kpi-value" style="color: #DC2626;">{n_moderate:,}</div>
                <div class="kpi-sub">Verifikasi Catat Meter</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #16A34A;">
                <div class="kpi-title">Pemakaian Normal (-30% s.d +30%)</div>
                <div class="kpi-value" style="color: #16A34A;">{n_normal:,}</div>
                <div class="kpi-sub">Pola Konsumsi Wajar</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #D97706;">
                <div class="kpi-title">Anomali Rendah / Macet (&lt; -30%)</div>
                <div class="kpi-value" style="color: #D97706;">{n_low_drop:,}</div>
                <div class="kpi-sub">Potensi Meter Jammed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        tab_sp1, tab_sp2, tab_sp3 = st.tabs(["Daftar Pelanggan Terindikasi Lonjakan", "Anomali Rendah & Potensi Meter Macet", "Sebaran Besaran Kenaikan"])

        with tab_sp1:
            st.subheader("Daftar Pelanggan Mengalami Lonjakan Konsumsi Air (> +30%)")
            
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                min_spike_slider = st.slider("Minimal Persentase Kenaikan (%)", min_value=30, max_value=500, value=50, step=10)
            with f_col2:
                rayon_spike_filter = st.multiselect("Filter Rayon Wilayah", options=sorted(df_spikes_page["RAYON"].unique().tolist()), default=[])

            # Filter data
            df_spike_filtered = df_spikes_page[df_spikes_page["pct_diff"] >= min_spike_slider]
            if rayon_spike_filter:
                df_spike_filtered = df_spike_filtered[df_spike_filtered["RAYON"].isin(rayon_spike_filter)]

            st.write(f"Ditemukan **{len(df_spike_filtered):,}** pelanggan terindikasi lonjakan.")

            show_cols = ["NoPEL", "Nama", "Alamat", "RAYON", "actual", "prediction", "error_abs", "pct_diff", "recommendation_priority", "action_type"]
            show_cols_avail = [c for c in show_cols if c in df_spike_filtered.columns]
            st.dataframe(df_spike_filtered[show_cols_avail], use_container_width=True, height=400)

        with tab_sp2:
            st.subheader("Daftar Pelanggan Mengalami Penurunan Drastis / Potensi Meter Macet")
            df_low_filtered = df_spikes_page[df_spikes_page["spike_status"] == "ANOMALI_RENDAH"]
            st.write(f"Ditemukan **{len(df_low_filtered):,}** pelanggan dengan konsumsi di bawah normal.")
            st.dataframe(df_low_filtered[show_cols_avail], use_container_width=True, height=400)

        with tab_sp3:
            st.subheader("Distribusi Besaran Kenaikan & Selisih Konsumsi (m³)")
            fig_spike_hist = px.histogram(
                df_spikes_page[df_spikes_page["pct_diff"].between(-50, 300)],
                x="pct_diff",
                color="spike_status",
                nbins=60,
                title="Distribusi Persentase Selisih Pemakaian Relatif terhadap Prediksi (%)",
                labels={"pct_diff": "Persentase Selisih (%)", "spike_status": "Status"},
                color_discrete_map={
                    "INDIKASI_LONJAKAN": "#DC2626",
                    "NORMAL": "#16A34A",
                    "ANOMALI_RENDAH": "#D97706",
                },
                template="plotly_white",
            )
            fig_spike_hist.add_vline(x=30.0, line_dash="dash", line_color="#DC2626", annotation_text="Batas Lonjakan (+30%)")
            fig_spike_hist.add_vline(x=-30.0, line_dash="dash", line_color="#D97706", annotation_text="Batas Anomali Rendah (-30%)")
            st.plotly_chart(fig_spike_hist, use_container_width=True)
            st.markdown(render_chart_caption("Distribusi deviasi pemakaian air di luar batas wajar (±30%). Kenaikan melebihi +30% diklasifikasikan sebagai lonjakan pemakaian, sedangkan penurunan di bawah -30% mengindikasikan meter macet/rusak.", "#DC2626"), unsafe_allow_html=True)

    render_next_step_guide(4)


# =============================================================================
# HALAMAN 5: PETA SEBARAN PELANGGAN (FOLIUM INTERAKTIF)
# =============================================================================
elif selected_menu == "Peta Sebaran Pelanggan":
    st.markdown("""
    <div class="pdam-header">
        <h1>Peta Spasial Sebaran Pelanggan & Anomali Lonjakan</h1>
    </div>
    """, unsafe_allow_html=True)

    df_map_raw = load_cached_dashboard_data("ALL", "ALL")

    # Filter yang valid koordinatnya
    valid_geo = df_map_raw[
        df_map_raw["latitude"].notna() & 
        df_map_raw["longitude"].notna() & 
        (df_map_raw["latitude"] != 0) & 
        (df_map_raw["longitude"] != 0)
    ].copy()

    # Kontrol Filter Peta & Skema Visual
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns([1.2, 1.2, 1.2, 1.5, 1.2])
    with m_col1:
        map_rayon_opt = ["ALL"] + sorted([r for r in valid_geo["RAYON"].unique().tolist() if pd.notna(r)])
        map_rayon = st.selectbox("Wilayah / Rayon", map_rayon_opt, index=0)
    with m_col2:
        map_status_opt = ["ALL", "INDIKASI_LONJAKAN", "ANOMALI_RENDAH", "NORMAL"]
        map_status = st.selectbox("Status Lonjakan", map_status_opt, index=0)
    with m_col3:
        map_clust_opt = [
            "ALL",
            "Klaster 1 (Rendah - ABRI/Khusus)",
            "Klaster 2 (Sedang - Rumah Tangga)",
            "Klaster 3 (Tinggi - Niaga/Industri)"
        ]
        map_clust = st.selectbox("Filter Klaster", map_clust_opt, index=0)
    with m_col4:
        color_scheme_opt = [
            "Status Lonjakan (Merah/Amber/Hijau)",
            "Klaster K-Means (Hijau/Biru/Dongker)",
        ]
        selected_scheme = st.selectbox("Skema Warna Marker", color_scheme_opt, index=0)
    with m_col5:
        sample_limit = st.slider("Maksimal Titik", min_value=100, max_value=2500, value=500, step=100, help="Jumlah titik ditampilkan pada peta. 500 titik direkomendasikan untuk performa tercepat dan mulus.")

    # Terapkan filter
    filtered_geo = valid_geo.copy()
    if map_rayon != "ALL":
        filtered_geo = filtered_geo[filtered_geo["RAYON"] == map_rayon]
    if map_status != "ALL":
        filtered_geo = filtered_geo[filtered_geo["spike_status"] == map_status]
    if map_clust != "ALL":
        cid_filter = 0 if "1" in map_clust else (1 if "2" in map_clust else 2)
        filtered_geo = filtered_geo[filtered_geo["cluster_id"] == cid_filter]

    n_total_matched = len(filtered_geo)
    sample_geo = filtered_geo.head(sample_limit).copy()

    # Baris Legenda Visual & Info
    leg_col1, leg_col2 = st.columns([2.5, 1.5])
    with leg_col1:
        if selected_scheme.startswith("Status"):
            st.markdown("""
            <div style='background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 7px 14px; font-size: 0.84rem; display: flex; gap: 18px; align-items: center;'>
                <span style='font-weight: 600; color: #1E293B;'>Keterangan Status:</span>
                <span><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#DC2626; margin-right:4px;'></span><b style='color:#DC2626;'>Indikasi Lonjakan</b> (> +30%)</span>
                <span><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#D97706; margin-right:4px;'></span><b style='color:#D97706;'>Anomali Rendah</b> (< -30%)</span>
                <span><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#16A34A; margin-right:4px;'></span><b style='color:#16A34A;'>Normal</b> (Wajar)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 7px 14px; font-size: 0.84rem; display: flex; gap: 18px; align-items: center;'>
                <span style='font-weight: 600; color: #1E293B;'>Keterangan Klaster:</span>
                <span><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#16A34A; margin-right:4px;'></span><b style='color:#16A34A;'>Klaster 1</b>: Rendah (ABRI/Khusus)</span>
                <span><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#0284C7; margin-right:4px;'></span><b style='color:#0284C7;'>Klaster 2</b>: Sedang (Rumah Tangga)</span>
                <span><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#0F4C81; margin-right:4px;'></span><b style='color:#0F4C81;'>Klaster 3</b>: Tinggi (Niaga/Industri)</span>
            </div>
            """, unsafe_allow_html=True)
    with leg_col2:
        st.markdown(f"""
        <div style='text-align: right; padding-top: 8px; font-size: 0.84rem; color: #475569;'>
            Menampilkan <b>{len(sample_geo):,}</b> titik dari total <b>{n_total_matched:,}</b> pelanggan.
        </div>
        """, unsafe_allow_html=True)

    if sample_geo.empty:
        st.warning("Tidak ditemukan data pelanggan dengan koordinat valid untuk kombinasi filter ini.")
    else:
        # Bangun peta Folium interaktif multi-layer dengan sparkline popups (Cached for high performance)
        color_by_mode = "spike_status" if selected_scheme.startswith("Status") else "cluster"
        map_html = generate_optimized_map_html(sample_geo, color_by_mode)

        # Render peta Leaflet secara mulus
        components.html(map_html, height=660)

        st.caption("**Panduan Navigasi Peta:** Klik ikon layer di pojok kanan atas untuk berganti antara *Google Hybrid Satellite*, *OpenStreetMap*, *Esri Satellite*, atau *OpenTopoMap*. Klik setiap titik marker untuk melihat pop-up profil dan **grafik riwayat 12 periode konsumsi air** (Aktual vs Prediksi ANN vs Batas +30%). Titik yang berdekatan/sama telah didispersikan secara otomatis sehingga setiap pelanggan dapat diklik secara individual.")

        render_next_step_guide(5)


# =============================================================================
# HALAMAN 6: REKOMENDASI PDAM & EKSPOR SPK
# =============================================================================
elif selected_menu == "Rekomendasi PDAM":
    st.markdown("""
    <div class="pdam-header">
        <h1>Rekomendasi Operasional & Surat Perintah Kerja (SPK)</h1>
    </div>
    """, unsafe_allow_html=True)

    df_rec_page = load_cached_dashboard_data("ALL", "ALL")

    if df_rec_page.empty:
        st.info("Belum ada data pelanggan untuk rekomendasi operasional.")
    else:
        # Ringkasan prioritas
        p_counts = df_rec_page["recommendation_priority"].value_counts().to_dict()
        n_high = p_counts.get("TINGGI", 0)
        n_med = p_counts.get("SEDANG", 0)
        n_low = p_counts.get("RENDAH", 0)
        n_norm = p_counts.get("NORMAL", 0)

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card" style="border-top: 3px solid #DC2626;">
                <div class="kpi-title">SPK Investigasi Urgent (Tinggi)</div>
                <div class="kpi-value" style="color: #DC2626;">{n_high:,}</div>
                <div class="kpi-sub">SLA Tindak Lanjut: 3 Hari</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #D97706;">
                <div class="kpi-title">Verifikasi Catat Meter (Sedang)</div>
                <div class="kpi-value" style="color: #D97706;">{n_med:,}</div>
                <div class="kpi-sub">SLA Tindak Lanjut: 7 Hari</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #0284C7;">
                <div class="kpi-title">Monitoring Rutin (Rendah)</div>
                <div class="kpi-value" style="color: #0284C7;">{n_low:,}</div>
                <div class="kpi-sub">SLA Tindak Lanjut: 14 Hari</div>
            </div>
            <div class="kpi-card" style="border-top: 3px solid #16A34A;">
                <div class="kpi-title">Pembacaan Normal Terverifikasi</div>
                <div class="kpi-value" style="color: #16A34A;">{n_norm:,}</div>
                <div class="kpi-sub">Diterbitkan Langsung</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        tab_r1, tab_r2 = st.tabs(["Daftar Pelanggan Prioritas Tindakan", "Analisis Risiko per Rayon"])

        with tab_r1:
            st.subheader("Daftar Rekomendasi Tindakan Operasional Petugas")

            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                prio_filter = st.multiselect("Filter Prioritas Tindakan", ["TINGGI", "SEDANG", "RENDAH", "NORMAL"], default=["TINGGI", "SEDANG"])
            with r_col2:
                rayon_rec_filter = st.multiselect("Filter Rayon", options=sorted(df_rec_page["RAYON"].unique().tolist()), default=[], key="rec_rayon")

            df_prio = df_rec_page[df_rec_page["recommendation_priority"].isin(prio_filter)]
            if rayon_rec_filter:
                df_prio = df_prio[df_prio["RAYON"].isin(rayon_rec_filter)]

            st.write(f"Ditemukan **{len(df_prio):,}** rekomendasi tindakan.")

            rec_cols = [
                "NoPEL", "Nama", "Alamat", "RAYON", "actual", "prediction", "pct_diff",
                "recommendation_priority", "action_type", "recommendation_text"
            ]
            st.dataframe(df_prio[[c for c in rec_cols if c in df_prio.columns]], use_container_width=True, height=400)

            # Ekspor Dokumen SPK Excel
            st.markdown("#### Unduh Surat Perintah Kerja (SPK) Lapangan")

            spk_df = recommendation.export_spk_work_orders(df_prio, priority_filter=prio_filter)
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                spk_df.to_excel(writer, index=False, sheet_name="Daftar_SPK_Lapangan")
            excel_bytes = excel_buffer.getvalue()

            st.download_button(
                label="Unduh Format SPK Lapangan (.xlsx)",
                data=excel_bytes,
                file_name=f"SPK_PDAM_Investigasi_Lapangan_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

        with tab_r2:
            st.subheader("Ringkasan Sebaran Risiko Lonjakan Berdasarkan Rayon Wilayah")
            
            # Grouping rayon
            r_group = df_rec_page.groupby("RAYON").agg(
                Total_Pelanggan=("NoPEL", "count"),
                Lonjakan=("spike_status", lambda s: (s == "INDIKASI_LONJAKAN").sum()),
                Anomali_Rendah=("spike_status", lambda s: (s == "ANOMALI_RENDAH").sum()),
                Prioritas_Tinggi=("recommendation_priority", lambda p: (p == "TINGGI").sum()),
            ).reset_index()
            r_group["Persentase_Lonjakan_%"] = np.round((r_group["Lonjakan"] / r_group["Total_Pelanggan"]) * 100, 2)
            r_group = r_group.sort_values("Prioritas_Tinggi", ascending=False).reset_index(drop=True)

            col_rg1, col_rg2 = st.columns([1.5, 1])
            with col_rg1:
                st.dataframe(r_group, use_container_width=True, height=380)
            with col_rg2:
                top_r_plot = r_group.head(10)
                fig_rp = px.bar(
                    top_r_plot,
                    x="RAYON",
                    y="Prioritas_Tinggi",
                    title="Top 10 Rayon dengan Kasus Lonjakan Tinggi (SPK Urgent)",
                    labels={"RAYON": "Kode Rayon", "Prioritas_Tinggi": "Jumlah SPK Urgent"},
                    color="Prioritas_Tinggi",
                    color_continuous_scale="Reds",
                    template="plotly_white",
                )
                fig_rp.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_rp, use_container_width=True)
                st.markdown(render_chart_caption("Peringkat 10 rayon dengan kasus lonjakan prioritas tinggi terbanyak yang membutuhkan penerbitan Surat Perintah Kerja (SPK) inspeksi fisik segera ke petugas lapangan.", "#DC2626"), unsafe_allow_html=True)

        render_next_step_guide(6)

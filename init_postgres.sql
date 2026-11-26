-- =============================================================================
-- PDAM Skripsi Database Schema (PostgreSQL)
-- Sistem Monitoring Konsumsi Air Pelanggan PDAM
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'petugas',
    rayon VARCHAR(50) DEFAULT 'ALL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    NoPEL VARCHAR(20) PRIMARY KEY,
    Nama VARCHAR(255),
    Alamat TEXT,
    RAYON VARCHAR(50),
    source_area VARCHAR(100),
    tarif DOUBLE PRECISION DEFAULT 1.0,
    tarif_name VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    cluster_id INTEGER,
    cluster_label VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS readings (
    id SERIAL PRIMARY KEY,
    NoPEL VARCHAR(20) NOT NULL,
    Periode VARCHAR(20) NOT NULL,
    period_dt TIMESTAMP,
    Ini DOUBLE PRECISION NOT NULL,
    Lalu DOUBLE PRECISION NOT NULL,
    pemakaian_air DOUBLE PRECISION NOT NULL,
    source_file VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_readings_nopel_periode UNIQUE (NoPEL, Periode)
);

CREATE TABLE IF NOT EXISTS predictions_anomalies (
    id SERIAL PRIMARY KEY,
    NoPEL VARCHAR(20) NOT NULL,
    Periode VARCHAR(20) NOT NULL,
    cluster_id INTEGER,
    cluster_label VARCHAR(100),
    actual DOUBLE PRECISION,
    prediction DOUBLE PRECISION NOT NULL,
    error_abs DOUBLE PRECISION,
    pct_diff DOUBLE PRECISION,
    spike_status VARCHAR(50) NOT NULL,
    severity_score DOUBLE PRECISION,
    model_version VARCHAR(50) DEFAULT 'ANN-S1-k3',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_predictions_nopel_periode UNIQUE (NoPEL, Periode)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    NoPEL VARCHAR(20) NOT NULL,
    Periode VARCHAR(20) NOT NULL,
    spike_status VARCHAR(50) NOT NULL,
    priority VARCHAR(50) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    recommendation_text TEXT NOT NULL,
    status_action VARCHAR(50) DEFAULT 'OPEN',
    assigned_to VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_recommendations_nopel_periode UNIQUE (NoPEL, Periode)
);

CREATE TABLE IF NOT EXISTS processing_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    process_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    records_processed INTEGER DEFAULT 0,
    duplicates_skipped INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    message TEXT,
    user_info VARCHAR(100) DEFAULT 'SYSTEM'
);

-- Indeks Performa
CREATE INDEX IF NOT EXISTS idx_pg_readings_nopel ON readings(NoPEL);
CREATE INDEX IF NOT EXISTS idx_pg_readings_periode ON readings(Periode);
CREATE INDEX IF NOT EXISTS idx_pg_predictions_nopel ON predictions_anomalies(NoPEL);
CREATE INDEX IF NOT EXISTS idx_pg_predictions_spike ON predictions_anomalies(spike_status);
CREATE INDEX IF NOT EXISTS idx_pg_customers_rayon ON customers(RAYON);

-- Default Users: admin / admin123, petugas / pdam2026
INSERT INTO users (username, password_hash, full_name, role, rayon)
VALUES 
    ('admin', '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9', 'Administrator PDAM', 'admin', 'ALL'),
    ('petugas', 'f34fbe9fd488ff8db237addaf068439963f4e276b24ab06b31dc4ef343a848b4', 'Petugas Lapangan Utama', 'petugas', 'ALL'),
    ('petugas_purbalingga', 'f34fbe9fd488ff8db237addaf068439963f4e276b24ab06b31dc4ef343a848b4', 'Petugas Rayon Purbalingga', 'petugas', '010001'),
    ('petugas_bukateja', 'f34fbe9fd488ff8db237addaf068439963f4e276b24ab06b31dc4ef343a848b4', 'Petugas Rayon Bukateja', 'petugas', '020001')
ON CONFLICT (username) DO NOTHING;

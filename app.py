import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Sistem Prediksi Diabetes - Klinik Medika",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button {
        background-color: #0ea5e9;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #0284c7; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def ambil_model():
    try:
        with open('model_linear_regression.pkl', 'rb') as f:
            mdl_regresi = pickle.load(f)
        with open('scaler_linear_regression.pkl', 'rb') as f:
            skl_regresi = pickle.load(f)
        with open('model_naive_bayes_smote.pkl', 'rb') as f:
            mdl_nb = pickle.load(f)
        with open('scaler_classification.pkl', 'rb') as f:
            skl_nb = pickle.load(f)
        return mdl_regresi, skl_regresi, mdl_nb, skl_nb
    except Exception as err:
        st.error(f"Gagal memuat model: {err}")
        st.stop()

mdl_regresi, skl_regresi, mdl_nb, skl_nb = ambil_model()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8254/8254580.png", width=80)
    st.markdown("## 🏥 Klinik Medika")
    st.markdown("Sistem prediksi berbasis Machine Learning untuk mendukung keputusan klinis.")
    st.markdown("---")
    modul = st.radio(
        "📂 Pilih Modul:",
        ["📈 Prediksi Kadar Glukosa", "🩺 Deteksi Risiko Diabetes"]
    )
    st.markdown("---")
    st.caption("© 2026 Klinik Medika | Data Mining Project")

# ── MODUL 1: REGRESI ──────────────────────────────────────────────
if modul == "📈 Prediksi Kadar Glukosa":
    st.title("📈 Prediksi Kadar Glukosa Darah")
    st.info("Isi data klinis pasien di bawah ini untuk memperkirakan kadar glukosa darah (tanpa variabel Glucose & Outcome).")
    st.markdown("---")

    with st.form("formulir_regresi"):
        st.subheader("🗂️ Data Klinis Pasien")
        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown("**Informasi Umum**")
            usia        = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=40)
            kehamilan   = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1)
            bmi         = st.number_input("BMI (× 100)", min_value=1000, value=2500, help="Contoh: BMI 25.0 → isi 2500")
            tek_darah   = st.number_input("Tekanan Darah", min_value=40, value=80)
            hba1c       = st.number_input("HbA1c (× 10)", min_value=10, value=55, help="Contoh: HbA1c 5.5 → isi 55")

        with k2:
            st.markdown("**Profil Lipid**")
            ldl          = st.number_input("Kadar LDL", min_value=10, value=100)
            hdl          = st.number_input("Kadar HDL", min_value=10, value=50)
            trigliserida = st.number_input("Trigliserida", min_value=10, value=150)
            pinggang     = st.number_input("Lingkar Pinggang (cm)", min_value=30, value=85)
            pinggul      = st.number_input("Lingkar Pinggul (cm)", min_value=30, value=95)

        with k3:
            st.markdown("**Riwayat Kesehatan**")
            whr          = st.number_input("WHR (× 100)", min_value=10, value=89, help="Contoh: WHR 0.89 → isi 89")
            riwayat_kel  = st.selectbox("Riwayat Keluarga Diabetes", [0, 1], format_func=lambda x: "Tidak Ada" if x == 0 else "Ada")
            pola_makan   = st.selectbox("Pola Makan", [0, 1], format_func=lambda x: "Sehat" if x == 0 else "Kurang Sehat")
            hipertensi   = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            konsumsi_obat = st.selectbox("Konsumsi Obat-obatan", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

        st.markdown("---")
        tombol_regresi = st.form_submit_button("🔍 Hitung Estimasi Glukosa")

    if tombol_regresi:
        data_masuk = np.array([[usia, kehamilan, bmi, tek_darah, hba1c,
                                ldl, hdl, trigliserida, pinggang, pinggul,
                                whr, riwayat_kel, pola_makan, hipertensi, konsumsi_obat]])
        data_ternormalisasi = skl_regresi.transform(data_masuk)
        nilai_glukosa       = mdl_regresi.predict(data_ternormalisasi)

        st.markdown("---")
        st.success("✅ Prediksi berhasil!")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.metric(label="🩸 Estimasi Kadar Glukosa Darah", value=f"{nilai_glukosa[0]:.2f} mg/dL")

# ── MODUL 2: KLASIFIKASI ──────────────────────────────────────────
elif modul == "🩺 Deteksi Risiko Diabetes":
    st.title("🩺 Deteksi Dini Risiko Diabetes")
    st.info("Isi rekam medis lengkap pasien untuk menganalisis kemungkinan risiko diabetes.")
    st.markdown("---")

    with st.form("formulir_klasifikasi"):
        st.subheader("🗂️ Rekam Medis Pasien")
        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown("**Informasi Umum**")
            usia        = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45)
            kehamilan   = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=2)
            bmi         = st.number_input("BMI (× 100)", min_value=1000, value=2800)
            glukosa     = st.number_input("Kadar Glukosa", min_value=10, value=140)
            tek_darah   = st.number_input("Tekanan Darah", min_value=40, value=85)
            hba1c       = st.number_input("HbA1c (× 10)", min_value=10, value=60)

        with k2:
            st.markdown("**Profil Lipid**")
            ldl          = st.number_input("Kadar LDL", min_value=10, value=120)
            hdl          = st.number_input("Kadar HDL", min_value=10, value=45)
            trigliserida = st.number_input("Trigliserida", min_value=10, value=180)
            pinggang     = st.number_input("Lingkar Pinggang (cm)", min_value=30, value=90)
            pinggul      = st.number_input("Lingkar Pinggul (cm)", min_value=30, value=100)

        with k3:
            st.markdown("**Riwayat Kesehatan**")
            whr          = st.number_input("WHR (× 100)", min_value=10, value=90)
            riwayat_kel  = st.selectbox("Riwayat Keluarga Diabetes", [0, 1], format_func=lambda x: "Tidak Ada" if x == 0 else "Ada")
            pola_makan   = st.selectbox("Pola Makan", [0, 1], format_func=lambda x: "Sehat" if x == 0 else "Kurang Sehat")
            hipertensi   = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            konsumsi_obat = st.selectbox("Konsumsi Obat-obatan", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

        st.markdown("---")
        tombol_klasifikasi = st.form_submit_button("🔬 Analisis Risiko Diabetes")

    if tombol_klasifikasi:
        data_masuk = np.array([[usia, kehamilan, bmi, glukosa, tek_darah, hba1c,
                                ldl, hdl, trigliserida, pinggang, pinggul,
                                whr, riwayat_kel, pola_makan, hipertensi, konsumsi_obat]])
        data_ternormalisasi = skl_nb.transform(data_masuk)
        hasil_deteksi       = mdl_nb.predict(data_ternormalisasi)

        st.markdown("---")
        if hasil_deteksi[0] == 1:
            st.error("⚠️ HASIL: PASIEN BERISIKO TINGGI DIABETES")
            st.warning("Rekomendasi: Segera jadwalkan pemeriksaan HbA1c lanjutan dan konsultasi dokter spesialis endokrin.")
        else:
            st.success("✅ HASIL: PASIEN TIDAK BERISIKO DIABETES")
            st.info("Rekomendasi: Kondisi pasien dalam batas normal. Anjurkan untuk mempertahankan pola hidup sehat.")

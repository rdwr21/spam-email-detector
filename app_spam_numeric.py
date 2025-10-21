# ======================================================
# STREAMLIT APP - DETEKSI SPAM (FIXED UNTUK SCALER)
# ======================================================

import streamlit as st
import numpy as np
import joblib

# 1. Muat Model & Scaler
model = joblib.load("model_spam_naive_bayes.pkl")
scaler = joblib.load("scaler.pkl")

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Spam Email", page_icon="📊", layout="centered")
st.title("📊 Aplikasi Deteksi Spam (Dataset UCI Spambase)")
st.markdown("Masukkan beberapa nilai fitur utama di bawah ini untuk memprediksi apakah email termasuk **Spam** atau **Non-Spam**.")

# 3. Input Beberapa Fitur Penting
st.subheader("🧩 Input Nilai Fitur:")
word_freq_free = st.number_input("Frekuensi kata 'free' (word_freq_free):", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
word_freq_money = st.number_input("Frekuensi kata 'money' (word_freq_money):", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
word_freq_receive = st.number_input("Frekuensi kata 'receive' (word_freq_receive):", min_value=0.0, max_value=10.0, value=0.3, step=0.1)
capital_run_length_average = st.number_input("Rata-rata huruf kapital (capital_run_length_average):", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
capital_run_length_total = st.number_input("Total huruf kapital (capital_run_length_total):", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)

# 4. Prediksi
if st.button("🔍 Deteksi Sekarang"):
    # Buat dummy input dengan jumlah fitur sama seperti model
    input_data = np.zeros((1, scaler.mean_.shape[0]))  # isi semua 0
    # isi beberapa fitur penting sesuai posisi (urutan di dataset)
    input_data[0, 16] = word_freq_free         # kolom ke-17 = word_freq_free
    input_data[0, 52] = word_freq_money        # kolom ke-53 = word_freq_money
    input_data[0, 50] = word_freq_receive      # kolom ke-51 = word_freq_receive
    input_data[0, 55] = capital_run_length_average
    input_data[0, 56] = capital_run_length_total

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # 5. Hasil
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 **Hasil: SPAM** – Email ini terindikasi sebagai spam.")
    else:
        st.success("✅ **Hasil: NON-SPAM** – Email ini aman dan tidak terdeteksi sebagai spam.")
    st.markdown("---")

# 6. Footer
st.caption("Dikembangkan oleh: **Ridwan & Tim AI 05TPLE007** | Model: Naive Bayes | Dataset: UCI Spambase")


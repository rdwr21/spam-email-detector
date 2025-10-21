# ======================================================
# STREAMLIT APP - DETEKSI SPAM (DATA NUMERIK UCI SPAMBASE)
# ======================================================

import streamlit as st
import numpy as np
import joblib

# 1. Muat Model & Scaler
model = joblib.load("model_spam_naive_bayes.pkl")
scaler = joblib.load("scaler.pkl")

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Spam Email", page_icon="📊", layout="centered")
st.title("📊 Aplikasi Deteksi Spam (Dataset Numerik UCI)")
st.markdown("Masukkan nilai fitur di bawah ini untuk memprediksi apakah email termasuk **Spam** atau **Non-Spam**.")

# 3. Input Fitur
st.subheader("🧩 Masukkan Nilai Fitur Penting:")

# Beberapa fitur utama dari dataset (bisa dikembangkan lagi)
word_freq_free = st.number_input("Frekuensi kata 'free' (word_freq_free):", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
word_freq_money = st.number_input("Frekuensi kata 'money' (word_freq_money):", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
word_freq_receive = st.number_input("Frekuensi kata 'receive' (word_freq_receive):", min_value=0.0, max_value=10.0, value=0.3, step=0.1)
capital_run_length_average = st.number_input("Rata-rata huruf kapital (capital_run_length_average):", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
capital_run_length_total = st.number_input("Total huruf kapital (capital_run_length_total):", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)

# 4. Prediksi
if st.button("🔍 Deteksi Sekarang"):
    # Susun input ke dalam array sesuai urutan fitur
    # Untuk kesederhanaan, hanya sebagian fitur dipakai
    input_data = np.array([[word_freq_free, word_freq_money, word_freq_receive, capital_run_length_average, capital_run_length_total]])

    # Lakukan scaling menggunakan scaler hasil training
    input_scaled = scaler.transform(input_data)

    # Prediksi dengan model Naive Bayes
    prediction = model.predict(input_scaled)[0]

    # 5. Tampilkan Hasil
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 **Hasil: SPAM** – Email ini terindikasi sebagai spam.")
    else:
        st.success("✅ **Hasil: NON-SPAM** – Email ini aman dan tidak terdeteksi sebagai spam.")
    st.markdown("---")

# 6. Footer
st.caption("Dikembangkan oleh: **Ridwan** | Model: Naive Bayes | Dataset: UCI Spambase")

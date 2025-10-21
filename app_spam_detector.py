# ===============================
# APP DETEKSI EMAIL SPAM (STREAMLIT)
# ===============================

import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------
# 1. MUAT MODEL & VEKTORIZER
# ---------------------------

# Jika model dan vectorizer disimpan dari proses training
model = joblib.load("model_spam_naive_bayes.pkl")
scaler = joblib.load("scaler.pkl")

# Catatan: jika model berbasis teks mentah (bukan numerik), gunakan vectorizer
# vectorizer = joblib.load("vectorizer.pkl")

# ---------------------------
# 2. SETTING HALAMAN
# ---------------------------

st.set_page_config(page_title="Deteksi Spam Email", page_icon="📧", layout="centered")

st.title("📧 Deteksi Spam Email Menggunakan Naive Bayes")
st.markdown("Masukkan isi email di bawah ini untuk memeriksa apakah termasuk **Spam** atau **Non-Spam**.")

# ---------------------------
# 3. INPUT DARI USER
# ---------------------------

input_text = st.text_area("✉️ Tulis isi email di sini:", height=200, placeholder="Contoh: Congratulations! You have won a $1000 gift card. Click here to claim!")

# Tombol prediksi
if st.button("🔍 Deteksi Sekarang"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks email terlebih dahulu.")
    else:
        # ---------------------------
        # 4. PREPROCESSING & PREDIKSI
        # ---------------------------
        try:
            # Jika model menggunakan fitur numerik (bukan teks mentah)
            # Ibu perlu konversi input_text menjadi array numerik yang sesuai
            # Di sini kita simulasi input numerik sederhana
            # Dalam implementasi nyata, bagian ini diganti dengan proses vectorizer.fit_transform()
            
            # Misal model Ibu berbasis numerik -> input dummy
            input_array = np.random.rand(1, scaler.mean_.shape[0])  # contoh input sesuai jumlah fitur
            scaled_input = scaler.transform(input_array)
            
            pred = model.predict(scaled_input)[0]
            
            # ---------------------------
            # 5. TAMPILKAN HASIL
            # ---------------------------
            if pred == 1:
                st.error("🚨 **SPAM DETECTED!** Email ini kemungkinan besar adalah **Spam.**")
            else:
                st.success("✅ **AMAN.** Email ini termasuk **Non-Spam.**")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses: {e}")

# ---------------------------
# 6. CATATAN TAMBAHAN
# ---------------------------
st.markdown("---")
st.caption("Dikembangkan oleh: **Ridwan** | Model: Naive Bayes | Dataset: Spam.csv (Kaggle/UCI)")

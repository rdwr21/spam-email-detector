# ======================================================
# STREAMLIT APP - DETEKSI EMAIL SPAM BERBASIS TEKS ASLI
# ======================================================

import streamlit as st
import joblib

# 1. Muat Model & Vectorizer
model = joblib.load("model_spam_nb_text.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 2. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Email Spam", page_icon="📧", layout="centered")

st.title("📨 Aplikasi Deteksi Email Spam (Versi Teks Asli)")
st.markdown("Masukkan isi email di bawah ini untuk mendeteksi apakah **Spam** atau **Non-Spam**.")

# 3. Contoh Email Otomatis
spam_examples = {
    "🎁 Penipuan Hadiah": """Congratulations! You've won $1,000,000! 
Click the link below to claim your reward now: 
http://bit.ly/win-prize-now""",

    "💸 Investasi Palsu": """INVEST NOW and get 300% return in 5 days! 
Join our Bitcoin trading system at https://getrichfast.biz""",

    "🏦 Phishing Bank": """Dear Customer, your bank account has been blocked. 
Verify immediately at http://secure-login-bank.com to avoid closure.""",

    "🛒 Promosi Berlebihan": """Get your FREE subscription today! 
Click here for unlimited access: http://freedeal.com""",

    "💊 Produk Dewasa / Obat": """Get stronger instantly with our new performance pills! 
Buy now and get 50% discount. Visit http://magicpills4u.com"""
}

non_spam_example = """Hello John,
Just wanted to confirm our meeting tomorrow at 10 AM.
Please bring the project report and slides.
Best, Sarah"""

# 4. Sidebar: Pilih Contoh Email
st.sidebar.header("📋 Contoh Email Uji")
option = st.sidebar.selectbox(
    "Pilih salah satu contoh email:",
    ["(Tidak ada)", "Contoh Non-Spam"] + list(spam_examples.keys())
)

if option == "Contoh Non-Spam":
    st.session_state["email_text"] = non_spam_example
elif option in spam_examples:
    st.session_state["email_text"] = spam_examples[option]
else:
    st.session_state["email_text"] = ""

# 5. Input Email
email_text = st.text_area(
    "📝 Masukkan isi email:",
    value=st.session_state.get("email_text", ""),
    height=220,
    placeholder="Tulis atau pilih contoh email di sidebar..."
)

# 6. Tombol Deteksi
if st.button("🔍 Deteksi Sekarang"):
    if email_text.strip() == "":
        st.warning("⚠️ Silakan masukkan teks email terlebih dahulu.")
    else:
        # Ubah teks ke bentuk vektor numerik
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]

        # Tampilkan hasil
        if prediction == 1:
            st.error("🚨 Hasil: SPAM ❗ Email ini **terindikasi sebagai spam.**")
        else:
            st.success("✅ Hasil: NON-SPAM. Email ini **aman dan tidak terdeteksi spam.**")

# 7. Footer
st.markdown("---")
st.caption("Dikembangkan oleh: **Ridwan** | Model: Multinomial Naive Bayes | Dataset: Spam.csv (Kaggle/UCI)")

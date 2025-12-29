import streamlit as st

st.title("💻 Panduan Cara Menggunakan Gree")
st.divider()

st.subheader("1. Buka Sidebar")
st.write("Buka sidebar pada bagian kiri halaman dan pilih menu **Mulai Prediksi**")
st.write("")

st.subheader("2. Pilih Mata Uang")
st.write("Pilih salah satu mata uang yang ingin Anda prediksi")
st.write("")

st.subheader("3. Tentukan Rentang Prediksi")
st.write("Pilih rentang waktu prediksi yang Anda inginkan")
st.write("")

st.subheader("4. Jalankan Prediksi")
st.write("Klik tombol **Prediksi** untuk memulai analisis")

st.divider()

# Info box untuk hasil
st.info("📊 **Hasil Prediksi** — Nilai penutupan dan volatilitas akan ditampilkan pada halaman utama Gree setelah proses selesai.")
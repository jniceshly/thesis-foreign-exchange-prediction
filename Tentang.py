import streamlit as st

st.set_page_config(
    page_title="Sistem Prediksi Forex",
    page_icon="💱",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: left;'>💱 Gree: Sistem Prediksi Harga Penutupan Valuta Asing</h1>
    """,
    unsafe_allow_html=True
)

st.divider()

st.subheader("Tentang Gree")

st.markdown("""
    Gree adalah sebuah sistem yang dirancang untuk memberikan **informasi prediktif** mengenai **harga penutupan valuta asing terhadap Rupiah (IDR)**.
    
    Nilai tukar yang dapat diprediksi dengan Gree adalah 3 nilai tukar yang sering dijadikan sarana investasi oleh masyarakat Indonesia (sumber: *https://www.dbs.id/*). Nilai tukar tersebut yaitu:
    1. USD/IDR
    2. EUR/IDR
    3. GBP/IDR
        
    Gree mengombinasikan **pendekatan statistik** untuk membantu pengguna memahami pergerakan nilai tukar serta tingkat risiko fluktuasinya.
    """
)

st.write("")
st.write("")
st.subheader("Fitur Utama")

st.markdown("##### 📈 Prediksi Harga Penutupan")
st.markdown("""
    Gree menyediakan fitur prediksi harga penutupan
    untuk pasangan mata uang USD/IDR, EUR/IDR, dan GBP/IDR.
    Prediksi dapat dilakukan untuk 1 hari ke depan (hari kerja).
""")

st.write("")
st.write("")
st.subheader("Sumber Data")

st.markdown("""
    **1. Data Harga Penutupan Valuta Asing**
            
    - USD/IDR  
    - EUR/IDR  
    - GBP/IDR  
            
    Sumber: *Investing*  
    🔗 https://www.investing.com/
    """)

st.write("")

st.markdown("""
    **2. Variabel Eksternal**

    - Inflasi  
    - Suku Bunga (BI Rate)  

    Sumber: *Bank Indonesia*  
    🔗 https://www.bi.go.id/id/
    """)

st.info("Variabel eksternal berperan sebagai faktor makroekonomi " \
"tambahan untuk meningkatkan akurasi prediksi harga penutupan valuta asing.""")

st.write("")
st.write("")
st.subheader("Model Prediksi yang Digunakan")

st.markdown("##### 📈 ARIMAX")

st.markdown(
    """
    Autoregressive Integrated Moving Average with Exogenous Variable (ARIMAX) digunakan untuk
    memprediksi harga penutupan valuta asing.
    """
)

st.divider()
st.caption(
    "📌 Sistem ini dikembangkan sebagai bagian dari tugas akhir akademik di bidang" 
    " Data Science. Informasi dan hasil prediksi yang ditampilkan bersifat informatif" 
    " dan bukan merupakan panduan, rekomendasi, atau ajakan"
    " untuk melakukan investasi atau transaksi valuta asing."
)

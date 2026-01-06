import sys
import os
import io
import streamlit as st
# from streamlit_gsheets import GSheetsConnection
from statsmodels.tsa.statespace.sarimax import SARIMAX
st.set_page_config(layout="wide")

try:
    from arch import arch_model
    import numba
except Exception as e:
    print(f"Import error: {e}")
    
import pandas as pd
import numpy
import numpy as np
import pickle as pkl
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import datetime
from datetime import date, timedelta
import joblib

import holidays
from pandas.tseries.offsets import CustomBusinessDay

if 'current_currency' not in st.session_state:
    st.session_state.current_currency = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# tgl merah indo
years = range(2025, 2026)
id_holidays = holidays.Indonesia(years=years)
holiday_dates = pd.to_datetime(list(id_holidays.keys()))

# custom business day: exclude weekend + tgl merah indo
custom_bd = CustomBusinessDay(holidays=holiday_dates)

bulan_mapping = {
    'Januari': 'January',
    'Februari': 'February',
    'Maret': 'March',
    'April': 'April',
    'Mei': 'May',
    'Juni': 'June',
    'Juli': 'July',
    'Agustus': 'August',
    'September': 'September',
    'Oktober': 'October',
    'November': 'November',
    'Desember': 'December'
}
month_map = {
    'Januari': 'January', 'Februari': 'February', 'Maret': 'March',
    'April': 'April', 'Mei': 'May', 'Juni': 'June',
    'Juli': 'July', 'Agustus': 'August', 'September': 'September',
    'Oktober': 'October', 'November': 'November', 'Desember': 'December'
}

######### LOAD DATA
def load_usd():
    df = pd.read_excel("USD_IDR_Investing.xlsx")
    df = df.drop(0, axis=0)
    df = df.rename(columns={'USD_IDR Historical Data':'Date'})
    df = df.rename(columns={'Unnamed: 1':'Close Price'})
    df = df.rename(columns={'Unnamed: 2':'Open'})
    df = df.rename(columns={'Unnamed: 3':'High'})
    df = df.rename(columns={'Unnamed: 4':'Low'})
    df = df.rename(columns={'Unnamed: 5':'Vol.'})
    df = df.rename(columns={'Unnamed: 6':'Change %'})
    df = df.drop(['Open', 'High', 'Low','Vol.', 'Change %'], axis=1)
    df = df.rename(columns={'USD_IDR Historical Data':'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df.index = df['Date']
    df = df.drop("Date", axis=1)
    df = df.sort_index(ascending=True)
    df['Close Price'] = (df['Close Price'].str.replace(',', '', regex=False).astype(float))
    return df

def load_eur():
    forex_data = pd.read_excel("EUR_IDR_Investing.xlsx")
    forex_data.drop(0, axis=0)
    forex_data = forex_data.rename(columns={'EUR_IDR Historical Data':'Date'})
    forex_data = forex_data.rename(columns={'Unnamed: 1':'Close Price'})
    forex_data = forex_data.rename(columns={'Unnamed: 2':'Open'})
    forex_data = forex_data.rename(columns={'Unnamed: 3':'High'})
    forex_data = forex_data.rename(columns={'Unnamed: 4':'Low'})
    forex_data = forex_data.rename(columns={'Unnamed: 5':'Vol.'})
    forex_data = forex_data.rename(columns={'Unnamed: 6':'Change %'})
    forex_data = forex_data.drop(0, axis=0)
    forex_data = forex_data.drop(['Open', 'High', 'Low','Vol.', 'Change %'], axis=1)
    forex_data['Date'] = pd.to_datetime(forex_data['Date'])
    forex_data.index = forex_data['Date']
    forex_data = forex_data.drop(['Date'], axis=1)
    df = forex_data
    df = df.sort_index(ascending=True)
    df['Close Price'] = (df['Close Price'].str.replace(',', '', regex=False).astype(float))
    return df

def load_gbp():
    forex_data = pd.read_excel("GBP_IDR_Investing.xlsx")
    forex_data.drop(0, axis=0)
    forex_data = forex_data.rename(columns={'GBP_IDR Historical Data':'Date'})
    forex_data = forex_data.rename(columns={'Unnamed: 1':'Close Price'})
    forex_data = forex_data.rename(columns={'Unnamed: 2':'Open'})
    forex_data = forex_data.rename(columns={'Unnamed: 3':'High'})
    forex_data = forex_data.rename(columns={'Unnamed: 4':'Low'})
    forex_data = forex_data.rename(columns={'Unnamed: 5':'Vol.'})
    forex_data = forex_data.rename(columns={'Unnamed: 6':'Change %'})
    forex_data = forex_data.drop(0, axis=0)
    forex_data = forex_data.drop(['Open', 'High', 'Low','Vol.', 'Change %'], axis=1)
    forex_data['Date'] = pd.to_datetime(forex_data['Date'])
    forex_data.index = forex_data['Date']
    forex_data = forex_data.drop(['Date'], axis=1)
    forex_data = forex_data.sort_index(ascending=True)
    df = forex_data
    df['Close Price'] = (df['Close Price'].str.replace(',', '', regex=False).astype(float))
    return df

SHEET_ID = "1rObGLsEYcWcHkpqHVhVfHtvPfnI15-npOsl3bj3a5us"

@st.cache_data(ttl=300)
def load_sheet(gid):
    url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
        f"?format=csv&gid={gid}"
    )

    df = pd.read_csv(
        url,
        header=0,          # paksa header
        usecols=[0, 1],    # cuma 2 kolom
        skip_blank_lines=True
    )

    df = df.dropna()      # buang baris kosong
    df = df.reset_index(drop=True)

    return df

df_new_usd = load_sheet(gid="0")
df_new_eur = load_sheet(gid="753250280")
df_new_gbp = load_sheet(gid="1222767202")
df_new_inflasi = load_sheet(gid="1835176075")
df_new_caddev = load_sheet(gid="453134674")
df_new_birate = load_sheet(gid="2095066760")

def load_usd_latest():
    df_usd = df_new_usd.copy()
    df_usd.columns = ["Date", "Close Price"]
    df_usd["Date"] = pd.to_datetime(df_usd["Date"], errors="coerce")
    df_usd = df_usd.dropna(subset=["Date"])
    df_usd["Close Price"] = (
        df_usd["Close Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df_usd["Close Price"] = pd.to_numeric(df_usd["Close Price"], errors="coerce")
    df_usd = df_usd.dropna()
    df_usd = df_usd.set_index("Date").sort_index()
    return df_usd

def load_eur_latest():
    df_eur = df_new_eur.copy()
    df_eur.columns = ["Date", "Close Price"]
    df_eur["Date"] = pd.to_datetime(df_eur["Date"], errors="coerce")
    df_eur = df_eur.dropna(subset=["Date"])
    df_eur['Close Price'] = (df_eur['Close Price'].str.replace(',', '', regex=False).astype(float))
    df_eur["Close Price"] = (
        df_eur["Close Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df_eur["Close Price"] = pd.to_numeric(df_eur["Close Price"], errors="coerce")
    df_eur = df_eur.dropna()
    df_eur = df_eur.set_index("Date").sort_index()
    return df_eur

def load_gbp_latest():
    df_gbp = df_new_gbp.copy()
    df_gbp.columns = ["Date", "Close Price"]
    df_gbp["Date"] = pd.to_datetime(df_gbp["Date"], errors="coerce")
    df_gbp = df_gbp.dropna(subset=["Date"])
    df_gbp['Close Price'] = (df_gbp['Close Price'].str.replace(',', '', regex=False).astype(float))
    df_gbp["Close Price"] = (
        df_gbp["Close Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df_gbp["Close Price"] = pd.to_numeric(df_gbp["Close Price"], errors="coerce")
    df_gbp = df_gbp.dropna()
    df_gbp = df_gbp.set_index("Date").sort_index()
    return df_gbp

def combine_usd():
    df_hist = load_usd()
    df_new = load_usd_latest()
    df = pd.concat([df_hist, df_new])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df

def combine_eur():
    df_hist = load_eur()
    df_new = load_eur_latest()
    df = pd.concat([df_hist, df_new])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df

def combine_gbp():
    df_hist = load_gbp()
    df_new = load_gbp_latest()
    df = pd.concat([df_hist, df_new])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


df_map = {
    'USD/IDR': combine_usd,
    'EUR/IDR': combine_eur,
    'GBP/IDR': combine_gbp
}
st.header("Analisis Historis dengan Indikator Teknikal")
st.write("")
st.write("")

currency = st.radio("Pilih satu mata uang yang ingin dilihat analisisnya",["USD/IDR", "EUR/IDR", "GBP/IDR"], horizontal=True)

if st.session_state.current_currency != currency:
    st.session_state.current_df = None
    st.session_state.current_currency = currency
    st.session_state.show_analysis = False

if currency == "USD/IDR":
    df = combine_usd()
    st.session_state.current_df = df
    choice = 1
    
elif currency == "EUR/IDR":
    df = combine_eur()
    st.session_state.current_df = df
    choice = 3

elif currency == "GBP/IDR":
    df = combine_gbp()
    st.session_state.current_df = df
    choice = 5

def calculate_macd(df, fast=12, slow=26, signal=9):
    df_macd = df.copy()
    
    # Calculate EMAs
    ema_fast = df_macd['Close Price'].ewm(span=fast, adjust=False).mean()
    ema_slow = df_macd['Close Price'].ewm(span=slow, adjust=False).mean()
    
    # MACD Line
    df_macd['MACD'] = ema_fast - ema_slow
    
    # Signal Line
    df_macd['Signal'] = df_macd['MACD'].ewm(span=signal, adjust=False).mean()
    
    # Histogram
    df_macd['Histogram'] = df_macd['MACD'] - df_macd['Signal']
    
    return df_macd


def plot_macd(df, n_days):
    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=n_days)
    
    # Filter data berdasarkan range yang dipilih
    df_filtered = df.loc[df.index >= start_date]
    
    # Calculate MACD
    df_macd = calculate_macd(df_filtered)
    
    # Create figure
    fig = go.Figure()
    
    # MACD Line
    fig.add_trace(go.Scatter(
        x=df_macd.index,
        y=df_macd['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    # Signal Line
    fig.add_trace(go.Scatter(
        x=df_macd.index,
        y=df_macd['Signal'],
        mode='lines',
        name='Signal',
        line=dict(color='red', width=2)
    ))
    
    # Histogram
    colors = ['green' if val >= 0 else 'red' for val in df_macd['Histogram']]
    fig.add_trace(go.Bar(
        x=df_macd.index,
        y=df_macd['Histogram'],
        name='Histogram',
        marker_color=colors,
        opacity=0.5
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        template="plotly_white",
        title={
            'text': "MACD (Moving Average Convergence Divergence)",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="lightgray", title="MACD Value"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def calculate_rsi(df, period=14):
    df_rsi = df.copy()
    
    delta = df_rsi['Close Price'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df_rsi['RSI'] = 100 - (100 / (1 + rs))
    
    return df_rsi

# Fungsi untuk plot RSI
def plot_rsi(df, n_days, period=14):
    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=n_days)
    
    # Filter data berdasarkan range yang dipilih
    df_filtered = df.loc[df.index >= start_date]
    
    # Calculate RSI
    df_rsi = calculate_rsi(df_filtered, period=period)
    
    # Create figure
    fig = go.Figure()
    
    # RSI Line
    fig.add_trace(go.Scatter(
        x=df_rsi.index,
        y=df_rsi['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.1)'
    ))
    
    # Overbought line (70)
    fig.add_hline(
        y=70, 
        line_dash="dash", 
        line_color="red", 
        opacity=0.7,
        annotation_text="Overbought (70)",
        annotation_position="right"
    )
    
    # Oversold line (30)
    fig.add_hline(
        y=30, 
        line_dash="dash", 
        line_color="green", 
        opacity=0.7,
        annotation_text="Oversold (30)",
        annotation_position="right"
    )
    
    # Middle line (50)
    fig.add_hline(
        y=50, 
        line_dash="dot", 
        line_color="gray", 
        opacity=0.5
    )
    
    # Add shaded regions
    fig.add_hrect(
        y0=70, y1=100, 
        fillcolor="red", 
        opacity=0.1, 
        line_width=0
    )
    fig.add_hrect(
        y0=0, y1=30, 
        fillcolor="green", 
        opacity=0.1, 
        line_width=0
    )
    
    fig.update_layout(
        template="plotly_white",
        title={
            'text': f"RSI (Relative Strength Index) - Period {period}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        xaxis=dict(showgrid=False, title="Tanggal"),
        yaxis=dict(
            showgrid=True, 
            gridcolor="lightgray", 
            title="RSI Value",
            range=[0, 100]
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk memberikan sinyal RSI
def get_rsi_signal(df):
    df_rsi = calculate_rsi(df)
    latest_rsi = df_rsi['RSI'].iloc[-1]
    
    if pd.isna(latest_rsi):
        return None, "Data tidak cukup untuk menghitung RSI"
    
    if latest_rsi >= 70:
        signal = "🔴 Overbought - Potensi Jual"
        color = "red"
    elif latest_rsi <= 30:
        signal = "🟢 Oversold - Potensi Beli"
        color = "green"
    else:
        signal = "⚪ Netral"
        color = "gray"
    
    return latest_rsi, signal, color


def choose_plot_range():
    st.sidebar.markdown("### 📊 Pilih Rentang Visualisasi")
    range_option = st.sidebar.radio(
    "Pilih rentang visualisasi",
    [
        "1 Minggu Terakhir",
        "2 Minggu Terakhir",
        "1 Bulan Terakhir",
        "3 Bulan Terakhir",
        "6 Bulan Terakhir",
        "1 Tahun Terakhir"
    ], index=0,
    horizontal=True, key="range_option")

    range_mapping = {
        "1 Minggu Terakhir": 7,
        "2 Minggu Terakhir": 14,
        "1 Bulan Terakhir": 30,
        "3 Bulan Terakhir": 90,
        "6 Bulan Terakhir": 180,
        "1 Tahun Terakhir": 365
    }
    n_days = range_mapping[range_option]
    return n_days

st.sidebar.write("")

if st.button("📈 Lihat Analisis"):
    st.session_state.show_analysis = True

if st.session_state.show_analysis:
    st.write("---")
    st.subheader("📈 Indikator Teknikal", help ="Indikator teknikal merupakan hasil analisis berdasarkan data historis, bukan hasil prediksi nilai di masa datang.")
    
    tab1, tab2 = st.tabs(["MACD", "RSI"])
    n_days = choose_plot_range()

    with tab1:
        plot_macd(st.session_state.current_df, n_days)
        with st.container(border=True):
            st.markdown("**MACD** mengukur momentum dengan membandingkan dua moving average untuk mengidentifikasi sinyal beli/jual.")
            st.markdown("""
                        Parameter MACD yang digunakan dalam analisis adalah: \
                        
                        - Fast EMA (Exponential Moving Average) 12 periode
                        - Slow EMA 26 periode
                        - Signal Line (EMA dari MACD Line) 9 periode")
                        """)
            st.write("")
            with st.expander("📚 Pelajari lebih lanjut tentang MACD"):
                st.markdown("""
                **MACD (Moving Average Convergence Divergence)** terdiri dari:
                
                1. **MACD Line (Biru)**: Selisih antara EMA 12-hari dan 26-hari
                2. **Signal Line (Merah)**: EMA 9-hari dari MACD Line
                3. **Histogram**: Visualisasi perbedaan MACD dan Signal
                
                **Sinyal:**
                - 🟢 **Golden Cross**: MACD memotong Signal dari bawah → Sinyal BELI
                - 🔴 **Death Cross**: MACD memotong Signal dari atas → Sinyal JUAL
                - Histogram yang membesar → Momentum menguat
                - Histogram yang mengecil → Momentum melemah
                
                **MACD lebih baik jika digabungkan dengan analisis:**
                - Konfirmasi dengan volume trading
                - Perhatikan divergence (harga naik tapi MACD turun)
                - Gunakan bersama indikator lain seperti RSI
                """)
    
    with tab2:
        latest_rsi, signal, color = get_rsi_signal(st.session_state.current_df)
        
        if latest_rsi:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("RSI Saat Ini", f"{latest_rsi:.2f}")
            with col2:
                if color == "red":
                    st.error(signal)
                elif color == "green":
                    st.success(signal)
                else:
                    st.info(signal)
        
        plot_rsi(st.session_state.current_df, n_days)
        with st.container(border=True):
            st.markdown("**RSI** mengukur kecepatan pergerakan harga (0-100) untuk mendeteksi kondisi overbought/oversold.")
            st.markdown("""
                        Periode perhitungan RSI yang digunakan dalam analisis ini adalah 14 hari.
                        """)
            st.info("Jika garis tidak muncul pada grafik, artinya rentang waktu yang dipilih masih kurang dari 14 hari.")
            st.write("")
            with st.expander("📚 Pelajari lebih lanjut tentang RSI"):
                st.markdown("""
                **RSI (Relative Strength Index)** menunjukkan momentum harga:
                
                **Level Kritis:**
                - **RSI > 70**: Overbought (jenuh beli)
                - Harga mungkin sudah terlalu tinggi
                - Potensi koreksi/penurunan
                
                - **RSI 30-70**: Zona Netral
                - Kondisi normal
                - Tidak ada tekanan ekstrim
                
                - **RSI < 30**: Oversold (jenuh jual)
                - Harga mungkin sudah terlalu rendah
                - Potensi rebound/kenaikan
                
                **Sinyal:**
                1. **Basic**: Beli saat RSI < 30, Jual saat RSI > 70
                2. **Advanced**: Cari divergence untuk sinyal pembalikan tren
                3. **Confirmation**: Kombinasikan dengan MACD dan trendline
                
                **Catatan:**
                - Dalam tren kuat, RSI bisa tetap di zona overbought/oversold lama
                - Jangan trading hanya berdasarkan RSI saja
                - Period 14 hari adalah standar, tapi bisa disesuaikan
                """)


            

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

######### LOAD MODEL
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pkl.load(f)
    return model

def load_model_vol_arimax(path):
    model_arimax = joblib.load(path)
    return model_arimax


def load_model_vol_garchx(path):
    model_garchx = joblib.load(path)
    
    return model_garchx

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

def exog_birate():
    int_rate = pd.read_excel("BI-7Day-RR.xlsx", header=4)
    int_rate = int_rate.drop(['NO','Unnamed: 3'],axis=1)
    for indo, eng in bulan_mapping.items():
        int_rate['Tanggal'] = int_rate['Tanggal'].str.replace(indo, eng)
            
    int_rate.rename(columns={"Tanggal": "Date","BI-7Day-RR": "BI Rate",},inplace=True)
    int_rate['Date'] = pd.to_datetime(int_rate['Date'])
    int_rate['BI Rate'] = int_rate['BI Rate'].str.replace('%', '').astype(float)
    int_rate = int_rate.set_index('Date')
    int_rate = int_rate.sort_index(ascending=True)
        
    #today = date.today()
    #int_rate = int_rate.loc['2019-12-31':]
    full_index = pd.date_range(start = int_rate.index.min(), end = int_rate.index.max(), freq='D')
    interest_daily = int_rate.reindex(full_index)
    interest_daily = interest_daily.ffill()

    return interest_daily
#st.dataframe(exog_birate())

def exog_inflasi():
    inflasi = pd.read_excel('Data Inflasi.xlsx')
    inflasi = inflasi.iloc[4:]
    inflasi = inflasi.drop(columns=['Unnamed: 0', 'Unnamed: 3'])
    inflasi = inflasi.rename(columns={
        'Unnamed: 1': 'Date',
        'Unnamed: 2': 'Inflasi'
    })

    inflasi['Inflasi'] = inflasi['Inflasi'].str.replace('%', '').astype(float)
    inflasi['Date'] = inflasi['Date'].astype(str)
    inflasi['Date'] = inflasi['Date'].replace(month_map, regex=True)
    inflasi['Date'] = pd.to_datetime(inflasi['Date'], format="%B %Y", errors='coerce')
    inflasi.dropna(subset=['Date'], inplace=True)
    inflasi.set_index('Date', inplace=True)
    inflasi = inflasi.sort_index()
    last_date = inflasi.index.max()
    end_month = last_date.to_period('M').to_timestamp('M')
    
    inflation_daily = (inflasi.reindex(pd.date_range(inflasi.index.min(), end_month, freq="D")).ffill())
    return inflation_daily

def exog_devisa():
    cad_devisa = pd.read_excel("Cadangan Devisa-BI.xlsx")
    cad_devisa = cad_devisa.set_index(cad_devisa['Tanggal'])
    cad_devisa = cad_devisa.drop('Tanggal', axis=1)
    cad_devisa.index = cad_devisa.index.to_period('M').to_timestamp()
    cad_devisa = cad_devisa.rename(columns={'Cadangan Devisa (konsep IRFCL) 4) dalam USD':'Cadangan Devisa'})
    last_date = cad_devisa.index.max()
    end_month = last_date.to_period('M').to_timestamp('M')
    
    devisa_daily = (cad_devisa.reindex(pd.date_range(cad_devisa.index.min(), end_month, freq="D")).ffill())
    return devisa_daily
#st.dataframe(exog_devisa())

def merge_exog():
    inflasi = exog_inflasi()
    birate = exog_birate()
    devisa = exog_devisa()
    exog = pd.merge(inflasi, birate, left_index=True, right_index=True, how="inner")
    exog = pd.merge(exog, devisa, left_index=True, right_index=True, how="inner")
    return exog
#st.dataframe(merge_exog())

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

# url_usd = "https://docs.google.com/spreadsheets/d/1rObGLsEYcWcHkpqHVhVfHtvPfnI15-npOsl3bj3a5us/edit?usp=sharing"

# conn_usd = st.connection("gsheets", type=GSheetsConnection)

# df_new_usd = conn_usd.read(spreadsheet=url_usd, usecols=[0,1], worksheet="0", ttl=5)
# df_new_eur = conn_usd.read(spreadsheet=url_usd, usecols=[0,1], worksheet="753250280", ttl=5)
# df_new_gbp = conn_usd.read(spreadsheet=url_usd, usecols=[0,1], worksheet="1222767202", ttl=5)
# df_new_inflasi = conn_usd.read(spreadsheet=url_usd, usecols=[0,1], worksheet="1835176075",  ttl=5)
# df_new_caddev = conn_usd.read(spreadsheet=url_usd, usecols=[0,1], worksheet="453134674", ttl=5)
# df_new_birate = conn_usd.read(spreadsheet=url_usd, usecols=[0,1], worksheet="2095066760", ttl=5)

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

def exog_inflasi_latest():
    df_inf = df_new_inflasi.copy()
    df_inf.columns = ["Date", "Inflasi"]
    df_inf["Date"] = pd.to_datetime(df_inf["Date"], errors="coerce")
    df_inf = df_inf.dropna(subset=["Date"])
    df_inf["Inflasi"] = (df_inf["Inflasi"].str.replace('%','',regex=False).astype(float))
    df_inf = df_inf.set_index("Date").sort_index()

    df_inf_daily = (df_inf.reindex(pd.date_range(df_inf.index.min(), pd.Timestamp.today().normalize(), freq="D")).ffill())

    return df_inf_daily

def exog_birate_latest():
    df_rate = df_new_birate.copy()
    df_rate.columns = ["Date", "BI Rate"]
    df_rate["Date"] = pd.to_datetime(df_rate["Date"], errors="coerce")
    df_rate = df_rate.dropna(subset=["Date"])
    df_rate["BI Rate"] = (df_rate["BI Rate"].str.replace("%", '',regex=False).astype(float))
    df_rate = df_rate.set_index("Date").sort_index()
    df_rate_daily = (df_rate.reindex(pd.date_range(df_rate.index.min(), pd.Timestamp.today().normalize(), freq="D")).ffill())

    return df_rate_daily

def exog_devisa_latest():
    df_dev = df_new_caddev.copy()
    df_dev.columns = ["Date", "Cadangan Devisa"]
    df_dev["Date"] = pd.to_datetime(df_dev["Date"], errors="coerce")
    df_dev = df_dev.dropna(subset=["Date"])
    df_dev["Cadangan Devisa"] = df_dev["Cadangan Devisa"].str.replace(",",'',regex=False).astype(float)
    df_dev = df_dev.set_index("Date").sort_index()

    df_dev_daily = (df_dev.reindex(pd.date_range(df_dev.index.min(), pd.Timestamp.today().normalize(), freq="D")).ffill())

    return df_dev_daily

def merge_exog_latest():
    inflasi = exog_inflasi_latest()
    birate = exog_birate_latest()
    devisa = exog_devisa_latest()

    exog = inflasi.join(birate, how="outer").join(devisa, how="outer")
    exog = exog.sort_index().ffill()
    return exog

def combine_exog():
    df_hist = merge_exog()
    df_new = merge_exog_latest()
    df_combine = pd.concat([df_hist, df_new])
    df_combine = df_combine[~df_combine.index.duplicated(keep="last")]
    df_combine = df_combine.sort_index()
    return df_combine

######### PRICE FORECASTING
def forecast_price(model, exog=None, steps=1):
    forecast = model.forecast(steps=steps, exog=exog)
    return forecast


def plot_forex(df, df_forecast, step):
    df_close = pd.DataFrame(df['Close Price']).copy()
    df_close = df_close.reset_index()   # ensure Date is a column
    df_close.rename(columns={"index": "Date"}, inplace=True)

    last_date = df_close["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=step, freq='B')

    fig = go.Figure()
    
    # past
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close Price"],
        mode="lines",
        name="Historis",
        line=dict(color="blue", width=2),
    ))
    
    # CI
    fig.add_trace(go.Scatter(
        x=list(df_forecast["Date"]) + list(df_forecast["Date"][::-1]),
        y=list(df_forecast["Upper CI"]) + list(df_forecast["Lower CI"][::-1]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.35)',
        line=dict(color='rgba(255,0,0,0)'),
        name="Confidence Interval"
    ))
    
    # forecast
    fig.add_trace(
       go.Scatter(
         x=df_forecast["Date"],
         y=df_forecast["Forecast"],
         mode="lines+markers+text",
         name="Hasil Prediksi",
        line=dict(color="red", width=2, dash="dash"),
         marker=dict(color="red", size=6),
         textposition="top center",
         hovertemplate="Tanggal: %{x}<br>Prediksi: Rp %{y:,.3f}<extra></extra>"))

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template = "plotly_white",
        title={'text': "Data Historis 2020-2025 dan Hasil Prediksi Harga Penutupan",
               'x': 0.5,
               'xanchor':'center',
               'yanchor':'top',
               'font': dict(size=18)},
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        hovermode="x unified",
        legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="center",
          x=0.5),
       height=450)

    st.plotly_chart(fig, use_container_width=True)


def plot_volatility(df, forecast_df, future_dates):
    df["log_return"] = np.log(df["Close Price"] / df["Close Price"].shift(1))
    df["hist_vol"] = df["log_return"].rolling(window=22).std()  
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["hist_vol"],
        mode="lines",
        name="Historical",
        line=dict(color="blue", width=2)
    )) # past data

    # CI
    fig.add_trace(go.Scatter(
        x=list(forecast_df["Date"]) + list(forecast_df["Date"][::-1]),
        y=list(forecast_df["Upper CI"]) + list(forecast_df["Lower CI"][::-1]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.35)',
        line=dict(color='rgba(255,0,0,0)'),
        name="Confidence Interval"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Vol Forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=6)
    )) # forecast

    fig.update_layout(
        title={
            'text':"Data Historis 2020-2025 dan Hasil Prediksi Volatilitas",
            'x':0.5, 'xanchor':'center', 'yanchor':'top', 'font':dict(size=18)},
        xaxis_title="Tahun",
        yaxis_title="Volatilitas",
        template="plotly_white",
        hovermode="x unified",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    return fig

st.header("Hasil Prediksi")
st.write("---")

st.sidebar.subheader("Pilih Mata Uang")
currency = st.sidebar.radio("Pilih satu yang ingin dilihat prediksinya",["USD/IDR", "EUR/IDR", "GBP/IDR"])

st.sidebar.markdown("")
st.sidebar.markdown("### Tanggal Penutupan Terakhir")

df_map = {
    'USD/IDR': combine_usd,
    'EUR/IDR': combine_eur,
    'GBP/IDR': combine_gbp
}

last_date = df_map[currency]().index[-1]
st.sidebar.write(last_date.strftime('%d-%m-%Y'))

st.sidebar.markdown("")
st.sidebar.markdown("### Pilih Rentang Prediksi")
range_option = st.sidebar.radio("Rentang prediksi ke depan (business day)", ["1 hari ke depan", "5 hari ke depan"])

if currency == "USD/IDR" and range_option == "1 hari ke depan":
    p = 2
    d = 1
    q = 2
    p_vol = 1
    d_vol = 0
    q_vol = 0
    p_gar = 1
    q_gar = 1
    step = 1
    df = combine_usd()
    choice = 1
    
elif currency == "USD/IDR" and range_option == "5 hari ke depan":
    step = 5
    p = 1
    d = 1
    q = 1
    p_vol = 1
    d_vol = 0
    q_vol = 0
    p_gar = 1
    q_gar = 1
    df = combine_usd()
    choice = 2
    
elif currency == "EUR/IDR" and range_option == "1 hari ke depan":
    step = 1
    p = 1
    d = 1
    q = 2
    p_vol = 0
    d_vol = 0
    q_vol = 1
    p_gar = 1
    q_gar = 2
    df = combine_eur()
    choice = 3

elif currency == "EUR/IDR" and range_option == "5 hari ke depan":
    step = 5
    p = 2
    d = 1
    q = 1
    p_vol = 0
    d_vol = 0
    q_vol = 1
    p_gar = 2
    q_gar = 2
    df = combine_eur()
    choice = 4

elif currency == "GBP/IDR" and range_option == "1 hari ke depan":
    step = 1
    window = 30
    p = 1
    d = 1
    q = 1
    p_vol = 1
    d_vol = 0
    q_vol = 0
    p_gar = 1
    q_gar = 2
    df = combine_gbp()
    choice = 5
    
elif currency == "GBP/IDR" and range_option == "5 hari ke depan":
    step = 5
    p = 1
    d = 1
    q = 2
    p_vol = 1
    d_vol = 0
    q_vol = 0
    p_gar = 2
    q_gar = 2
    df = combine_gbp()
    choice = 6

def info_ci_price():
    st.info("Tingkat kepercayaan atau confidence interval pada prediksi harga penutupan menunjukkan rentang nilai di mana harga penutupan yang sebenarnya kemungkinan besar akan berada.")

def info_ci_vol():
    st.info("Tingkat kepercayaan atau confidence interval pada volatilitas menunjukkan rentang nilai di mana volatilitas sebenarnya kemungkinan besar berada.")
    
def arimax_1_horizon(df, exog, p,d,q, step,currency):
    exog = exog.reindex(df.index)
    last_date = df.index[-1]
    
    model = SARIMAX(df["Close Price"],
                   exog=exog,
                   order=(p,d,q),
                   enforce_stationary=False,
                enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    future_exog = exog.tail(step)
    forecast = model_fit.forecast(steps=step, exog=future_exog)
    
    # CI
    forecast_obj = model_fit.get_forecast(steps=step, exog=future_exog)
    mean_forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]
    
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=df.index[-1] + timedelta(days=1), periods=step,
        freq='B'
    )
    
    forecast = np.array(forecast).flatten()
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": mean_forecast,
        "Lower CI": lower,
        "Upper CI": upper
    })
    
    last_price = df['Close Price'].iloc[-1]
    next_price = forecast[-1]

    expected_return = ((next_price - last_price) / last_price) * 100
    
    st.header(f"1️⃣ Prediksi Close Price {currency}")
    st.write("")
    st.write("")
    
    perubahan_prediksi = next_price - last_price
    perubahan_persen = (perubahan_prediksi / last_price) * 100
    upper_ci = forecast_df["Upper CI"].values[0]
    lower_ci = forecast_df["Lower CI"].values[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("##### Harga Hari Ini")
        st.metric(
            label=last_date.strftime('%d-%m-%Y'),
            value=f"Rp {last_price:,.2f}"
        )

    with col2:
        st.markdown("##### Hasil Prediksi")
        st.metric(
            label=f"H+1 ({future_dates[0].strftime('%d-%m-%Y')})",
            value=f"Rp {next_price:,.2f}",
            delta=f"Perubahan: Rp.{perubahan_prediksi:,.2f} ({perubahan_persen:,.2f}%)",
            delta_color="normal" if perubahan_prediksi >= 0 else "inverse"
        )
    
    with col3:
        st.markdown("##### Batas Atas")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {upper_ci:,.2f}"
        )
    with col4:
        st.markdown("##### Batas Bawah")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {lower_ci:,.2f}"
        )
        
    st.write("")
    info_ci_price()
    
    plot_forex(df, forecast_df, step)
    st.write(f"**Expected Return:** {perubahan_persen:.3f}%")

    if expected_return > 0:
        st.markdown("**Sinyal:** 🟢 Harga penutupan lebih rendah hari ini dibandingkan hasil prediksi, potensi return diprediksikan lebih tinggi.")
    else:
        st.markdown("**Sinyal:** 🔴 Harga penutupan lebih tinggi hari ini dibandingkan hasil prediksi, potensi return diprediksikan lebih rendah.")
    
    st.divider()
    st.write(" ")    


def arimaxgarchx_1_horizon(df, exog, p_vol, d_vol, q_vol, p_gar, q_gar, step, currency):
    exog = exog.reindex(df.index)

    last_date = df.index[-1]
    df['price_diff'] = np.log(df['Close Price'] / df["Close Price"].shift(1))

    model = SARIMAX(df["price_diff"],
                   exog=exog,
                   order=(p_vol, d_vol, q_vol),
                   enforce_stationary=False,
                    enforce_invertibility=False)
    
    arimax_model = model.fit(disp=False)
    
    future_exog = exog.tail(step)
    arimax_forecast = arimax_model.forecast(steps=step, exog=future_exog)

    # GARCHX Forecast for Volatility
    residuals = arimax_model.resid
    residuals = residuals.dropna()
    garch = arch_model(residuals, vol="GARCH", p = p_gar, o = 0, q = q_gar,
                               dist = "normal")
    garch_fit = garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=step)
    
    # variance forecast need to be converted to daily volatility
    variance = garch_forecast.variance.values[-1]
    vol = np.sqrt(variance)
    
    lower_vol = vol * 0.95
    upper_vol = vol * 1.05
    
    # create future business dates
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=step,
        freq="B"
    )
    
    result = pd.DataFrame({
        "Date" : future_dates,
        "Mean Return Forecast": arimax_forecast.values,
        "Vol Forecast": vol,
        "Upper CI": upper_vol,
        "Lower CI": lower_vol
    })

    st.header(f"2️⃣ Prediksi Volatilitas {currency}")
    st.write("")
    
    df["hist_vol_daily"] = df["price_diff"].rolling(22).std()
    last_vol = df["hist_vol_daily"].iloc[-1]

    diff = result['Vol Forecast'].values[0] - last_vol
    percentage_diff = (diff / last_vol)*100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("##### Volatilitas Harian Terakhir")
        st.metric(
            label=f"{last_date.strftime('%d-%m-%Y')}",
            value=f"Rp {last_vol:,.5f}"
        )

    with col2:
        st.markdown("##### Hasil Prediksi")
        st.metric(
            label=f"H+1 ({future_dates[0].strftime('%d-%m-%Y')})",
            value=f"Rp {vol[-1]:,.5f}",
            delta=f"Perubahan: {percentage_diff:,.2f}%",
            delta_color="normal" if diff >= 0 else "inverse"
        )
    
    with col3:
        st.markdown("##### Batas Atas")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {upper_vol[0]:,.5f}"
        )
    with col4:
        st.markdown("##### Batas Bawah")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {lower_vol[0]:,.5f}"
        )
        
    st.write("")
    info_ci_vol()
        
    fig = plot_volatility(df, result, future_dates)
    st.plotly_chart(fig, use_container_width=True)

    if vol < last_vol:
        st.markdown("**Sinyal:** 🟢 Volatilitas lebih tinggi hari ini dibandingkan hasil prediksi, potensi resiko diprediksikan menurun")
    else:
        st.markdown("**Sinyal:** 🔴 Volatilitas lebih rendah hari ini dibandingkan hasil prediksi, potensi resiko diprediksikan meningkat")

    
def arimax_5_horizon(df, exog, p,d,q, step,currency):
    exog = exog.reindex(df.index)
    model = SARIMAX(df["Close Price"],
                   exog=exog,
                   order=(p,d,q),
                   enforce_stationary=False,
                   enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    future_exog = exog.tail(step)

    # CI
    forecast_obj = model_fit.get_forecast(steps=step, exog=future_exog)
    mean_forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1),
                            periods=step, freq='B')
    
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": mean_forecast,
        "Lower CI": lower,
        "Upper CI": upper
    })
    
    last_price = df['Close Price'].iloc[-1]
        
    st.header(f"1️⃣ Prediksi Close Price {currency}")
    st.write("")
    st.write("")

    next_price =  forecast_df["Forecast"].iloc[0] 
    perubahan_prediksi = next_price - last_price
    perubahan_persen = (perubahan_prediksi / last_price) * 100
    upper_ci = forecast_df["Upper CI"].values[0]
    lower_ci = forecast_df["Lower CI"].values[0]

    next_price_2 = forecast_df["Forecast"].iloc[1] 
    next_price_3 = forecast_df["Forecast"].iloc[2] 
    next_price_4 = forecast_df["Forecast"].iloc[3] 
    next_price_5 = forecast_df["Forecast"].iloc[4] 
    perubahan_prediksi_2 = []
    perubahan_persen_2 =[]
    upper_ci_2 =[]
    lower_ci_2=[]
    
    for i in range(1,len(forecast_df)):
        perubahan = forecast_df["Forecast"].iloc[i] - forecast_df["Forecast"].iloc[i-1]
        persentase = (perubahan / forecast_df["Forecast"].iloc[i-1]) * 100

        perubahan_prediksi_2.append(perubahan)
        perubahan_persen_2.append(persentase)
        upper_ci_2.append(forecast_df["Upper CI"].iloc[i])
        lower_ci_2.append(forecast_df["Lower CI"].iloc[i])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("##### Harga Hari Ini")
        st.metric(
            label=f"{last_date.date()}",
            value=f"Rp {last_price:,.2f}"
        )

    with col2:
        st.markdown("##### Hasil Prediksi")
        st.metric(
            label=f"H+1 ({future_dates[0].date()})",
            value=f"Rp {next_price:,.2f}",
            delta=f"Perubahan: Rp.{perubahan_prediksi:,.2f} ({perubahan_persen:,.2f}%)",
            delta_color="normal" if perubahan_prediksi >= 0 else "inverse"
        )
        st.metric(
            label=f"H+2 ({future_dates[1].date()})",
            value=f"Rp {next_price_2:,.2f}",
            delta=f"Perubahan: Rp.{perubahan_prediksi_2[0]:,.2f} ({perubahan_persen_2[0]:,.2f}%)",
            delta_color="normal" if perubahan_prediksi >= 0 else "inverse"
        )
        st.metric(
            label=f"H+3 ({future_dates[2].date()})",
            value=f"Rp {next_price_3:,.2f}",
            delta=f"Perubahan: Rp.{perubahan_prediksi_2[1]:,.2f} ({perubahan_persen_2[1]:,.2f}%)",
            delta_color="normal" if perubahan_prediksi >= 0 else "inverse"
        )
        st.metric(
            label=f"H+4 ({future_dates[3].date()})",
            value=f"Rp {next_price_4:,.2f}",
            delta=f"Perubahan: Rp.{perubahan_prediksi_2[2]:,.2f} ({perubahan_persen_2[2]:,.2f}%)",
            delta_color="normal" if perubahan_prediksi >= 0 else "inverse"
        )
        st.metric(
            label=f"H+5 ({future_dates[4].date()})",
            value=f"Rp {next_price_5:,.2f}",
            delta=f"Perubahan: Rp.{perubahan_prediksi_2[3]:,.2f} ({perubahan_persen_2[3]:,.2f}%)",
            delta_color="normal" if perubahan_prediksi >= 0 else "inverse"
        )
    
    with col3:
        st.markdown("##### Batas Atas")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {upper_ci:,.2f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[0]:,.2f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[1]:,.2f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[2]:,.2f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[3]:,.2f}"
        )

        
    with col4:
        st.markdown("##### Batas Bawah")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {lower_ci:,.2f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[0]:,.2f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[1]:,.2f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[2]:,.2f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[3]:,.2f}"
        )

    st.write("")
    info_ci_price()
    plot_forex(df, forecast_df, step)
    
    if perubahan_persen > 0:
        st.markdown("**Sinyal:** 🟢 Harga penutupan lebih rendah hari ini dibandingkan hasil prediksi, potensi return diprediksikan lebih tinggi.")
    else:
        st.markdown("**Sinyal:** 🔴 Harga penutupan lebih tinggi hari ini dibandingkan hasil prediksi, potensi return diprediksikan lebih rendah.")
    
    st.write("---")
    st.write(" ")
    st.write(" ")


def arimax_5_horizon_vol(df, exog, p_vol, d_vol, q_vol, p_gar, q_gar, step,currency):
    exog = exog.reindex(df.index)

    last_date = df.index[-1]
    df['price_diff'] = np.log(df['Close Price'] / df["Close Price"].shift(1))

    model = SARIMAX(df["price_diff"],
                    exog=exog,
                    order=(p_vol, d_vol, q_vol),
                    enforce_stationary=False,
                    enforce_invertibility=False)
    
    arimax_model = model.fit(disp=False)
    
    # ARIMAX Forecast for Mean 
    future_exog = exog.tail(step)
    arimax_forecast = arimax_model.forecast(steps=step, exog=future_exog)

    # GARCHX Forecast for Volatility
    residuals = arimax_model.resid
    residuals = residuals.dropna()
    garch = arch_model(residuals, vol="GARCH", p = p_gar, o = 0, q = q_gar,
                               dist = "normal")
    garch_fit = garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=step)
    
    # variance forecast need to be converted to daily volatility
    variance = garch_forecast.variance.values[-1]
    vol = np.sqrt(variance)
    
    lower_vol = vol * 0.95
    upper_vol = vol * 1.05
    
    # create future business dates
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=step,
        freq="B"
    )
    
    result = pd.DataFrame({
        "Date" : future_dates,
        "Mean Return Forecast": arimax_forecast.values,
        "Vol Forecast": vol,
        "Upper CI": upper_vol,
        "Lower CI": lower_vol
    })

    st.header(f"2️⃣ Prediksi Volatilitas {currency}")
    st.write("")
    
    df["hist_vol_daily"] = df["price_diff"].rolling(22).std()
    last_vol = df["hist_vol_daily"].iloc[-1]

    diff = result['Vol Forecast'].values[0] - last_vol
    percentage_diff = (diff / last_vol)*100

    next_vol =  result["Vol Forecast"].iloc[0] 
    perubahan_prediksi = next_vol - last_vol
    perubahan_persen = (perubahan_prediksi / last_vol) * 100
    upper_ci = result["Upper CI"].values[0]
    lower_ci = result["Lower CI"].values[0]

    next_vol_2 = result["Vol Forecast"].iloc[1] 
    next_vol_3 = result["Vol Forecast"].iloc[2] 
    next_vol_4 = result["Vol Forecast"].iloc[3] 
    next_vol_5 = result["Vol Forecast"].iloc[4] 
    perubahan_prediksi_2 = []
    perubahan_persen_2 =[]
    upper_ci_2 =[]
    lower_ci_2=[]
    
    for i in range(1,len(result)):
        perubahan = result["Vol Forecast"].iloc[i] - result["Vol Forecast"].iloc[i-1]
        persentase = (perubahan / result["Vol Forecast"].iloc[i-1]) * 100

        perubahan_prediksi_2.append(perubahan)
        perubahan_persen_2.append(persentase)
        upper_ci_2.append(result["Upper CI"].iloc[i])
        lower_ci_2.append(result["Lower CI"].iloc[i])

    perubahan_persen_2 = [float(i) for i in perubahan_persen_2]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("##### Volatilitas Harian Terakhir")
        st.metric(
            label=f"{last_date.date()}",
            value=f"Rp {last_vol:,.5f}"
        )

    with col2:
        st.markdown("##### Hasil Prediksi")
        st.metric(
            label=f"H+1 ({future_dates[0].date()})",
            value=f"Rp {next_vol:,.5f}",
            delta=f"Perubahan: {perubahan_persen:,.3f}%",
            delta_color="normal" if perubahan_persen >= 0 else "inverse" #perli ganti warnanya
        )
        st.metric(
            label=f"H+2 ({future_dates[1].date()})",
            value=f"Rp {next_vol_2:,.5f}",
            delta=f"Perubahan: {perubahan_persen_2[0]:,.3f}%",
            delta_color="normal" if perubahan_persen_2[0] >= 0 else "inverse"
        )
        st.metric(
            label=f"H+3 ({future_dates[2].date()})",
            value=f"Rp {next_vol_3:,.5f}",
            delta=f"Perubahan: {perubahan_persen_2[1]:,.3f}%",
            delta_color="normal" if perubahan_persen_2[1] >= 0 else "inverse"
        )
        st.metric(
            label=f"H+4 ({future_dates[3].date()})",
            value=f"Rp {next_vol_4:,.5f}",
            delta=f"Perubahan: {perubahan_persen_2[2]:,.3f}%",
            delta_color="normal" if perubahan_persen_2[2] >= 0 else "inverse"
        )
        st.metric(
            label=f"H+5 ({future_dates[4].date()})",
            value=f"Rp {next_vol_5:,.5f}",
            delta=f"Perubahan: {perubahan_persen_2[3]:,.3f}%",
            delta_color="normal" if perubahan_persen_2[3] >= 0 else "inverse"
        )
    
    with col3:
        st.markdown("##### Batas Atas")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {upper_ci:,.5f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[0]:,.5f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[1]:,.5f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[2]:,.5f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {upper_ci_2[3]:,.5f}"
        )

        
    with col4:
        st.markdown("##### Batas Bawah")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {lower_ci:,.5f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[0]:,.5f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[1]:,.5f}"
        )
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[2]:,.5f}"
        )
        st.write("")
        st.write("")
        st.metric(
            label="",
            value=f"Rp {lower_ci_2[3]:,.5f}"
        )

    st.write("")
    info_ci_vol()
        
    fig = plot_volatility(df, result, future_dates)
    st.plotly_chart(fig, use_container_width=True)

    if next_vol < last_vol:
        st.markdown("**Sinyal:** 🟢 Volatilitas lebih tinggi hari ini dibandingkan hasil prediksi, potensi resiko diprediksikan menurun")
    else:
        st.markdown("**Sinyal:** 🔴 Volatilitas lebih rendah hari ini dibandingkan hasil prediksi, potensi resiko diprediksikan meningkat")

st.sidebar.markdown("")
if st.sidebar.button("🔮 Prediksi"):
    exog = combine_exog()
    with st.spinner("Predicting..."):
        if choice == 1 or choice==3 or choice == 5:
            arimax_1_horizon(df, exog, p, d, q, step, currency)
            arimaxgarchx_1_horizon(df, exog, p_vol, d_vol, q_vol, p_gar, q_gar, step, currency)
            
        elif choice == 2 or choice == 4 or choice==6:
            arimax_5_horizon(df, exog, p,d,q, step, currency)
            arimax_5_horizon_vol(df, exog, p_vol, d_vol, q_vol, p_gar, q_gar, step,currency)
            
        
        
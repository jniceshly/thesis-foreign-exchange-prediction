import sys
import os
import io
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from arch import arch_model
    import numba
except Exception as e:
    print(f"Import error: {e}")
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
import joblib
import holidays
from pandas.tseries.offsets import CustomBusinessDay

st.set_page_config(layout="wide")

if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'current_currency' not in st.session_state:
    st.session_state.current_currency = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

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

### LOAD DATA
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

def merge_exog():
    inflasi = exog_inflasi()
    birate = exog_birate()
    #devisa = exog_devisa()
    exog = pd.merge(inflasi, birate, left_index=True, right_index=True, how="inner")
    #exog = pd.merge(exog, devisa, left_index=True, right_index=True, how="inner")
    return exog
#st.dataframe(merge_exog())

SHEET_ID = "1rObGLsEYcWcHkpqHVhVfHtvPfnI15-npOsl3bj3a5us"

@st.cache_data(ttl=300)
def load_sheet(gid):
    url = (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
        f"?format=csv&gid={gid}"
    )

    df = pd.read_csv(url, header=0, usecols=[0, 1], skip_blank_lines=True)

    df = df.dropna() # drop baris kosong
    df = df.reset_index(drop=True)

    return df

df_new_usd = load_sheet(gid="0")
df_new_eur = load_sheet(gid="753250280")
df_new_gbp = load_sheet(gid="1222767202")
df_new_inflasi = load_sheet(gid="1835176075")
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

def merge_exog_latest():
    inflasi = exog_inflasi_latest()
    birate = exog_birate_latest()
    #devisa = exog_devisa_latest()

    #exog = inflasi.join(birate, how="outer").join(devisa, how="outer")
    exog = inflasi.join(birate, how="outer")
    exog = exog.sort_index().ffill()
    return exog

def combine_exog():
    df_hist = merge_exog()
    df_new = merge_exog_latest()
    df_combine = pd.concat([df_hist, df_new])
    df_combine = df_combine[~df_combine.index.duplicated(keep="last")]
    df_combine = df_combine.sort_index()
    return df_combine

def combine_all_data(df_endog):
    full_exog_data = combine_exog()
    full_endog_data = df_endog
    all_data = pd.concat([full_endog_data, full_exog_data])
    all_data = all_data.sort_index()
    return all_data

def split_data(window_size, horizon, df):
    window_size = window_size
    horizon = horizon

    data = df[['Close Price', 'Inflasi', 'BI Rate']]

    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

def create_windows(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        window = data.iloc[i : i + window_size]
        target = data.iloc[i + window_size : i + window_size + horizon]['Close Price'].values
        X.append(window.values)
        y.append(target)
    return np.array(X), np.array(y)

def windowing(train_data, test_data, window_size, horizon):
    X_train, y_train = create_windows(train_data, window_size, horizon)
    X_test, y_test = create_windows(test_data, window_size, horizon)
    return X_train, y_train, X_test, y_test

### PRICE FORECASTING
def forecast_price(model, exog=None, steps=1):
    forecast = model.forecast(steps=steps, exog=exog)
    return forecast

def plot_forex(df, df_forecast, step, n_days):
    end_date = df.index.max()
    start_date = end_date - pd.Timedelta(days=n_days)

    df_filtered = df.loc[df.index >= start_date]

    df_close = pd.DataFrame(df['Close Price']).copy()
    df_close = df_close.reset_index()   # ensure Date is a column
    df_close.rename(columns={"index": "Date"}, inplace=True)

    last_date = df_close["Date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), 
        periods=step, 
        freq=custom_bd
    )

    fig = go.Figure()
    
    # past
    fig.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered["Close Price"],
        mode="lines",
        name="Historis",
        line=dict(color="blue", width=2),
    ))
    
    # CI
    fig.add_trace(go.Scatter(
        x=list(df_forecast["Date"]) + list(df_forecast["Date"][::-1]),
        y=list(df_forecast["Upper CI"]) + list(df_forecast["Lower CI"][::-1]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,100,100,0.5)', width=1),
        name="Confidence Interval (95%)",
        showlegend = True,
        hoverinfo='skip'
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
        title={'text': "Visualisasi Data Historis dan Hasil Prediksi",
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

df_map = {
    'USD/IDR': combine_usd,
    'EUR/IDR': combine_eur,
    'GBP/IDR': combine_gbp
}

st.sidebar.markdown("### 💱 Pilih Mata Uang")
currency = st.sidebar.radio("Pilih satu yang ingin dilihat prediksinya",["USD/IDR", "EUR/IDR", "GBP/IDR"])

if st.session_state.current_currency != currency:
    st.session_state.predicted = False
    st.session_state.forecast_df = None
    st.session_state.prediction_results = None
    st.session_state.current_df = None
    st.session_state.current_currency = currency

st.sidebar.write("")
st.sidebar.markdown("### 🗓 Tanggal Penutupan Terakhir")
last_date = df_map[currency]().index[-1]
st.sidebar.write(last_date.strftime('%d %B %Y'))
st.sidebar.markdown("###### (Tanggal yang tertera berdasarkan data terakhir Investing.com)")

st.sidebar.write("")
st.sidebar.markdown("### 🗓 Tanggal yang Akan Diprediksi")
future_dates = pd.date_range(
        start=df_map[currency]().index[-1] + timedelta(days=1), periods=1,
        freq=custom_bd
    )
st.sidebar.write(future_dates[0].strftime('%d %B %Y'))
st.sidebar.markdown("###### (Prediksi H+1 di Business Day)")

if currency == "USD/IDR":
    step = 1
    p = 0
    d = 1
    q = 2
    p_vol = 1
    d_vol = 0
    q_vol = 0
    p_gar = 1
    q_gar = 1
    df = combine_usd()
    choice = 1
    
elif currency == "EUR/IDR":
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

elif currency == "GBP/IDR":
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

def backtest_model(df, exog, p, d, q, n_days=5):
    results = []
    
    # Ambil n_days terakhir untuk evaluasi
    for i in range(n_days, 0, -1):
        # Data sampai i hari sebelum hari terakhir
        train_df = df.iloc[:-i]
        train_exog = exog.reindex(train_df.index)
        
        # Actual value yang akan diprediksi
        actual_date = df.index[-i]
        actual_price = df['Close Price'].iloc[-i]
        
        try:
            # Training model
            model = SARIMAX(
                train_df["Close Price"],
                exog=train_exog,
                order=(p, d, q),
                enforce_stationary=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            
            # Prediksi 1 step ahead dengan CI
            future_exog = exog.iloc[-i].values.reshape(1, -1)
            forecast_obj = model_fit.get_forecast(steps=1, exog=future_exog)
            
            # Get predicted value
            predicted_price = forecast_obj.predicted_mean.values[0]
            
            # Get confidence interval
            conf_int = forecast_obj.conf_int()
            lower_ci = conf_int.iloc[0, 0]
            upper_ci = conf_int.iloc[0, 1]
            
            # Hitung error
            error = actual_price - predicted_price
            error_pct = (error / actual_price) * 100
            
            # Cek apakah actual price dalam CI
            within_ci = lower_ci <= actual_price <= upper_ci
            
            results.append({
                'Date': actual_date,
                'Actual': actual_price,
                'Predicted': predicted_price,
                'Lower_CI': lower_ci,
                'Upper_CI': upper_ci,
                'Within_CI': within_ci,
                'Error': error,
                'Error_Pct': error_pct
            })
        except:
            # Jika error, skip
            continue
    
    return pd.DataFrame(results)

def display_comparison_table(backtest_df, currency):
    st.header(f"📊 Evaluasi Performa Model")
    st.write("Perbandingan hasil prediksi model dengan data aktual 5 hari kebelakang (backtesting).")
    st.write("")
    
    # Format dataframe untuk display
    display_df = backtest_df.copy()
    
    # Format tanggal
    display_df['Date'] = display_df['Date'].dt.strftime('%d-%m-%Y')
    display_df = display_df.sort_index(ascending=False)
    
    # Format currency
    display_df['Actual'] = display_df['Actual'].apply(lambda x: f"Rp {x:,.2f}")
    display_df['Predicted'] = display_df['Predicted'].apply(lambda x: f"Rp {x:,.2f}")
    display_df['Lower_CI'] = display_df['Lower_CI'].apply(lambda x: f"Rp {x:,.2f}")
    display_df['Upper_CI'] = display_df['Upper_CI'].apply(lambda x: f"Rp {x:,.2f}")
    display_df['Error'] = display_df['Error'].apply(lambda x: f"Rp {x:,.2f}")
    display_df['Error_Pct'] = display_df['Error_Pct'].apply(lambda x: f"{x:,.2f}%")
    
    # Format within CI
    display_df['Within_CI'] = display_df['Within_CI'].apply(
        lambda x: "✅ Ya" if x else "❌ Tidak"
    )
    
    # Rename columns
    display_df = display_df[[
        'Date', 'Actual', 'Predicted', 'Lower_CI', 'Upper_CI', 
        'Within_CI', 'Error', 'Error_Pct'
    ]]
    display_df.columns = [
        'Tanggal', 'Harga Aktual', 'Hasil Prediksi', 
        'Batas Bawah (95%)', 'Batas Atas (95%)', 
        'Dalam CI?', 'Selisih', 'Selisih (%)'
    ]
    
    # Display table dengan styling
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Metrics summary
    st.write("")
    st.write("")
    st.markdown("##### Ringkasan Evaluasi")
    
    mae = backtest_df['Error'].abs().mean()
    mape = backtest_df['Error_Pct'].abs().mean()
    rmse = np.sqrt((backtest_df['Error'] ** 2).mean())
    ci_coverage = (backtest_df['Within_CI'].sum() / len(backtest_df)) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "MAE",
            f"Rp {mae:,.2f}",
            help="Mean Absolute Error - Rata-rata selisih absolut antara prediksi dan aktual"
        )
    
    with col2:
        st.metric(
            "MAPE",
            f"{mape:.2f}%",
            help="Mean Absolute Percentage Error - Rata-rata persentase error"
        )
    
    with col3:
        st.metric(
            "RMSE",
            f"Rp {rmse:,.2f}",
            help="Root Mean Squared Error - Akar dari rata-rata kuadrat error"
        )
    
    with col4:
        st.metric(
            "CI Coverage",
            f"{ci_coverage:.0f}%",
            help="Persentase actual price yang berada dalam confidence interval 95%"
        )
    
    # Interpretasi
    st.write("")
    col1, col2 = st.columns(2)
    
    with col1:
        #with st.container(border=True):
        if mape < 1:
            st.success("🎯 Akurasi Prediksi: Persentase error (MAPE) di bawah 1%")
        elif mape < 2:
            st.success("🎯 Akurasi Prediksi: Persentase error (MAPE) di bawah 2%")
        elif mape < 5:
            st.warning("🎯 Akurasi Prediksi: Persentase error (MAPE) di atas 2%")
        else:
            st.error("🎯 Akurasi Prediksi: Persentase error (MAPE) di atas 5%")
    
    with col2:
        if ci_coverage >= 90:
            st.success(f"✔️ Reliabilitas CI: {ci_coverage:.0f}% hasil prediksi dalam rentang CI")
        elif ci_coverage >= 80:
            st.info(f"✔️ Reliabilitas CI: {ci_coverage:.0f}% hasil prediksi dalam rentang CI")
        elif ci_coverage >= 70:
            st.warning(f"✔️ Reliabilitas CI: Hanya {ci_coverage:.0f}% hasil prediksi dalam rentang CI")
        else:
            st.error(f"✔️ Reliabilitas CI: Hanya {ci_coverage:.0f}% hasil prediksi dalam rentang CI")
    
    return mae, mape, rmse, ci_coverage

def plot_comparison(backtest_df):
    st.write("")
    st.write("")
    st.markdown("##### Visualisasi Nilai Aktual dan Hasil Prediksi")
    fig = go.Figure()
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=list(backtest_df['Date']) + list(backtest_df['Date'][::-1]),
        y=list(backtest_df['Upper_CI']) + list(backtest_df['Lower_CI'][::-1]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Confidence Interval (95%)',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Actual prices
    fig.add_trace(go.Scatter(
        x=backtest_df['Date'],
        y=backtest_df['Actual'],
        mode='lines+markers',
        name='Harga Aktual',
        line=dict(color='blue', width=2),
        marker=dict(size=10),
        hovertemplate="Tanggal: %{x}<br>Aktual: Rp %{y:,.2f}<extra></extra>"
    ))
    
    # Predicted prices
    fig.add_trace(go.Scatter(
        x=backtest_df['Date'],
        y=backtest_df['Predicted'],
        mode='lines+markers',
        name='Hasil Prediksi',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=10),
        hovertemplate="Tanggal: %{x}<br>Prediksi: Rp %{y:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_white",
        title={
            'text': "Perbandingan Harga Aktual vs Prediksi Model (dengan 95% CI)",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        xaxis=dict(showgrid=False, title="Tanggal"),
        yaxis=dict(showgrid=True, gridcolor="lightgray", title="Harga (Rp)"),
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
    """
    Calculate RSI (Relative Strength Index)
    """
    df_rsi = df.copy()
    
    # Calculate price changes
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
    """
    Get latest RSI value and signal
    """
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
        freq=custom_bd
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
    perubahan_prediksi = next_price - last_price
    perubahan_persen = (perubahan_prediksi / last_price) * 100
    #upper_ci = forecast_df["Upper CI"].values[0]
    #lower_ci = forecast_df["Lower CI"].values[0]
    expected_return = ((next_price - last_price) / last_price) * 100
    
    #st.subheader(f"Hasil Prediksi Close Price {currency}")
    #st.write("")
    
    #perubahan_prediksi = next_price - last_price
    #perubahan_persen = (perubahan_prediksi / last_price) * 100
    #upper_ci = forecast_df["Upper CI"].values[0]
    #lower_ci = forecast_df["Lower CI"].values[0]

    #col1, col2, col3, col4 = st.columns(4)
    
    #return forecast_df   
    return {
        'forecast_df': forecast_df,
        'last_date': last_date,
        'last_price': last_price,
        'next_price': next_price,
        'future_dates': future_dates,
        'perubahan_prediksi': perubahan_prediksi,
        'perubahan_persen': perubahan_persen,
        'upper_ci': forecast_df["Upper CI"].values[0],
        'lower_ci': forecast_df["Lower CI"].values[0],
        'expected_return': expected_return,
        'currency': currency
    }

def display_prediction_results(results):
    #st.subheader(f"Menampilkan Hasil Prediksi {results['currency']}")
    #st.write("")
    #st.write("")
    st.header(f"📊 Prediksi Close Price {results['currency']}")
    st.write("")
    st.write("")

    st.markdown("##### Harga Hari Ini")
    st.metric(
        label=results['last_date'].strftime('%d-%m-%Y'),
        value=f"Rp {results['last_price']:,.2f}"
    )
    st.write("")
    col1, col2, col3 = st.columns(3)

    #with col1:
        #st.markdown("##### Harga Hari Ini")
        #st.metric(
            #label=results['last_date'].strftime('%d-%m-%Y'),
            #value=f"Rp {results['last_price']:,.2f}"
        #)
    with col1:
        st.markdown(f"##### Prediksi H+1")
        st.metric(
            label=f"{results['future_dates'][0].strftime('%d-%m-%Y')}",
            value=f"Rp {results['next_price']:,.2f}",
            delta=f"Perubahan: Rp.{results['perubahan_prediksi']:,.2f} ({results['perubahan_persen']:,.2f}%)",
            delta_color="normal" if results['perubahan_prediksi'] >= 0 else "inverse"
        )
    with col2:
        st.markdown("##### Batas Atas")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {results['upper_ci']:,.2f}", help = "batas atas dengan 95% tingkat kepercayaan berarti 95% kemungkinan harga penutupan sebenarnya tidak akan melebihi nilai ini"
        )
    with col3:
        st.markdown("##### Batas Bawah")
        st.metric(
            label="Tingkat kepercayaan: 95%",
            value=f"Rp {results['lower_ci']:,.2f}", help="batas bawah dengan 95% tingkat kepercayaan berarti 95% kemungkinan harga penutupan sebenarnya tidak akan lebih rendah dari nilai ini"
        )

    st.write("")
    if results['expected_return'] > 0:
        st.info("**Sinyal:** 🟢 Harga penutupan lebih rendah hari ini dibandingkan hasil prediksi, potensi return diprediksikan lebih tinggi.")
    else:
        st.info("**Sinyal:** 🔴 Harga penutupan lebih tinggi hari ini dibandingkan hasil prediksi, potensi return diprediksikan lebih rendah.")

    #st.write(f"**Expected Return:** {results['perubahan_persen']:.3f}%")
    #st.divider()

def choose_plot_range():
    st.write("")
    st.write("")
    st.markdown("##### Visualisasi Data Historis dan Rentang Prediksi")
    range_option = st.radio(
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

def info(exog_df):
    st.subheader(f"ℹ️ Informasi Tambahan")
    st.write("Hasil prediksi di atas merupakan hasil prediksi berdasarkan data harga penutupan historis serta " \
    "data tingkat inflasi dan suku bunga/BI rate.")
    st.write("Berikut data bulanan dari tingkat inflasi dan suku bunga/BI rate 5 tahun kebelakang (data dalam persen %).")
    exog_df = exog_df.resample('M').last().sort_index(ascending=False)
    exog_df.index = exog_df.index.strftime('%B %Y')
    st.dataframe(exog_df[['Inflasi', 'BI Rate']])

st.header("Hasil Prediksi")
st.write("---")

if st.sidebar.button("🔮 Prediksi"):
    exog = combine_exog()

    with st.spinner("Predicting..."):
        if choice in [1, 3, 5]:
            results = arimax_1_horizon(df, exog, p, d, q, step, currency)
            st.session_state.prediction_results = results
            st.session_state.current_df = df

            backtest_df = backtest_model(df, exog, p, d, q, n_days=5)
            st.session_state.backtest_results = backtest_df

            st.session_state.predicted = True
if (st.session_state.predicted != True):
    st.info("Klik tombol '🔮 Prediksi' pada sidebar untuk menampilkan hasil prediksi")


if st.session_state.predicted and st.session_state.prediction_results:
    # Tampilkan hasil prediksi
    display_prediction_results(st.session_state.prediction_results)
    # Pilih range plot
    n_days = choose_plot_range()
    
        # Tampilkan plot
    plot_forex(
        st.session_state.current_df, 
        st.session_state.prediction_results['forecast_df'], 
        step, 
        n_days
    )

    st.write("---")
    
    if 'backtest_results' in st.session_state and not st.session_state.backtest_results.empty:
        
        st.write("")

        # Tabel perbandingan
        mae, mape, rmse, ci_coverage = display_comparison_table(
            st.session_state.backtest_results, 
            st.session_state.prediction_results['currency']
        )

        # Penjelasan
        with st.expander("ℹ️ Pelajari lebih lanjut mengenai MAE, MAPE, dan RMSE"):
            st.markdown("""
            **Metrik Evaluasi Model:**
            
            1. **MAE (Mean Absolute Error)**
               - Rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual (dalam Rupiah)
               - Semakin kecil semakin baik
            
            2. **MAPE (Mean Absolute Percentage Error)**
               - Rata-rata persentase error dari semua prediksi
            
            3. **RMSE (Root Mean Squared Error)**
               - Akar kuadrat dari rata-rata error kuadrat (dalam Rupiah)
            """)
        with st.expander("ℹ️ Pelajari lebih lanjut mengenai CI (confidence interval)"):
            st.markdown("""
            Confidence Interval (CI) 95% menunjukkan rentang ketidakpastian dalam prediksi model.
            
            Cara Membaca:

            - Batas Bawah: Nilai terendah yang mungkin terjadi dengan kepercayaan 95%
            - Batas Atas: Nilai tertinggi yang mungkin terjadi dengan kepercayaan 95%
            - Tingkat Kepercayaan 95%: Jika kita melakukan prediksi 100 kali, sekitar 95 kali harga aktual akan berada dalam rentang CI
            """)

        # Plot perbandingan
        plot_comparison(st.session_state.backtest_results)
        st.write("---")
        info(combine_exog())
        st.write("---")
        st.info("Prediksi selesai. Pilih mata uang lain untuk dilihat hasil prediksinya atau kunjungi halaman 'Analisis Historis' untuk melihat analisis indikator teknikal.")

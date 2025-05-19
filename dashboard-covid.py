# =========================
# IMPORT DAN LOAD DATA
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Judul Dashboard
st.title("📊 Dashboard Interaktif COVID-19 Indonesia")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_indonesia_time_series_all (1).csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# =========================
# PREPROCESSING DATA
# =========================
# Ambil data terbaru per lokasi
df_latest = df.sort_values('Date').groupby('Location').tail(1).reset_index(drop=True)

# Pilih kolom penting
df_model = df_latest[['Location', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']].dropna()

# Hitung Case Fatality Rate (%)
df_model['Case Fatality Rate'] = (df_model['Total Deaths'] / df_model['Total Cases']) * 100

# =========================
# PETA INTERAKTIF CLUSTERING
# =========================
st.subheader("🗺️ Peta Interaktif Clustering Wilayah")

# Normalisasi dan clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_model['Cluster'] = kmeans.fit_predict(X_scaled)

# Tambahkan dummy koordinat
np.random.seed(42)
df_model['Latitude'] = -2.0 + np.random.randn(len(df_model)) * 3
df_model['Longitude'] = 117.0 + np.random.randn(len(df_model)) * 3

# Tampilkan peta
fig_map = px.scatter_mapbox(
    df_model,
    lat="Latitude",
    lon="Longitude",
    hover_name="Location",
    color="Cluster",
    zoom=4,
    height=500,
    color_continuous_scale="Viridis"
)
fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map)

# =========================
# GRAFIK TREN KASUS HARIAN
# =========================
st.subheader("📉 Grafik Tren Kasus Harian")

lokasi_terpilih = st.selectbox("Pilih Lokasi:", sorted(df['Location'].unique()))
df_lokasi = df[df['Location'] == lokasi_terpilih].sort_values('Date')

fig_trend = px.line(df_lokasi, x='Date', y='Total Cases', title=f'Tren Kasus di {lokasi_terpilih}')
st.plotly_chart(fig_trend)

# =========================
# RINGKASAN RISIKO WILAYAH
# =========================
st.subheader("🚦 Ringkasan Tingkat Risiko Wilayah")

# Klasifikasi risiko (dummy rules)
def classify_risk(total_cases):
    if total_cases > 100000:
        return "Tinggi"
    elif total_cases > 50000:
        return "Sedang"
    else:
        return "Rendah"

df_model['Risiko'] = df_model['Total Cases'].apply(classify_risk)

# Tampilkan tabel risiko
st.dataframe(
    df_model[[
        'Location', 'Total Cases', 'Total Deaths',
        'Total Recovered', 'Population Density',
        'Case Fatality Rate', 'Risiko'
    ]].sort_values(by='Total Cases', ascending=False)
)

# Visualisasi bar chart
st.subheader("📊 Distribusi Jumlah Wilayah per Risiko")
st.bar_chart(df_model['Risiko'].value_counts())

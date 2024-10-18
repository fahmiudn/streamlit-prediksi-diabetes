import streamlit as st
import pandas as pd
import folium
import tensorflow as tf
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from folium.plugins import MarkerCluster
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset once for all pages
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_skripsi.csv')

df = load_data()

# Define sidebar for navigation
st.sidebar.title("Diabetes Prediction Web App")
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Informasi Dataset", "Dashboard Visualisasi", "Pelatihan Model", "Input Data Baru untuk Prediksi"]
)

# 1. Halaman Informasi Dataset
if page == "Informasi Dataset":
    st.title("Informasi Dataset")
    st.write("Deskripsi dataset dan kolom yang tersedia.")
    st.write(df.head())
    st.write("Statistik deskriptif:")
    st.write(df.describe())

# 2. Halaman Dashboard Visualisasi
elif page == "Dashboard Visualisasi":
    st.title("Dashboard Visualisasi")

    # Map with Folium Marker Cluster
    st.subheader("Pemetaan Kasus Diabetes di Kota Bogor")

    # Create a map centered on Kota Bogor
    map_bogor = folium.Map(location=[-6.595038, 106.816635], zoom_start=12)

    # Create a MarkerCluster object
    marker_cluster = MarkerCluster().add_to(map_bogor)

    # Iterate through the DataFrame and add markers to the cluster
    for index, row in df.iterrows():
        try:
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            umur = row['umur']
            jk = row['jk']
            diagnosis = row['diagnosis']

            # Customize marker color based on diagnosis
            color = 'red' if diagnosis == 1 else 'blue'

            # Create a popup with information
            popup_text = f"<b>Umur:</b> {umur}<br><b>Jenis Kelamin:</b> {jk}<br><b>Diagnosis:</b> {diagnosis}"

            # Add a marker to the cluster with the popup and color
            folium.Marker(
                location=[latitude, longitude],
                popup=popup_text,
                icon=folium.Icon(color=color)
            ).add_to(marker_cluster)
        except (ValueError, KeyError) as e:
            print(f"Error processing row {index}: {e}")
            continue  # Skip this row if there's an error

    # Display the map
    folium_static(map_bogor)

    # 2. Distribusi Diagnosa Diabetes di Setiap Kelurahan
    st.subheader("Distribusi Diagnosa Diabetes di Setiap Kelurahan")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='kelurahan', hue='diagnosis', palette='Set1')
    plt.xticks(rotation=90)
    plt.title('Distribusi Diagnosa Diabetes di Setiap Kelurahan')
    plt.xlabel('Kelurahan')
    plt.ylabel('Jumlah Kasus')
    st.pyplot(plt.gcf())
    plt.clf()

    # Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes (Pie Chart)
    st.subheader("Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes")
    plt.figure(figsize=(6, 6))
    df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'green'], startangle=90, explode=[0.1, 0])
    plt.title('Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes')
    plt.ylabel('')
    st.pyplot(plt.gcf())
    plt.clf()

    # Heatmap Korelasi Antar Variabel Numerik
    st.subheader("Heatmap Korelasi Antar Variabel Numerik")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) > 0:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Heatmap Korelasi Antar Variabel Numerik')
        st.pyplot(plt.gcf())
        plt.clf()

    # Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin
    st.subheader("Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='jk', hue='diagnosis', palette='Set1')
    plt.title('Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin')
    plt.xlabel('Jenis Kelamin')
    plt.ylabel('Jumlah Kasus')
    plt.legend(title='Diagnosis')
    st.pyplot(plt.gcf())
    plt.clf()

# 3. Halaman Pelatihan Model

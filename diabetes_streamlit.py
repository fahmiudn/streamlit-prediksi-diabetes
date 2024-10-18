import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
from math import sqrt
from streamlit_folium import folium_static

# Buat tampilan web menggunakan Streamlit
st.title('Aplikasi Prediksi Diabetes')

# Load dataset langsung di dalam kode
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_skripsi.csv')

df = load_data()

# Fungsi untuk menampilkan halaman awal
def halaman_awal():
    st.header("Pengertian Diabetes")
    st.write(
        """
        Diabetes adalah penyakit kronis yang terjadi ketika tubuh tidak dapat memproduksi cukup insulin atau tidak dapat menggunakan insulin secara efektif.
        Insulin adalah hormon yang mengatur kadar glukosa dalam darah.
        Terdapat dua jenis diabetes: Diabetes Tipe 1 dan Diabetes Tipe 2.
        Diabetes tipe 2 lebih umum dan sering terkait dengan faktor gaya hidup.
        """
    )
    st.header("Informasi Umum Dataset")
    st.write(
        """
        Dataset ini berisi informasi tentang pasien yang memiliki risiko diabetes.
        Fitur-fitur dalam dataset ini meliputi umur, tekanan darah, BMI, dan beberapa indikator kesehatan lainnya.
        """
    )
    st.header("Akurasi Model")
    st.write("Akurasi Model: **97.25%**")
    st.write("Precision: **100%**")
    st.write("Recall: **94.31%**")
    st.write("F1 Score: **97.07%**")

# Fungsi untuk menampilkan halaman dashboard visualisasi
def halaman_dashboard():
    st.header("Dashboard Visualisasi")

    # Load dataset
    df = pd.read_csv("diabetes_skripsi.csv")  # Ganti dengan path dataset Anda

    # 1. PEMETAAN MENGGUNAKAN FOLIUM
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
            if diagnosis == 1:
                color = 'red'
            else:
                color = 'blue'

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

    # 3. Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes (Pie Chart)
    st.subheader("Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes")
    plt.figure(figsize=(6, 6))
    df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'green'], startangle=90, explode=[0.1, 0])
    plt.title('Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes')
    plt.ylabel('')
    st.pyplot(plt.gcf())

    # 4. Heatmap Korelasi
    st.subheader("Heatmap Korelasi Antar Variabel Numerik")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) > 0:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Heatmap Korelasi Antar Variabel Numerik')
        st.pyplot(plt.gcf())

    # 5. Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin
    st.subheader("Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='jk', hue='diagnosis', palette='Set1')
    plt.title('Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin')
    plt.xlabel('Jenis Kelamin')
    plt.ylabel('Jumlah Kasus')
    plt.legend(title='Diagnosis')
    st.pyplot(plt.gcf())

# Fungsi untuk menampilkan halaman prediksi
def halaman_prediksi():
    st.header("Prediksi Diabetes")

    # Input form untuk data baru
    umur = st.number_input("Umur:", min_value=0, max_value=120, value=0)
    jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    merokok = st.selectbox("Merokok", ["Ya", "Tidak"])
    aktivitas_fisik = st.selectbox("Aktivitas Fisik", ["Ya", "Tidak"])
    konsumsi_alkohol = st.selectbox("Konsumsi Alkohol", ["Ya", "Tidak"])
    tekanan_darah = st.number_input("Tekanan Darah:", min_value=0, value=0)
    bmi = st.number_input("BMI:", min_value=0.0, value=0.0)
    lingkar_perut = st.number_input("Lingkar Perut (cm)", min_value=0, max_value=200, value=0)
    pemeriksaan_gula = st.number_input("Hasil Pemeriksaan Gula (mg/dL)", min_value=0, max_value=400, value=0)

    # Konversi input ke format numerik
    jk = 1 if jk == "Laki-laki" else 0
    merokok = 1 if merokok == "Ya" else 0
    aktivitas_fisik = 1 if aktivitas_fisik == "Ya" else 0
    konsumsi_alkohol = 1 if konsumsi_alkohol == "Ya" else 0

    if st.button("Prediksi"):
        # Persiapkan data untuk prediksi
        new_data = np.array([[umur, jk, merokok, aktivitas_fisik, konsumsi_alkohol, tekanan_darah, bmi, lingkar_perut, pemeriksaan_gula]])

        # Gunakan scaler untuk transformasi data baru
        scaler = MinMaxScaler()
        new_data_scaled = scaler.transform(new_data)
        new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

        # Prediksi menggunakan model
        new_prediction_prob = model.predict(new_data_scaled)
        new_prediction_class = (new_prediction_prob > 0.5).astype("int32")

        # Tampilkan hasil prediksi
        st.write(f"Probabilitas Diabetes: {new_prediction_prob[0][0]:.2f}")
        st.write(f"Prediksi Kelas: {'Diabetes' if new_prediction_class[0][0] == 1 else 'Non-Diabetes'}")

# Menu navigasi
pages = {
    "Halaman Awal": halaman_awal,
    "Dashboard Visualisasi": halaman_dashboard,
    "Prediksi Diabetes": halaman_prediksi,
}

selected_page = st.sidebar.selectbox("Pilih Halaman", options=list(pages.keys()))

# Jalankan halaman yang dipilih
pages[selected_page]()

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
elif page == "Pelatihan Model":
    st.title("Pelatihan Model LSTM")

    # Feature and target
    X = df.drop(['diagnosis', 'puskesmas', 'kelurahan', 'longitude', 'latitude'], axis=1)
    y = df['diagnosis']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

    # Scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Simpan scaler di session state
    st.session_state['scaler'] = scaler

    # Tidak ada input parameter dari pengguna, hard-code parameter training
    neurons = 64
    epochs = 50
    batch_size = 128
    learning_rate = 0.001

    st.write(f"Jumlah Neuron: {neurons}")
    st.write(f"Jumlah Epoch: {epochs}")
    st.write(f"Batch Size: {batch_size}")
    st.write(f"Learning Rate: {learning_rate}")
    
    if st.button("Latih Model"):
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', 
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      metrics=['accuracy'])

        # Train model
        model.fit(X_train, 
                  y_train, 
                  epochs=epochs, 
                  batch_size=batch_size, 
                  validation_split=0.2, verbose=1)

        # Save model in session state
        st.session_state['model'] = model

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Accuracy: {test_accuracy}")
        
        # Setelah model dilatih, kita lakukan prediksi pada test set
        y_pred_prob = model.predict(X_test)
        
        # Konversi prediksi probabilitas menjadi nilai kelas biner (0 atau 1)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        
        # Hitung MAE, RMSE, dan MAPE berdasarkan kelas prediksi
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Print hasil evaluasi
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# 4. Halaman Input Data Baru untuk Prediksi
elif page == "Input Data Baru untuk Prediksi":
    st.title("Input Data Baru untuk Prediksi Diabetes")

    # Pastikan model sudah dilatih dan disimpan di session state
    if 'model' not in st.session_state:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu di halaman 'Pelatihan Model'.")
    else:
        # Form input data
        umur = st.number_input("Umur:", min_value=0, max_value=120, value=0)
        jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        merokok = st.selectbox("Merokok", ["Ya", "Tidak"])
        aktivitas_fisik = st.selectbox("Aktivitas Fisik", ["Aktif", "Tidak Aktif"])
        riwayat_keluarga = st.selectbox("Riwayat Keluarga Diabetes", ["Ya", "Tidak"])
        
        # Map input ke dalam fitur sesuai dengan dataset
        features = {
            'umur': umur,
            'jk': 1 if jk == "Laki-laki" else 0,
            'merokok': 1 if merokok == "Ya" else 0,
            'aktivitas_fisik': 1 if aktivitas_fisik == "Aktif" else 0,
            'riwayat_keluarga': 1 if riwayat_keluarga == "Ya" else 0
        }

        # Ubah fitur menjadi DataFrame
        input_data = pd.DataFrame(features, index=[0])
        
        # Lakukan scaling pada input_data
        scaler = st.session_state.get('scaler')
        if scaler is not None:
            input_data_scaled = scaler.transform(input_data)

            # Reshape untuk LSTM
            input_data_reshaped = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

            # Buat prediksi
            model = st.session_state['model']
            prediction_prob = model.predict(input_data_reshaped)
            prediction = (prediction_prob > 0.5).astype("int32")[0][0]

            # Tampilkan hasil prediksi
            if prediction == 1:
                st.success("Prediksi: Pasien berisiko terkena diabetes.")
            else:
                st.success("Prediksi: Pasien tidak berisiko terkena diabetes.")
        else:
            st.error("Scaler tidak ditemukan. Pastikan model telah dilatih dan scaler telah disimpan.")

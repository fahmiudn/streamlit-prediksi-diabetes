import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset once for all pages
@st.cache
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
    st.subheader("Map Folium Marker Cluster")
    map_data = df[['latitude', 'longitude']].dropna()
    map_center = [map_data['latitude'].mean(), map_data['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=10)
    for idx, row in map_data.iterrows():
        folium.Marker([row['latitude'], row['longitude']]).add_to(m)
    folium_static(m)

    # Distribusi Diagnosa Diabetes di Setiap Kelurahan
    st.subheader("Distribusi Diagnosa Diabetes di Setiap Kelurahan")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='kelurahan', hue='diagnosis', ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes
    st.subheader("Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='diagnosis', ax=ax)
    st.pyplot(fig)

    # Heatmap Korelasi Antar Variabel Numerik
    st.subheader("Heatmap Korelasi Antar Variabel Numerik")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin
    st.subheader("Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='jenis_kelamin', hue='diagnosis', ax=ax)
    st.pyplot(fig)

# 3. Halaman Pelatihan Model
elif page == "Pelatihan Model":
    st.title("Pelatihan Model LSTM")
    
    # Hyperparameters input
    neuron = st.number_input("Jumlah Neuron", min_value=10, max_value=200, value=50, step=10)
    epoch = st.number_input("Epoch", min_value=10, max_value=200, value=50, step=10)
    batch_size = st.number_input("Batch Size", min_value=16, max_value=128, value=32, step=16)
    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)

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

    if st.button("Latih Model"):
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=neuron, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Train model
        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2, verbose=1)

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Accuracy: {test_accuracy}")

        # Save model and scaler in session state
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.success("Model dan scaler telah disimpan dalam session state.")

# 4. Halaman Input Data Baru untuk Prediksi
elif page == "Input Data Baru untuk Prediksi":
    st.title("Input Data Baru untuk Prediksi Diabetes")

    # Check if model exists in session state
    if 'model' in st.session_state and 'scaler' in st.session_state:
        model = st.session_state['model']
        scaler = st.session_state['scaler']

        st.write("Masukkan data baru:")
        # Input data baru
        new_data = {
            'usia': st.number_input('Usia', min_value=0, max_value=100),
            'berat_badan': st.number_input('Berat Badan', min_value=0, max_value=200),
            # tambahkan input field sesuai kolom pada dataset
        }
        new_data_df = pd.DataFrame([new_data])

        # Preprocess the new data
        new_data_scaled = scaler.transform(new_data_df)
        new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

        # Predict the result
        if st.button("Prediksi"):
            prediction = model.predict(new_data_scaled)
            diagnosis = "Diabetes" if prediction > 0.5 else "Non-Diabetes"
            st.write(f"Hasil prediksi: {diagnosis}")
    else:
        st.error("Model belum dilatih. Silakan latih model terlebih dahulu di halaman Pelatihan Model.")

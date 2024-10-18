import pandas as pd
import warnings
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, classification_report
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Title of the app
st.title("Diabetes Prediction Model with LSTM")

# Load the dataset
df = pd.read_csv('diabetes_skripsi.csv')
st.write("Data Preview:")
st.write(df.head())

# Delete columns 'puskesmas', 'kelurahan', 'longitude', and 'latitude'
df = df.drop(['puskesmas', 'kelurahan', 'longitude', 'latitude'], axis=1)

# Define features (X) and target (y)
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
st.write(f"Test Accuracy: {test_accuracy:.4f}")

# After training, make predictions on test set
y_pred_prob = model.predict(X_test)

# Convert predicted probabilities to binary class values (0 or 1)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Calculate MAE, RMSE, and MAPE based on predicted probabilities
mae = mean_absolute_error(y_test, y_pred_prob)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))

# Display evaluation results
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display classification results
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")

# Optional: Display classification report
classification_report_str = classification_report(y_test, y_pred)
st.text("Classification Report:\n" + classification_report_str)

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Last ned data
new_data = pd.read_csv('btc_new_data.csv')
new_data.set_index('Open Time', inplace=True)

# Skaler data
scaler = joblib.load('scaler.pkl')

new_scaled_data = scaler.transform(new_data['Close'].values.reshape(-1,1))

# Forberede data for LSTM
x_new = []
for i in range(180, len(new_data)):
    x_new.append(new_scaled_data[i-180:i, 0])
    
x_new = np.array(x_new)
x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1], 1))

# Laste modell
model = load_model('btc_lstm_model.h5')

# Predictions
predictions = model.predict(x_new)
predicted_labels = np.argmax(predictions, axis=1)

# Encoder
encoder = joblib.load('encoder.pkl')

# Oversett numeriske etiketter tilbake til originaltiketter
predicted_signals = encoder.inverse_transform(predicted_labels)

print(predicted_signals)

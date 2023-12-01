import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
import tensorflow as tf
import stockDataHandler

def get_full_path(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, filename)

def cost_sensitive_loss(y_true, y_pred):
    cost_matrix = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [1, 1, 0]
    ])
    cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
    loss = cost_values * keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss

# Les inn data
stock_data = stockDataHandler.LoadData(get_full_path('onehour2022_2003.csv'))

# Bruk stockDataHandler for å utføre forberedende funksjoner
stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
stockDataHandler.SetMacd(stock_data, 50)
stockDataHandler.CleanData(stock_data)

# Last inn tidligere trente objekter (scaler og encoder)
scaler = joblib.load(get_full_path('scalerRMSProp.pkl'))
encoder = joblib.load(get_full_path('encoderRMSProp.pkl'))

model_path = get_full_path('onehour_RMSProp.h5')

# Skaler testdata
scaled_test_data = scaler.transform(stock_data[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA50', 'EMA100', 'MACD']])

# Forberede testdata for prediksjon
x_test = []
numberOfBars = 100  # Satt til det samme antallet bars som i treningskoden
for i in range(numberOfBars, len(stock_data)):
    x_test.append(scaled_test_data[i - numberOfBars:i])
x_test = np.array(x_test)

# Last inn tidligere trent modell
model = keras.models.load_model(model_path, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

# Gjør prediksjoner
predictions = model.predict(x_test)

buy_probs = predictions[:, encoder.transform(['Buy'])[0]]
sell_probs = predictions[:, encoder.transform(['Sell'])[0]]


threshold = 0.9
predicted_labels = []

for buy_prob, sell_prob in zip(buy_probs, sell_probs):
    if buy_prob > threshold:
        predicted_labels.append('Buy')
    elif sell_prob > threshold:
        predicted_labels.append('Sell')
    else:
        predicted_labels.append('Hold')

#print(predicted_labels)

# Legg til predikerte etiketter til DataFrame
stock_data['Predicted_Signal'] = 'Hold'
stock_data['Predicted_Signal'].iloc[numberOfBars:] = predicted_labels

signal_counts = stock_data['Predicted_Signal'].value_counts()
print("Number of 'Buy', 'Sell', and 'Hold' signals in the data:")
print(signal_counts)

# predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1))

# print(predicted_labels)

# # Legg til predikerte etiketter til DataFrame
# stock_data['Predicted_Signal'] = 'Hold'
# stock_data['Predicted_Signal'].iloc[numberOfBars:] = predicted_labels

# signal_counts = stock_data['Predicted_Signal'].value_counts()
# print("Number of 'Buy', 'Sell', and 'Hold' signals in the data:")
# print(signal_counts)

# Visualiser resultater
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Original_Close'], mode='lines', name='Close'))
buy_signals = stock_data[stock_data['Predicted_Signal'] == 'Buy']['Original_Close']
sell_signals = stock_data[stock_data['Predicted_Signal'] == 'Sell']['Original_Close']
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))
fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
fig.show()

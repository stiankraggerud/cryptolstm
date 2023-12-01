import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
import tensorflow as tf
import stockDataHandler
from binance.client import Client
import time




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

# Sett opp Binance klienten
api_key = "8aonjjjJ0EJt1coxcR335wl1Qsq4eYov9k2CuXUxP7IQhiBwDPOrbp5D6LANdbkW"
api_secret = "Te0MrdyCGhTcNGvmTvWDfSEG4KLPRlaIecaC0PqTtiCAnO2bB5T8miWsfQJUYDSW"
client = Client(api_key, api_secret)

# Last inn tidligere trente objekter (scaler og encoder)
scaler = joblib.load(get_full_path('scalerRMSProp.pkl'))
encoder = joblib.load(get_full_path('encoderRMSProp.pkl'))
model_path = get_full_path('onehour_RMSProp.h5')
model = keras.models.load_model(model_path, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

# Initialiser en Plotly figur
fig = go.Figure()

while True:
    # Hent nyeste data fra Binance
    apikey = "8aonjjjJ0EJt1coxcR335wl1Qsq4eYov9k2CuXUxP7IQhiBwDPOrbp5D6LANdbkW"
    apisecret = "Te0MrdyCGhTcNGvmTvWDfSEG4KLPRlaIecaC0PqTtiCAnO2bB5T8miWsfQJUYDSW"
    # Initialiser Binance-klienten
    client = Client(apikey, apisecret)

    # Hent de siste 100 klines
    klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=300)

    # Konverter klines til en DataFrame
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(klines, columns=columns)

    # Behold bare de kolonnene vi trenger
    stock_data = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Konverter kolonnene til passende datatyper
    stock_data['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    stock_data.set_index('Open time', inplace=True)
    stock_data['Open'] = stock_data['Open'].astype(float)
    stock_data['High'] = stock_data['High'].astype(float)
    stock_data['Low'] = stock_data['Low'].astype(float)
    stock_data['Close'] = stock_data['Close'].astype(float)
    stock_data['Volume'] = stock_data['Volume'].astype(float)

    # Bruk stockDataHandler funksjonene
    stockDataHandler.SetOrignalData(stock_data)
    stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
    stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
    stockDataHandler.SetMacd(stock_data, 50)
    stockDataHandler.CleanData(stock_data)

    # Skaler data
    scaled_test_data = scaler.transform(stock_data[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA50', 'EMA100', 'MACD']])
    
    # Forberede testdata for prediksjon
    x_test = []
    numberOfBars = 100
    for i in range(numberOfBars, len(stock_data)):
        x_test.append(scaled_test_data[i - numberOfBars:i])
    x_test = np.array(x_test)

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

    stock_data['Predicted_Signal'] = 'Hold'
    stock_data['Predicted_Signal'].iloc[numberOfBars:] = predicted_labels

    # Oppdater figuren
    fig.data = []  # Fjern eksisterende data fra figuren
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Original_Close'], mode='lines', name='Close'))
    buy_signals = stock_data[stock_data['Predicted_Signal'] == 'Buy']['Original_Close']
    sell_signals = stock_data[stock_data['Predicted_Signal'] == 'Sell']['Original_Close']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))
    fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
    fig.show()

    time.sleep(20)  # Vent i en time før neste oppdatering

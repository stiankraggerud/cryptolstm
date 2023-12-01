import os
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import plotly.graph_objects as go


import stockDataHandler

# Les inn data
stock_data = stockDataHandler.LoadData('onehour2022_2003.csv')

stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
stockDataHandler.SetMacd(stock_data, 50)
stockDataHandler.CleanData(stock_data)


#modelname = get_full_path('btc_onehour_2018_oneEpoch.h5')
numberOfBars = 100
# Generere etiketter for klassifisering basert på toppene og dalene

min_prominence = 200
peak_indexes, _ = find_peaks(stock_data['Original_Close'].values, prominence=min_prominence) # Topper for salg
valley_indexes, _ = find_peaks(-stock_data['Original_Close'].values, prominence=min_prominence) # Bunner for kjøp

offset = 3


#sell_indexes = peak_indexes - offset
sell_indexes = peak_indexes
sell_indexes = sell_indexes[sell_indexes >= 0]
#buy_indexes = valley_indexes - offset
buy_indexes = valley_indexes

buy_indexes = buy_indexes[buy_indexes >= 0]

# Merk at vi nå bruker 'Sell' for sell_indexes og 'Buy' for buy_indexes
stock_data.loc[stock_data.index[sell_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[buy_indexes], 'Signal'] = 'Buy'
stock_data['Signal'].fillna('Hold', inplace=True)

# Splitt data i trening og test sett
training_data_len = math.ceil(len(stock_data) * 0.8)
train_data = stock_data[:training_data_len]

training_data_len = math.ceil(len(stock_data) * 0.8)
train_data = stock_data[:training_data_len]

# Skriv ut antall 'Buy', 'Sell', og 'Hold' signaler i treningsdataene
signal_counts = train_data['Signal'].value_counts()
print("Number of 'Buy', 'Sell', and 'Hold' signals in training data:")
print(signal_counts)

buy_signals = stock_data[stock_data['Signal'] == 'Buy']['Original_Close']
sell_signals = stock_data[stock_data['Signal'] == 'Sell']['Original_Close']

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Original_Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))


fig.update_layout(title='Stock Price with Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
fig.show()

assert False

# Skaler data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA50', 'EMA100', 'MACD']])

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(scaled_data[i - numberOfBars:i])
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

# Oversampling før one-hot encoding
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1),
                                                                y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)


cost_matrix = np.array([
    [0, 1, 2],  # Kostnad fra Buy til [Buy, Sell, Hold]
    [1, 0, 2],  # Kostnad fra Sell til [Buy, Sell, Hold]
    [1, 1, 0]   # Kostnad fra Hold til [Buy, Sell, Hold]
])

import tensorflow as tf
def cost_sensitive_loss(y_true, y_pred):
    cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
    loss = tf.reduce_sum(cost_values * tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)
    return loss



#LSTM model - ny trening
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], 8)))  # 8 features
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(y_train_resampled.shape[1], activation='softmax'))
model.summary()

modelname = stockDataHandler.get_full_path('onehour_rmsprop_epoch1.h5')

# Last inn tidligere trent modell
model = keras.models.load_model(modelname, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})


#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='RMSProp', loss=cost_sensitive_loss, metrics=['accuracy'])
model.fit(x_train_resampled, y_train_resampled, batch_size=1, epochs=1)

newModelname = stockDataHandler.get_full_path('onehour_rmsprop_epoch2.h5')
model.save(newModelname)

# Lagrer scaler og encoder til disk
joblib.dump(scaler, stockDataHandler.get_full_path('scaler.pkl'))
joblib.dump(encoder, stockDataHandler.get_full_path('encoder.pkl'))

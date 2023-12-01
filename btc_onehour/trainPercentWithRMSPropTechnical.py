import os
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import stockDataHandler
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


def min_max_normalization(data, columns):
    normalized_data = data.copy()
    scalers = {}
    for column in columns:
        scaler = MinMaxScaler()
        normalized_data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
        scalers[column] = scaler
    return normalized_data, scalers

def percent_return(data, columns):
    for column in columns:
        data[column] = data[column].pct_change()
    return data

# Load the data
stock_data = stockDataHandler.LoadData('onehour2022_2003.csv')
stockDataHandler.SetEMA(stock_data, 12, 'EMA12')
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
stockDataHandler.SetMacd(stock_data, 50)
stockDataHandler.compute_ADX(stock_data,14)
stockDataHandler.compute_RSI(stock_data,8)

stockDataHandler.CleanData(stock_data)

numberOfBars = 100

# Label generation
# Generer kjøp, salg og hold signaler basert på de nye tekniske indikatorene
for i in range(1, len(stock_data)):
    if stock_data['RSI'].iloc[i] < 40 and stock_data['EMA12'].iloc[i] > stock_data['EMA100'].iloc[i] and stock_data['ADX'].iloc[i] > 20:
        stock_data.loc[stock_data.index[i], 'Signal'] = 'Buy'
    elif stock_data['RSI'].iloc[i] > 60 and stock_data['EMA12'].iloc[i] < stock_data['EMA100'].iloc[i] and stock_data['ADX'].iloc[i] > 25:
        stock_data.loc[stock_data.index[i], 'Signal'] = 'Sell'
    else:
        stock_data.loc[stock_data.index[i], 'Signal'] = 'Hold'

stock_data['Signal'].fillna('Hold', inplace=True)
stock_data = stock_data.dropna()

# Split the data
training_data_len = math.ceil(len(stock_data) * 0.7)
train_data = stock_data[:training_data_len]

columns_to_percent_return = ['Open', 'Close', 'High', 'Low', 'RSI', 'ADX']
columns_to_min_max_normalize = ['Volume', 'EMA12', 'EMA200']

# Apply the desired transformations
train_data = percent_return(train_data, columns_to_percent_return)
train_data_min_max, min_max_scalers = min_max_normalization(train_data, columns_to_min_max_normalize)
train_data[columns_to_min_max_normalize] = train_data_min_max[columns_to_min_max_normalize]

# Drop NaN values after transformations
train_data = train_data.dropna()

# Print signal counts
signal_counts = train_data['Signal'].value_counts()
print("Number of 'Buy', 'Sell', and 'Hold' signals in training data:")
print(signal_counts)




stockDataHandler.showDataInGraph(train_data)

# assert False



# # Prepare data for LSTM
# x_train = []
# y_train = []

# for i in range(numberOfBars, len(train_data)):
#     x_train.append(train_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
#     y_train.append(train_data.iloc[i]['Signal'])

# x_train = np.array(x_train)
# encoder = LabelEncoder()
# y_train_encoded = encoder.fit_transform(y_train)

# y_train_encoded = keras.utils.to_categorical(y_train_encoded) 

# # Oversampling
# ros = RandomOverSampler(random_state=0)
# x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
# x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
# y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

# cost_matrix = np.array([
#     [0, 1, 1],  # Kostnad fra Buy til [Buy, Sell, Hold]
#     [1, 0, 1],  # Kostnad fra Sell til [Buy, Sell, Hold]
#     [1, 1, 0]   # Kostnad fra Hold til [Buy, Sell, Hold]
# ])

# import tensorflow as tf
# def cost_sensitive_loss(y_true, y_pred):
#     cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
#     cost_values = tf.cast(cost_values, tf.float32)
#     loss = tf.reduce_sum(cost_values * tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)
#     return loss

# # LSTM model
# model = keras.Sequential()
# model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], len(columns_to_percent_return + columns_to_min_max_normalize))))  
# model.add(layers.LSTM(100, return_sequences=False))
# model.add(layers.Dense(50))
# model.add(layers.Dense(y_train_encoded_resampled.shape[1], activation='softmax'))
# model.summary()

# # Laster inn tidligere modell:
# # modelname = stockDataHandler.get_full_path('onehour_2018_2023_technical_epoch1.h5')
# # model = keras.models.load_model(modelname, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

# model.compile(optimizer='RMSProp', loss=cost_sensitive_loss, metrics=['accuracy'])

# model.fit(x_train_resampled, y_train_encoded_resampled, batch_size=3, epochs=5)

# newModelname = stockDataHandler.get_full_path('onehour_2018_2023_technical_epoch1.h5')
# model.save(newModelname)

# joblib.dump(min_max_scalers, stockDataHandler.get_full_path('min_max_scalers.pkl'))
# joblib.dump(encoder, stockDataHandler.get_full_path('encoder.pkl'))


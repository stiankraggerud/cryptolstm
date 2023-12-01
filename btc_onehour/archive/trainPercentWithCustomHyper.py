import os
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

import joblib
import plotly.graph_objects as go
import stockDataHandler
import tensorflow as tf

# Les inn data
stock_data = stockDataHandler.LoadData('onehour2018_2023.csv')

stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
stockDataHandler.SetMacd(stock_data, 50)
stockDataHandler.CleanData(stock_data)

numberOfBars = 100

min_prominence = 200
peak_indexes, _ = find_peaks(stock_data['Original_Close'].values, prominence=min_prominence) # Topper for salg
valley_indexes, _ = find_peaks(-stock_data['Original_Close'].values, prominence=min_prominence) # Bunner for kjøp

sell_indexes = peak_indexes
sell_indexes = sell_indexes[sell_indexes >= 0]
buy_indexes = valley_indexes
buy_indexes = buy_indexes[buy_indexes >= 0]

stock_data.loc[stock_data.index[sell_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[buy_indexes], 'Signal'] = 'Buy'
stock_data['Signal'].fillna('Hold', inplace=True)

training_data_len = math.ceil(len(stock_data) * 0.8)
train_data = stock_data[:training_data_len]

# ...

scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA50', 'EMA100', 'MACD']])

x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(scaled_data[i - numberOfBars:i])
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1),
                                                                y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

cost_matrix = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

def cost_sensitive_loss(y_true, y_pred):
    cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
    loss = tf.reduce_sum(cost_values * tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)
    return loss

# BEGIN GRID SEARCH
def create_model(optimizer='adam'):
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], 8)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(50))
    model.add(layers.Dense(y_train_resampled.shape[1], activation='softmax'))
    model.compile(optimizer=optimizer, loss=cost_sensitive_loss, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=1, verbose=0)
param_grid = {
    'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adam'],
    'batch_size': [1, 10, 100]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(x_train_resampled, y_train_resampled)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Lagrer den beste modellen
best_model = grid_result.best_estimator_.model
joblib.dump(grid_result, stockDataHandler.get_full_path('grid_result.pkl'))

modelname = stockDataHandler.get_full_path('onehour_best_model.h5')
best_model.save(modelname)
# END GRID SEARCH

# Lagrer scaler og encoder til disk
joblib.dump(scaler, stockDataHandler.get_full_path('scaler.pkl'))
joblib.dump(encoder, stockDataHandler.get_full_path('encoder.pkl'))

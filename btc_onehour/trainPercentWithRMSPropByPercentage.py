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
from keras.utils import to_categorical
import plotly.io as pio  

pio.renderers.default = "browser"  

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

def mark_buy_signals(stock_data, threshold=0.01, window=3):
    """
    Marks Buy signals based on forward looking increase by a given threshold.
    Ensures no consecutive Buy signals within the given window.
    """
    length = len(stock_data)
    buy_indices = []
    skip_until_index = -1

    for i in range(length - window):
        if i < skip_until_index:
            continue
        future_prices = stock_data['Close'].iloc[i+1:i+window+1]
        if any(future_prices > stock_data['Close'].iloc[i] * (1 + threshold)):
            buy_indices.append(i)
            skip_until_index = i + window
    
    return buy_indices

buy_indices = mark_buy_signals(stock_data)
#Signal = 0 == Hold
stock_data['Signal'] = 0
#Signal = 1 == Buy
stock_data.loc[stock_data.index[buy_indices], 'Signal'] = 1

stock_data['Signal'].fillna(0, inplace=True)
stock_data = stock_data.dropna()

# Split the data
training_data_len = math.ceil(len(stock_data) * 0.7)
train_data = stock_data[:training_data_len]

columns_to_percent_return = ['Open', 'Close', 'High', 'Low', 'RSI', 'ADX']
columns_to_min_max_normalize = ['Volume', 'EMA12', 'EMA100']

# Apply the desired transformations
train_data = percent_return(train_data, columns_to_percent_return)
train_data_min_max, min_max_scalers = min_max_normalization(train_data, columns_to_min_max_normalize)
train_data[columns_to_min_max_normalize] = train_data_min_max[columns_to_min_max_normalize]

# Drop NaN values after transformations
train_data = train_data.dropna()

# Print signal counts
signal_counts = train_data['Signal'].value_counts()
print("Number of 'Buy' and 'Hold' signals in training data:")
print(signal_counts)


#stockDataHandler.showDataInGraph(train_data)


# Prepare data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(train_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

y_train_encoded = keras.utils.to_categorical(y_train_encoded, 2) 

# Oversampling
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
y_train_resampled = keras.utils.to_categorical(y_train_resampled, 2)


# Flatten the timesteps and features dimensions before oversampling
x_train_flattened = x_train.reshape(x_train.shape[0], -1)

# Perform oversampling
x_train_resampled, y_train_resampled = ros.fit_resample(x_train_flattened, y_train_encoded)

# Reshape back to the original 3-dimensional shape
x_train_resampled = x_train_resampled.reshape(-1, numberOfBars, len(columns_to_percent_return + columns_to_min_max_normalize))
y_train_resampled = to_categorical(y_train_resampled)



cost_matrix = np.array([
    [0, 1],  # Kostnad fra Buy til [Buy, Hold]
    [2, 0]   # Kostnad fra Hold til [Buy, Hold]
])

import tensorflow as tf
def cost_sensitive_loss(y_true, y_pred):
    # y_true: (batch_size, 2)
    # y_pred: (batch_size, 2)
    
    # Expand cost matrix to match batch size
    expanded_cost_matrix = tf.repeat(tf.expand_dims(cost_matrix, axis=0), repeats=tf.shape(y_true)[0], axis=0)
    
    # Expand the dimensions of y_true to make it [batch_size, 1, 2]
    y_true_expanded = tf.expand_dims(y_true, axis=1)
    
    # Calculate the costs for each data point in the batch
    cost_values = tf.cast(tf.reduce_sum(expanded_cost_matrix * y_true_expanded, axis=-1), dtype=tf.float32)
    
    # Compute the softmax cross entropy loss
    basic_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    
    # Apply the costs to the basic losses
    cost_weighted_loss = cost_values * tf.reshape(basic_losses, [-1, 1])
    
    return tf.reduce_mean(cost_weighted_loss)



# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], len(columns_to_percent_return + columns_to_min_max_normalize))))  
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

# Laster inn tidligere modell:
# modelname = stockDataHandler.get_full_path('onehour_2018_2023_technical_epoch1.h5')
# model = keras.models.load_model(modelname, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

# model.compile(optimizer='RMSProp', loss=cost_sensitive_loss, metrics=['accuracy'])
# model.fit(x_train_resampled, y_train_encoded_resampled, batch_size=3, epochs=5)

print(x_train_resampled.shape)
print(y_train_resampled.shape)


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_resampled, y_train_resampled, epochs=10, batch_size=32)

#model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x_train_resampled, y_train_resampled, batch_size=3, epochs=5)

newModelname = stockDataHandler.get_full_path('onehour_2018_2023_technical_epoch1.h5')
model.save(newModelname)

joblib.dump(min_max_scalers, stockDataHandler.get_full_path('min_max_scalers.pkl'))
joblib.dump(encoder, stockDataHandler.get_full_path('encoder.pkl'))


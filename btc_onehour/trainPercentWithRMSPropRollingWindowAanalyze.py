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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

cost_matrix = np.array([
    [0, 1, 1],  # Kostnad fra Buy til [Buy, Sell, Hold]
    [1, 0, 1],  # Kostnad fra Sell til [Buy, Sell, Hold]
    [1, 1, 0]   # Kostnad fra Hold til [Buy, Sell, Hold]
])


import tensorflow as tf
def cost_sensitive_loss(y_true, y_pred):
    cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
    cost_values = tf.cast(cost_values, tf.float32)
    loss = tf.reduce_sum(cost_values * tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)
    return loss

# Load the saved model
modelname = stockDataHandler.get_full_path('onehour_2018_2023_rmsprop_rolling_epoch60.h5')
loaded_model = keras.models.load_model(modelname, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})


# Load the data
stock_data = stockDataHandler.LoadData('onehour2018_2023.csv')
stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
stockDataHandler.SetMacd(stock_data, 50)
stockDataHandler.CleanData(stock_data)

numberOfBars = 100

# Label generation
min_prominence = 150
peak_indexes, _ = find_peaks(stock_data['Original_Close'].values, prominence=min_prominence)
valley_indexes, _ = find_peaks(-stock_data['Original_Close'].values, prominence=min_prominence)
sell_indexes = peak_indexes
buy_indexes = valley_indexes

stock_data.loc[stock_data.index[sell_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[buy_indexes], 'Signal'] = 'Buy'
stock_data['Signal'].fillna('Hold', inplace=True)

# Split the data
training_data_len = math.ceil(len(stock_data) * 0.7)
train_data = stock_data[:training_data_len]

columns_to_percent_return = ['Open', 'Close', 'High', 'Low']
columns_to_min_max_normalize = ['Volume', 'EMA50', 'EMA100', 'MACD']

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

x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(train_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

y_train_encoded = keras.utils.to_categorical(y_train_encoded) 

# Oversampling
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

# Predict probabilities on the training data
probabilities = loaded_model.predict(x_train_resampled)

# Calculate average probabilities for each class
mean_probabilities = np.mean(probabilities, axis=0)
print("Average Probabilities for each class: ", mean_probabilities)

# Visualize the probabilities
plt.figure(figsize=(10, 6))
plt.hist(probabilities[:, 0], bins=50, alpha=0.5, label='Buy')
plt.hist(probabilities[:, 1], bins=50, alpha=0.5, label='Sell')
plt.hist(probabilities[:, 2], bins=50, alpha=0.5, label='Hold')
plt.legend(loc='upper right')
plt.title('Probability Distributions for Each Class')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()

# Check the class with the highest probability for each prediction
predicted_classes = np.argmax(probabilities, axis=1)
unique, counts = np.unique(predicted_classes, return_counts=True)
print("Predicted Class Counts: ", dict(zip(unique, counts)))

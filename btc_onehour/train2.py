import os
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_full_path(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, filename)

# Last ned data
stock_data = pd.read_csv(get_full_path('onehour2022_2003.csv'))
stock_data.set_index('Open time', inplace=True)
modelname = get_full_path('btc_onehour.h5')
numberOfBars = 500

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].ewm(span=50).mean()
stock_data['EMA100'] = stock_data['Close'].ewm(span=100).mean()

# Beregne MACD
macd_line = stock_data['Close'].ewm(span=50).mean() - stock_data['Close'].ewm(span=100).mean()
signal_line = macd_line.ewm(span=9).mean()  # Default configuration
stock_data['MACD'] = macd_line
stock_data['Signal_Line'] = signal_line

print(stock_data.index[stock_data.index.duplicated()])
stock_data = stock_data[~stock_data.index.duplicated(keep='first')]

# Generere etiketter for klassifisering basert på toppene og dalene
min_prominence = 500
peak_indexes, _ = find_peaks(stock_data['Close'].values, prominence=min_prominence)
valley_indexes, _ = find_peaks(-stock_data['Close'].values, prominence=min_prominence)


# Forskyvning i antall barer
offset = 3

# Forskjøvet indekser for 'Sell'
sell_indexes = peak_indexes - offset
sell_indexes = sell_indexes[sell_indexes >= 0]  # Filter ut negative indekser

# Forskjøvet indekser for 'Buy'
buy_indexes = valley_indexes - offset
buy_indexes = buy_indexes[buy_indexes >= 0]  # Filter ut negative indekser

stock_data.loc[stock_data.index[sell_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[buy_indexes], 'Signal'] = 'Buy'


stock_data['Signal'].fillna('Hold', inplace=True)

# Splitt data i trening og test sett
training_data_len = math.ceil(len(stock_data) * 0.8)
train_data = stock_data[:training_data_len]

# Skaler data
scaler_price = MinMaxScaler(feature_range=(0, 1))
scaled_price = scaler_price.fit_transform(train_data['Close'].values.reshape(-1, 1))

scaler_volume = MinMaxScaler(feature_range=(0, 1))
scaled_volume = scaler_volume.fit_transform(train_data['Volume'].values.reshape(-1, 1))

scaler_ema50 = MinMaxScaler(feature_range=(0, 1))
scaled_ema50 = scaler_ema50.fit_transform(train_data['EMA50'].values.reshape(-1, 1))

scaler_ema100 = MinMaxScaler(feature_range=(0, 1))
scaled_ema100 = scaler_ema100.fit_transform(train_data['EMA100'].values.reshape(-1, 1))


scaler_macd = MinMaxScaler(feature_range=(0, 1))
scaled_macd = scaler_macd.fit_transform(train_data['MACD'].values.reshape(-1, 1))

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(np.column_stack((scaled_price[i - numberOfBars:i, 0], 
                                    scaled_volume[i - numberOfBars:i, 0], 
                                    scaled_ema100[i - numberOfBars:i, 0],
                                    scaled_ema50[i - numberOfBars:i, 0],
                                    scaled_macd[i - numberOfBars:i, 0]
                                    )))
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)

#assert False

# One-hot encoding av målvariabel
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

# Oversampling før one-hot encoding
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(400, return_sequences=True, input_shape=(x_train_resampled.shape[1], 5)))  # Changed from 5 to 6 because of an extra feature
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(y_train_resampled.shape[1], activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_resampled, y_train_resampled, batch_size=1, epochs=6)
model.save(modelname)

# Lagrer scaler og encoder til disk
joblib.dump(scaler_price, get_full_path('scaler_price.pkl'))
joblib.dump(scaler_volume, get_full_path('scaler_volume.pkl'))
joblib.dump(scaler_ema50, get_full_path('scaler_ema50.pkl'))
joblib.dump(scaler_ema100, get_full_path('scaler_ema100.pkl'))
#joblib.dump(scaler_ema300, get_full_path('scaler_ema300.pkl'))
joblib.dump(scaler_macd, get_full_path('scaler_macd.pkl'))
joblib.dump(encoder, get_full_path('encoder.pkl'))

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

def get_full_path(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, filename)

# Last ned data
stock_data = pd.read_csv(get_full_path('juni_juli.csv'))
stock_data.set_index('Open Time', inplace=True)
modelname = get_full_path('btc_two.h5')
numberOfBars = 180

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].ewm(span=50).mean()
stock_data['EMA100'] = stock_data['Close'].ewm(span=200).mean()

print(stock_data.index[stock_data.index.duplicated()])
stock_data = stock_data[~stock_data.index.duplicated(keep='first')]

# Generere etiketter for klassifisering basert på toppene og dalene
min_prominence = 120
peak_indexes, _ = find_peaks(stock_data['Close'].values, prominence=min_prominence)
valley_indexes, _ = find_peaks(-stock_data['Close'].values, prominence=min_prominence)
stock_data.loc[stock_data.index[peak_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[valley_indexes], 'Signal'] = 'Buy'
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

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(np.column_stack((scaled_price[i - numberOfBars:i, 0], 
                                    scaled_volume[i - numberOfBars:i, 0], 
                                    scaled_ema50[i - numberOfBars:i, 0], 
                                    scaled_ema100[i - numberOfBars:i, 0]
                                    )))
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)

# One-hot encoding av målvariabel
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

# Oversampling før one-hot encoding
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

# Plott toppene, dalene, RSI og SMA100
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=stock_data.index[peak_indexes], y=stock_data['Close'].values[peak_indexes], mode='markers', name='Sell - Peaks'))
fig.add_trace(go.Scatter(x=stock_data.index[valley_indexes], y=stock_data['Close'].values[valley_indexes], mode='markers', name='Buy - Valleys'))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA100'], mode='lines', name='EMA100'))

# Oppdater layout for å legge til en sekundær y-akse for RSI
fig.update_layout(
    yaxis=dict(domain=[0.3, 1]),
    yaxis2=dict(title="RSI", titlefont=dict(color="blue"), tickfont=dict(color="blue"), overlaying='y', side='right', domain=[0, 0.3])
)

#fig.show()

# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], 4)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(y_train_resampled.shape[1], activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_resampled, y_train_resampled, batch_size=1, epochs=3)
model.save(modelname)

# Lagrer scaler og encoder til disk
joblib.dump(scaler_price, get_full_path('scaler_price.pkl'))
joblib.dump(scaler_volume, get_full_path('scaler_volume.pkl'))
joblib.dump(scaler_ema50, get_full_path('scaler_ema50.pkl'))
joblib.dump(scaler_ema100, get_full_path('scaler_ema100.pkl'))
joblib.dump(encoder, get_full_path('encoder.pkl'))

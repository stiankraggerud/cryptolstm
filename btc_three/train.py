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
stock_data = pd.read_csv(get_full_path('april_mai_juni_juli.csv'))
stock_data.set_index('Open Time', inplace=True)
modelname = get_full_path('btc_three.h5')
numberOfBars = 500

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA100'] = stock_data['Close'].ewm(span=100).mean()
stock_data['EMA200'] = stock_data['Close'].ewm(span=200).mean()
stock_data['EMA1400'] = stock_data['Close'].ewm(span=1400).mean()

# Beregne MACD
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

scaler_ema100 = MinMaxScaler(feature_range=(0, 1))
scaled_ema100 = scaler_ema100.fit_transform(train_data['EMA100'].values.reshape(-1, 1))

scaler_ema200 = MinMaxScaler(feature_range=(0, 1))
scaled_ema200 = scaler_ema200.fit_transform(train_data['EMA200'].values.reshape(-1, 1))

scaler_ema1400 = MinMaxScaler(feature_range=(0, 1))
scaled_ema1400 = scaler_ema1400.fit_transform(train_data['EMA1400'].values.reshape(-1, 1))

scaler_macd = MinMaxScaler(feature_range=(0, 1))
scaled_macd = scaler_macd.fit_transform(train_data['MACD'].values.reshape(-1, 1))

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(np.column_stack((scaled_price[i - numberOfBars:i, 0], 
                                    scaled_volume[i - numberOfBars:i, 0], 
                                    scaled_ema100[i - numberOfBars:i, 0], 
                                    scaled_ema200[i - numberOfBars:i, 0],
                                    scaled_ema1400[i - numberOfBars:i, 0],
                                    scaled_macd[i - numberOfBars:i, 0]
                                    )))
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)



# Oppretter et subplot med 2 rader og 1 kolonne
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Price and EMAs', 'MACD and Signal Line'))


# Tegn data på hovedplottet
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=stock_data.index[peak_indexes], y=stock_data['Close'].values[peak_indexes], mode='markers', name='Sell - Peaks'), row=1, col=1)
fig.add_trace(go.Scatter(x=stock_data.index[valley_indexes], y=stock_data['Close'].values[valley_indexes], mode='markers', name='Buy - Valleys'), row=1, col=1)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA100'], mode='lines', name='EMA100'), row=1, col=1)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA1400'], mode='lines', name='EMA1400'), row=1, col=1)

# Tegn MACD data på undergrafen
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='blue')), row=2, col=1)

fig.update_layout(title='Price, EMAs and MACD Analysis')

fig.show()


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
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], 6)))  # Changed from 5 to 6 because of an extra feature
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
joblib.dump(scaler_ema100, get_full_path('scaler_ema100.pkl'))
joblib.dump(scaler_ema200, get_full_path('scaler_ema200.pkl'))
joblib.dump(scaler_ema1400, get_full_path('scaler_ema1400.pkl'))
joblib.dump(scaler_macd, get_full_path('scaler_macd.pkl'))
joblib.dump(encoder, get_full_path('encoder.pkl'))
